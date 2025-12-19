import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from feature_extraction import COLUMNS as FE_COLUMNS

from bs4 import BeautifulSoup
import requests as re
import pandas as pd
from urllib.parse import urlparse
import datetime
import time
from sklearn.exceptions import NotFittedError

# ---------------------------
# PAGE SETTINGS
# ---------------------------
st.set_page_config(page_title="Phishing Website Detector", layout="wide")

# ---------------------------
# SESSION STATE
# ---------------------------
if "url_input" not in st.session_state:
    st.session_state["url_input"] = ""

if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts: url, model, result, confidence

# ---------------------------
# REQUEST SETTINGS (reduce 403)
# ---------------------------
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2
RETRY_DELAY = 0.7

# feature names used in the model (remove URL and label from your COLUMNS)
FEATURE_NAMES = [c for c in FE_COLUMNS if c not in ("URL", "label")]

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def convert_df(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def safe_predict(model, vector):
    """Predict safely even if model has issues."""
    try:
        pred = model.predict(vector)[0]
    except NotFittedError:
        return None, None
    except Exception:
        return None, None

    proba_phishing = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(vector)[0]
            if hasattr(model, "classes_") and 1 in model.classes_:
                idx = list(model.classes_).index(1)
                proba_phishing = float(proba[idx])
            else:
                proba_phishing = float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception:
            proba_phishing = None

    return int(pred), proba_phishing


def build_feature_df(vector):
    """Build a one-row DataFrame of features for display."""
    try:
        return pd.DataFrame([vector[0]], columns=FEATURE_NAMES)
    except Exception:
        return pd.DataFrame([vector[0]], columns=[f"feature_{i}" for i in range(len(vector[0]))])


def simple_explanation(feature_row: pd.Series) -> list:
    reasons = []

    if "has_password" in feature_row.index and feature_row["has_password"] == 1:
        reasons.append("The page contains password input fields.")

    if "has_form" in feature_row.index and "number_of_inputs" in feature_row.index:
        if feature_row["has_form"] == 1 and feature_row["number_of_inputs"] > 10:
            reasons.append("The page has forms with many input fields.")

    if "has_iframe" in feature_row.index and feature_row["has_iframe"] == 1:
        reasons.append("The page uses iframes, which can be abused in phishing pages.")

    if "length_of_text" in feature_row.index and feature_row["length_of_text"] < 50:
        reasons.append("The visible text on the page is very short, which can be suspicious.")

    if "has_image" in feature_row.index and "number_of_images" in feature_row.index:
        if feature_row["has_image"] == 1 and feature_row["number_of_images"] > 20:
            reasons.append("The page has many images, possibly to mimic a brand interface.")

    if not reasons:
        reasons.append(
            "No obvious red-flag features were detected; the decision is based on subtler structural patterns."
        )

    return reasons


def url_heuristics(url: str) -> dict:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()

    shorteners = {"bit.ly", "ln.run", "tinyurl.com", "t.co", "goo.gl", "is.gd"}
    suspicious_words = {
        "login", "secure", "account", "verify", "update",
        "password", "bank", "confirm", "signin"
    }

    return {
        "url_length": len(url),
        "host_length": len(host),
        "num_dots_in_host": host.count("."),
        "has_at_sign": "@" in url,
        "has_dash_in_host": "-" in host,
        "path_length": len(path),
        "scheme": parsed.scheme,

        # extra indicators
        "uses_https": parsed.scheme == "https",
        "looks_like_shortener": host in shorteners,
        "has_ip_address": host.replace(".", "").isdigit(),
        "suspicious_words_in_path": any(w in path for w in suspicious_words),
    }


def url_only_score(url: str) -> tuple[str, float]:
    """
    URL-only fallback scoring.
    Returns (label, confidence_phishing_estimate).
    """
    h = url_heuristics(url)
    score = 0

    if h["url_length"] > 75:
        score += 2
    if h["num_dots_in_host"] >= 4:
        score += 2
    if h["has_at_sign"]:
        score += 3
    if h["has_dash_in_host"]:
        score += 1
    if not h["uses_https"]:
        score += 2
    if h["looks_like_shortener"]:
        score += 2
    if h["has_ip_address"]:
        score += 3
    if h["suspicious_words_in_path"]:
        score += 2

    confidence = min(0.95, 0.20 + score * 0.10)
    label = "Phishing" if confidence >= 0.50 else "Legitimate"
    return label, confidence


def get_cert_info(url: str):
    import ssl
    import socket

    if not url.lower().startswith("https://"):
        return {"error": "Certificate info only works for HTTPS URLs."}

    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return {"error": "Could not parse hostname from URL."}

        port = parsed.port or 443
        ctx = ssl.create_default_context()

        with socket.create_connection((host, port), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()

        not_after = cert.get("notAfter")
        issuer = dict(x[0] for x in cert.get("issuer", []))
        subject = dict(x[0] for x in cert.get("subject", []))

        expiry = datetime.datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
        days_left = (expiry - datetime.datetime.utcnow()).days

        return {
            "subject_common_name": subject.get("commonName"),
            "issuer_common_name": issuer.get("commonName"),
            "not_after": not_after,
            "days_left": days_left,
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_page(url: str):
    """
    Fetch HTML with headers + retries + redirects.
    Returns (status_code, content, final_url, error_string)
    """
    last_err = None
    session = re.Session()
    session.headers.update(REQUEST_HEADERS)

    for _ in range(MAX_RETRIES + 1):
        try:
            resp = session.get(
                url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=False,
            )
            return resp.status_code, resp.content, resp.url, None
        except Exception as e:
            last_err = str(e)
            time.sleep(RETRY_DELAY)

    return None, None, url, last_err


def handle_fallback(final_url: str, reason: str):
    """
    Single safe fallback path:
    - show warning + heuristics
    - compute fallback_label + fallback_conf
    - stop execution (so variables are never used undefined)
    """
    st.warning(reason)

    st.markdown("### 🔍 URL heuristics")
    st.json(url_heuristics(final_url))

    fallback_label, fallback_conf = url_only_score(final_url)

    st.markdown("### 🔐 Prediction Result (URL-only fallback)")
    if fallback_label == "Phishing":
        st.error(f"**Prediction:** {fallback_label} (fallback)")
    else:
        st.success(f"**Prediction:** {fallback_label} (fallback)")

    st.write(f"**Estimated confidence (phishing):** {fallback_conf:.2%}")
    st.progress(min(max(fallback_conf, 0.0), 1.0))

    st.caption("Note: HTML could not be analyzed, so detection is based on URL patterns only.")
    st.stop()


def get_model(model_name: str):
    """
    Map UI names -> fitted_models keys (machine_learning.py),
    fallback to old exported variables if fitted_models isn't available.
    """
    if hasattr(ml, "fitted_models") and isinstance(ml.fitted_models, dict):
        name_map = {
            "Gaussian Naive Bayes": "GaussianNB",
            "Support Vector Machine": "SVM (LinearSVC)",
            "Decision Tree": "Decision Tree",
            "Random Forest": "Random Forest",
            "AdaBoost": "AdaBoost",
            "Neural Network": "Neural Network (MLP)",
            "K-Neighbours": "KNN",
        }
        key = name_map.get(model_name, model_name)
        if key in ml.fitted_models:
            return ml.fitted_models[key]
        raise KeyError(f"Model '{model_name}' not found. Available: {list(ml.fitted_models.keys())}")

    # fallback old exports
    mapping = {
        "Gaussian Naive Bayes": ml.nb_model,
        "Support Vector Machine": ml.svm_model,
        "Decision Tree": ml.dt_model,
        "Random Forest": ml.rf_model,
        "AdaBoost": ml.ab_model,
        "Neural Network": ml.nn_model,
        "K-Neighbours": ml.kn_model,
    }
    return mapping[model_name]


# ---------------------------
# OPTIONAL: FEATURE IMPORTANCE (if exists)
# ---------------------------
st.subheader("🌲 Random Forest Feature Importance")
if hasattr(ml, "rf_importance"):
    st.dataframe(ml.rf_importance.head(15), use_container_width=True)
    st.bar_chart(ml.rf_importance.set_index("feature").head(15))
else:
    st.caption("Feature importance not available in machine_learning.py (optional).")

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    model_options = [
        "Gaussian Naive Bayes",
        "Support Vector Machine",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "Neural Network",
        "K-Neighbours",
    ]

    choice = st.selectbox("Select a model:", model_options)
    model = get_model(choice)
    st.success(f"✔️ Selected Model: **{choice}**")

    st.markdown("---")
    st.markdown("### ℹ️ About this app")
    st.write(
        "This tool analyzes a webpage's **HTML structure + URL features** using machine learning "
        "to predict whether it is **Legitimate** or **Phishing**."
    )

    st.markdown("---")
    st.markdown("### 🔍 Quick test URLs")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Apple (Legit)"):
            st.session_state["url_input"] = "https://www.apple.com/"
    with col_b:
        if st.button("phishing (Short URL)"):
            st.session_state["url_input"] = "https://bit.ly/41OnpuQ"

    st.caption("Some shorteners block bots and return 403. HTML upload mode can help.")


# ---------------------------
# MAIN LAYOUT
# ---------------------------
st.title("🛡️ Phishing Website Detector")
st.write(
    "Enter a URL below or upload an HTML file. The system will analyze the page structure, "
    "extract features, and use a machine learning model trained on legitimate & phishing websites "
    "to classify it."
)

tab_detect, tab_data = st.tabs(["🔎 Detector", "📚 Dataset & Models"])

# ============================================================
# TAB 1: DETECTOR
# ============================================================
with tab_detect:
    st.subheader("🔗 Analyze by URL")

    url = st.text_input(
        "Enter URL to analyze:",
        value=st.session_state.get("url_input", ""),
        placeholder="e.g. https://example.com/login",
    ).strip()

    analyze_btn = st.button("Check!", use_container_width=True, key="check_url_btn")
    result_box = st.empty()

    if analyze_btn:
        with result_box.container():
            if not url:
                st.error("Please enter a URL before clicking **Check!**")
            else:
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url

                st.session_state["url_input"] = url

                try:
                    with st.spinner("Analyzing webpage content..."):
                        status, content, final_url, err = fetch_page(url)

                    st.write(f"HTTP Status: **{status}**")
                    if final_url != url:
                        st.caption(f"Redirected to: {final_url}")

                    # 1) Request-level error -> fallback
                    if err:
                        handle_fallback(final_url, f"Request error: {err}")

                    # 2) Blocked / empty / non-200 -> fallback
                    if status != 200 or not content or len(content) < 200:
                        handle_fallback(final_url, "⚠️ Cannot fetch usable HTML content (blocked / redirected / requires JS).")

                    # 3) We have HTML -> proceed with ML
                    soup = BeautifulSoup(content, "html.parser")

                    # IMPORTANT: use final_url so URL features match destination
                    vector = [fe.create_vector(soup, final_url)]

                    pred, proba_phishing = safe_predict(model, vector)
                    if pred is None:
                        st.error("Model could not predict (not fitted or error). Falling back to URL-only.")
                        handle_fallback(final_url, "Model prediction failed; using URL-only fallback.")

                    prediction_label = "Phishing" if pred == 1 else "Legitimate"

                    st.markdown("---")
                    st.subheader("🔐 Prediction Result")

                    if pred == 1:
                        st.error(f"**Prediction:** {prediction_label}")
                    else:
                        st.success(f"**Prediction:** {prediction_label}")

                    if proba_phishing is not None:
                        st.write(f"**Model confidence (phishing):** {proba_phishing:.2%}")
                        st.progress(min(max(proba_phishing, 0.0), 1.0))
                    else:
                        st.caption("Confidence score is not available for this model.")

                    # store history
                    st.session_state["history"].append(
                        {
                            "url": final_url,
                            "model": choice,
                            "result": prediction_label,
                            "confidence_phishing": proba_phishing,
                        }
                    )
                    st.session_state["history"] = st.session_state["history"][-10:]

                    st.markdown("### 📊 Features used by the model")
                    feat_df = build_feature_df(vector)
                    with st.expander("Show feature values", expanded=False):
                        st.dataframe(feat_df.T.rename(columns={0: "value"}), use_container_width=True)

                    st.markdown("### 🧩 Simple explanation")
                    with st.expander("Why did the model make this decision?", expanded=True):
                        row = feat_df.iloc[0]
                        for r in simple_explanation(row):
                            st.write(f"- {r}")

                    st.markdown("### 🔍 URL heuristics")
                    with st.expander("Show URL risk indicators", expanded=False):
                        st.json(url_heuristics(final_url))

                    st.markdown("### 🔐 SSL / Certificate info")
                    with st.expander("Show certificate details", expanded=False):
                        st.json(get_cert_info(final_url))

                    with st.expander("🛠 Technical details"):
                        st.write("Raw feature vector:")
                        st.write(vector)
                        st.write("First 500 characters of cleaned HTML (for inspection):")
                        st.code(soup.get_text(separator=" ", strip=True)[:500])

                except re.exceptions.Timeout:
                    st.error("❌ Request Timeout: The server took too long to respond.")
                except re.exceptions.ConnectionError:
                    st.error("❌ Connection Error: Could not connect to the URL.")
                except Exception as e:
                    # show real traceback in streamlit
                    st.exception(e)

    st.markdown("---")
    st.subheader("📂 Analyze by HTML file (offline)")

    uploaded = st.file_uploader("Upload an HTML file", type=["html", "htm"], key="html_uploader")
    if uploaded is not None:
        if st.button("Analyze uploaded HTML", use_container_width=True, key="check_html_btn"):
            try:
                soup = BeautifulSoup(uploaded.read(), "html.parser")

                # offline: keep url empty (url features become minimal/default)
                vector = [fe.create_vector(soup, "")]

                pred, proba_phishing = safe_predict(model, vector)
                if pred is None:
                    st.error("Model could not predict (not fitted or error).")
                    st.stop()

                prediction_label = "Phishing" if pred == 1 else "Legitimate"

                st.markdown("---")
                st.subheader("🔐 Prediction Result (Uploaded HTML)")

                if pred == 1:
                    st.error(f"**Prediction:** {prediction_label}")
                else:
                    st.success(f"**Prediction:** {prediction_label}")

                if proba_phishing is not None:
                    st.write(f"**Model confidence (phishing):** {proba_phishing:.2%}")
                    st.progress(min(max(proba_phishing, 0.0), 1.0))
                else:
                    st.caption("Confidence score is not available for this model.")

                st.markdown("### 📊 Features used by the model")
                feat_df = build_feature_df(vector)
                with st.expander("Show feature values", expanded=False):
                    st.dataframe(feat_df.T.rename(columns={0: "value"}), use_container_width=True)

            except Exception as e:
                st.exception(e)

    st.markdown("---")
    st.subheader("🕒 Recent Checks (this session)")

    if st.session_state["history"]:
        st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True)
    else:
        st.caption("No URLs checked yet.")

# ============================================================
# TAB 2: DATASET & MODELS
# ============================================================
with tab_data:
    st.subheader("📊 Dataset Overview")

    # these exist in your machine_learning.py
    legit_count = ml.legitimate_df.shape[0]
    phishing_count = ml.phishing_df.shape[0]
    total_count = legit_count + phishing_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Legitimate samples", legit_count)
    col2.metric("Phishing samples", phishing_count)
    col3.metric("Total samples", total_count)

    st.markdown("### Preview of Structured Data")
    row_limit = st.slider("Number of rows to preview from each class:", 5, 50, 10)

    st.write("**Legitimate samples:**")
    st.dataframe(ml.legitimate_df.head(row_limit), use_container_width=True)

    st.write("**Phishing samples:**")
    st.dataframe(ml.phishing_df.head(row_limit), use_container_width=True)

    st.markdown("### ⬇ Download Combined Dataset")
    combined_df = ml.df
    csv_bytes = convert_df(combined_df)
    st.download_button(
        label="Download full dataset as CSV",
        data=csv_bytes,
        file_name="structured_dataset_full.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("🤖 Model Performance")

    st.write("Accuracy, precision and recall for each model used (evaluated on your CV/test split).")
    st.table(ml.df_results)
    st.bar_chart(ml.df_results[["accuracy", "precision", "recall"]])

    st.markdown("---")
    st.subheader("📈 Feature distribution explorer")

    feature_to_plot = st.selectbox("Choose a feature to compare (legit vs phishing):", FEATURE_NAMES)
    try:
        grouped = ml.df.groupby("label")[feature_to_plot].mean()
        st.bar_chart(grouped)
        st.caption("Mean value of the selected feature for each class (0 = Legitimate, 1 = Phishing).")
    except Exception as e:
        st.write("Could not plot this feature.")
        st.caption(str(e))
