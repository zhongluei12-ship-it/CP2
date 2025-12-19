import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# -----------------------
# Load data
# -----------------------
legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")

df = pd.concat([legitimate_df, phishing_df], axis=0)
df = df.drop_duplicates()
df = df.drop(columns=["URL"], errors="ignore")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

# -----------------------
# Train/test split (stratified)
# -----------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# Models
# -----------------------
models = {
    "SVM (LinearSVC)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(class_weight="balanced", random_state=42, max_iter=5000))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Neural Network (MLP)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=800,
            random_state=42
        ))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced"
    ),
    "Random Forest":  RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced_subsample"
    ),
    "AdaBoost": AdaBoostClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.5
    ),
    "GaussianNB": GaussianNB(),
}

# -----------------------
# Cross-validation (metrics table)
# -----------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}

rows = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    rows.append({
        "model": name,
        "accuracy": float(scores["test_accuracy"].mean()),
        "precision": float(scores["test_precision"].mean()),
        "recall": float(scores["test_recall"].mean()),
        "f1": float(scores["test_f1"].mean()),
    })

df_results = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

# -----------------------
# IMPORTANT: Fit models for Streamlit
# (cross_validate does NOT keep fitted models)
# -----------------------
fitted_models = {}
for name, est in models.items():
    fitted = clone(est)
    fitted.fit(x_train, y_train)
    fitted_models[name] = fitted

best_model_name = df_results.iloc[0]["model"]
best_model = fitted_models[best_model_name]

# Backwards-compatible names (your app can still use these)
svm_model = fitted_models["SVM (LinearSVC)"]
rf_model  = fitted_models["Random Forest"]
dt_model  = fitted_models["Decision Tree"]
ab_model  = fitted_models["AdaBoost"]
nb_model  = fitted_models["GaussianNB"]
nn_model  = fitted_models["Neural Network (MLP)"]
kn_model  = fitted_models["KNN"]
