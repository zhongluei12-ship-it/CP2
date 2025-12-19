"""
feature_extraction.py

All HTML -> numeric feature logic lives here.
Used by:
 - app.py          (for live URL prediction)
 - dataset_builder (for offline CSV generation)
"""

from bs4 import BeautifulSoup
from urllib.parse import urlparse

# --------------------------------------
# 1. Feature column names (no URL/label)
# --------------------------------------
COLUMNS = [
    'has_title', 'has_input', 'has_button', 'has_image', 'has_submit', 'has_link',
    'has_password', 'has_email_input', 'has_hidden_element', 'has_audio', 'has_video',
    'number_of_inputs', 'number_of_buttons', 'number_of_images', 'number_of_option',
    'number_of_list', 'number_of_th', 'number_of_tr', 'number_of_href',
    'number_of_paragraph', 'number_of_script', 'length_of_title',
    'has_h1', 'has_h2', 'has_h3', 'length_of_text',
    'number_of_clickable_button', 'number_of_a', 'number_of_img', 'number_of_div',
    'number_of_figure', 'has_footer', 'has_form', 'has_text_area',
    'has_iframe', 'has_text_input', 'number_of_meta', 'has_nav',
    'has_object', 'has_picture', 'number_of_sources', 'number_of_span',
    'number_of_table',

    # --- URL lexical features ---
    'url_length', 'host_length', 'num_dots_host', 'has_at_sign',
    'has_dash_host', 'path_length', 'is_https', 'is_shortener'
]

SHORTENERS = {"bit.ly", "ln.run", "tinyurl.com", "t.co", "goo.gl", "is.gd"}

# --------------------------------------
# 2. URL feature extractor (SAFE)
# --------------------------------------
def url_features(url: str) -> list[int]:
    """
    Always returns EXACTLY 8 URL features
    (must match URL columns in COLUMNS)
    """
    if not isinstance(url, str) or not url.strip():
        return [0] * 8

    url = url.strip()
    p = urlparse(url)
    host = (p.hostname or "").lower()
    path = p.path or ""

    return [
        len(url),                      # url_length
        len(host),                     # host_length
        host.count("."),               # num_dots_host
        int("@" in url),               # has_at_sign
        int("-" in host),              # has_dash_host
        len(path),                     # path_length
        int(p.scheme == "https"),      # is_https
        int(host in SHORTENERS),       # is_shortener
    ]

# --------------------------------------
# 3. HTML feature functions
# --------------------------------------
def has_title(soup): return int(bool(soup.title and soup.title.text.strip()))
def has_input(soup): return int(bool(soup.find('input')))
def has_button(soup): return int(bool(soup.find('button')))
def has_image(soup): return int(bool(soup.find('img')))
def has_submit(soup): return int(bool(soup.find('input', {'type': 'submit'}) or soup.find('button', {'type': 'submit'})))
def has_link(soup): return int(bool(soup.find('a')))
def has_password(soup): return int(bool(soup.find('input', {'type': 'password'})))
def has_email_input(soup): return int(bool(soup.find('input', {'type': 'email'})))
def has_hidden_element(soup): return int(bool(soup.find('input', {'type': 'hidden'})))
def has_audio(soup): return int(bool(soup.find('audio')))
def has_video(soup): return int(bool(soup.find('video')))

def number_of_inputs(soup): return len(soup.find_all('input'))
def number_of_buttons(soup): return len(soup.find_all('button'))
def number_of_images(soup): return len(soup.find_all('img'))
def number_of_option(soup): return len(soup.find_all('option'))
def number_of_list(soup): return len(soup.find_all(['ul', 'ol', 'li']))
def number_of_th(soup): return len(soup.find_all('th'))
def number_of_tr(soup): return len(soup.find_all('tr'))
def number_of_href(soup): return len([a for a in soup.find_all('a') if a.get('href')])
def number_of_paragraph(soup): return len(soup.find_all('p'))
def number_of_script(soup): return len(soup.find_all('script'))

def length_of_title(soup):
    return len(soup.title.text.strip()) if soup.title and soup.title.text else 0

def has_h1(soup): return int(bool(soup.find('h1')))
def has_h2(soup): return int(bool(soup.find('h2')))
def has_h3(soup): return int(bool(soup.find('h3')))
def length_of_text(soup): return len(soup.get_text(separator=" ", strip=True))

def number_of_clickable_button(soup):
    return sum(1 for btn in soup.find_all('button') if btn.get_text(strip=True))

def number_of_a(soup): return len(soup.find_all('a'))
def number_of_img(soup): return len(soup.find_all('img'))
def number_of_div(soup): return len(soup.find_all('div'))
def number_of_figure(soup): return len(soup.find_all('figure'))
def has_footer(soup): return int(bool(soup.find('footer')))
def has_form(soup): return int(bool(soup.find('form')))
def has_text_area(soup): return int(bool(soup.find('textarea')))
def has_iframe(soup): return int(bool(soup.find('iframe')))
def has_text_input(soup): return int(bool(soup.find('input', {'type': 'text'}) or soup.find('input', {'type': None})))
def number_of_meta(soup): return len(soup.find_all('meta'))
def has_nav(soup): return int(bool(soup.find('nav')))
def has_object(soup): return int(bool(soup.find('object')))
def has_picture(soup): return int(bool(soup.find('picture')))
def number_of_sources(soup): return len(soup.find_all('source'))
def number_of_span(soup): return len(soup.find_all('span'))
def number_of_table(soup): return len(soup.find_all('table'))

# --------------------------------------
# 4. Vector builder (HTML + URL)
# --------------------------------------
def create_vector(soup: BeautifulSoup, url: str = "") -> list[int]:
    """
    Returns feature vector in EXACT order of COLUMNS
    """
    html_features = [
        has_title(soup), has_input(soup), has_button(soup), has_image(soup),
        has_submit(soup), has_link(soup), has_password(soup), has_email_input(soup),
        has_hidden_element(soup), has_audio(soup), has_video(soup),
        number_of_inputs(soup), number_of_buttons(soup), number_of_images(soup),
        number_of_option(soup), number_of_list(soup), number_of_th(soup),
        number_of_tr(soup), number_of_href(soup), number_of_paragraph(soup),
        number_of_script(soup), length_of_title(soup),
        has_h1(soup), has_h2(soup), has_h3(soup),
        length_of_text(soup), number_of_clickable_button(soup),
        number_of_a(soup), number_of_img(soup), number_of_div(soup),
        number_of_figure(soup), has_footer(soup), has_form(soup),
        has_text_area(soup), has_iframe(soup), has_text_input(soup),
        number_of_meta(soup), has_nav(soup), has_object(soup),
        has_picture(soup), number_of_sources(soup),
        number_of_span(soup), number_of_table(soup),
    ]

    return html_features + url_features(url)
