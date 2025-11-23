import os
import json
import io
import time
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------
# Configuration / Paths
# ---------------------------
APP_DIR = Path.cwd()
DATA_DIR = APP_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = DATA_DIR / "models"
USERS_FILE = APP_DIR / "users.json"

# Default side image path
DEFAULT_SIDE_IMAGE = r"E:\New folder\Logo.png"

# Backend endpoints (if you have a backend, set these)
# If BASE_URL is empty, app will run in "local mode" (simulation).
BASE_URL = ""  # e.g. "http://localhost:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload-dataset" if BASE_URL else None
TRAIN_ENDPOINT = f"{BASE_URL}/train" if BASE_URL else None
STATUS_ENDPOINT = f"{BASE_URL}/status" if BASE_URL else None
PREDICT_ENDPOINT = f"{BASE_URL}/predict" if BASE_URL else None
DOWNLOAD_MODEL_ENDPOINT = f"{BASE_URL}/download-model" if BASE_URL else None

# Make sure directories exist
for p in (DATA_DIR, UPLOADS_DIR, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Ensure users file exists
if not USERS_FILE.exists():
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)


# ---------------------------
# Utility functions
# ---------------------------
def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


def save_uploaded_file(uploaded_file, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / uploaded_file.name
    # uploaded_file may be an UploadedFile (has getvalue())
    try:
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    except Exception:
        # fallback: read as bytes via buffer
        bytes_data = uploaded_file.read()
        with open(dest_path, "wb") as f:
            f.write(bytes_data)
    return dest_path


# ---------------------------
# Streamlit page config and styles
# ---------------------------
st.set_page_config(page_title="AutoML for Small Business", layout="wide", page_icon="⚡")

st.markdown(
    """
    <style>
    /* App background - Dark Brown */
    .stApp { background-color: #3E2723; }

    /* Headings & Text - Light/White */
    .title, h1, h2, h3, h4, h5, h6, p, div, span, label, li { 
        color: #ECEFF1 !important; 
        font-family: 'Helvetica Neue', Arial; 
    }

    /* Buttons */
    .stButton>button { 
        background-color: #8D6E63; 
        color: white; 
        border-radius: 6px; 
        padding: 8px 14px; 
        font-weight: 600; 
        border: none;
    }
    .stButton>button:hover { 
        background-color: #A1887F; 
    }

    /* Form card - Slightly lighter brown than bg */
    div[data-testid="stForm"] { 
        background-color: #4E342E; 
        padding: 18px; 
        border-radius: 10px; 
        box-shadow: 0 6px 18px rgba(0,0,0,0.5); 
        border: 1px solid #6D4C41; 
    }

    /* Narrow columns for centered forms */
    .center { display:flex; justify-content:center; }

    /* Input fields */
    .stTextInput>div>div>input { 
        color: #3E2723; 
        background-color: #D7CCC8; 
    }

    /* Radio buttons text */
    div[role="radiogroup"] label {
        color: #ECEFF1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Authentication UI
# ---------------------------
def signup_card():
    st.subheader("Create your account")
    with st.form("signup_form"):
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First name")
            email = st.text_input("Email")
        with col2:
            last_name = st.text_input("Last name")
            username = st.text_input("Username (optional)")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Sign up")
        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
                return
            if password != password2:
                st.error("Passwords do not match.")
                return
            users = load_users()
            if email in users:
                st.error("Email already registered. Please login.")
                return
            users[email] = {
                "first_name": first_name,
                "last_name": last_name,
                "username": username or email.split("@")[0],
                "password_hash": hash_password(password),
                "created_at": time.time()
            }
            save_users(users)
            st.success("Account created. Please log in.")


def login_card():
    st.subheader("Login to your account")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
                return
            users = load_users()
            if email not in users:
                st.error("No account found for this email.")
                return
            if not verify_password(password, users[email]["password_hash"]):
                st.error("Incorrect password.")
                return
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.session_state.user_name = users[email].get("first_name") or users[email].get("username") or email
            st.success(f"Welcome, {st.session_state.user_name}!")
            st.rerun()


# ---------------------------
# Layout: left column = side image; right column = form/content
# ---------------------------
def show_auth_page(side_image_path: str = DEFAULT_SIDE_IMAGE):
    left_col, right_col = st.columns([1, 1.2])
    # side image column
    with left_col:
        st.markdown("<div style='padding:10px;'>", unsafe_allow_html=True)
        try:
            img = Image.open(side_image_path)
            st.image(img, use_column_width=True)
        except Exception:
            st.info(f"Side image not found at {side_image_path}. Please check the path.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.title("Create Your Account" if auth_option == "Sign Up" else "Welcome Back")
        if auth_option == "Sign Up":
            signup_card()
            st.markdown("Already have an account? Switch to Login above.")
        else:
            login_card()
            st.markdown("Don't have an account? Switch to Sign Up above.")


# ---------------------------
# After-login dashboard pages
# ---------------------------
def header_after_login():
    st.markdown(
        f"<h2 style='text-align:center'>⚡ AutoML Dashboard — Welcome {st.session_state.get('user_name', '')}</h2>",
        unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#BCAAA4'>Step-by-step: Upload → Train → Predict → Model</p>",
                unsafe_allow_html=True)
    st.markdown("---")


def page_upload_dataset():
    st.subheader("Step 1 — Upload Dataset")
    st.markdown("Upload a CSV or Excel dataset. The app will store the file locally and show a preview.")
    uploaded_file = st.file_uploader("Choose dataset", type=["csv", "xlsx", "xls"], key="upload")
    if uploaded_file:
        # preview
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")
            return
        st.markdown("**Dataset preview (first 10 rows)**")
        st.dataframe(df.head(10))
        st.info(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
        if st.button("Save dataset locally", use_container_width=True):
            path = save_uploaded_file(uploaded_file, UPLOADS_DIR)
            st.success(f"Saved to: {path}")
            st.write("You can now proceed to Train Model.")


def simulate_training_process():
    # Simulated training progress + fake metrics
    with st.spinner("Training (simulated) — this will take a few seconds..."):
        progress = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.02)
            progress.progress(i)
    # Create a fake model file marker
    model_file = MODELS_DIR / "automl_model_v1.pkl"
    with open(model_file, "wb") as f:
        f.write(b"SIMULATED_MODEL")
    # Simulated metrics
    metrics = {
        "status": "COMPLETED",
        "accuracy": round(0.7 + 0.25 * (time.time() % 1), 3),
        "f1_score": round(0.65 + 0.2 * (time.time() % 1), 3),
        "trained_at": time.time()
    }
    # Save metrics locally
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics, str(model_file)


def page_train_model():
    st.subheader("Step 2 — Train Model")
    st.markdown(
        "Trigger training. If you have a backend, the app will call it; otherwise training runs in local simulated mode.")
    col1, col2 = st.columns([1, 1])
    if BASE_URL:
        if col1.button("Start Training (via backend)", use_container_width=True):
            st.info(
                "This app is configured to call a backend. Implement backend endpoints and set BASE_URL in the code.")
    else:
        if col1.button("Start Training (local simulation)", use_container_width=True):
            metrics, model_path = simulate_training_process()
            st.success("Training completed (simulated).")
            st.json(metrics)
            st.write("Model saved to:", model_path)

    if col2.button("Check latest training metrics", use_container_width=True):
        metrics_file = MODELS_DIR / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            st.json(metrics)
        else:
            st.info("No training run found yet. Run training first.")


def page_predict():
    st.subheader("Step 3 — Predict")
    st.markdown("Upload new data to generate predictions. In local mode the app returns simulated predictions.")
    predict_upload = st.file_uploader("Upload data for prediction", type=["csv", "xlsx", "xls"], key="predict_uploader")
    if predict_upload:
        try:
            if predict_upload.name.lower().endswith(".csv"):
                df = pd.read_csv(predict_upload)
            else:
                df = pd.read_excel(predict_upload)
            st.write("Preview of input data:")
            st.dataframe(df.head(5))
        except Exception as e:
            st.error(f"Unable to read file: {e}")
            return

        if BASE_URL:
            st.info("Prediction via backend not implemented in this example. Set up backend endpoints and enable URL.")
        else:
            if st.button("Run prediction (local simulated)", use_container_width=True):
                # Simulate predictions
                n = min(50, len(df))
                predictions = []
                for i in range(n):
                    predictions.append({
                        "index": i,
                        "prediction": "class_A" if i % 2 == 0 else "class_B",
                        "score": round(0.5 + (i % 10) * 0.05, 3)
                    })
                out_df = pd.DataFrame(predictions)
                st.success("Simulated predictions complete.")
                st.dataframe(out_df.head(20))

                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")


def page_models_and_reports():
    st.subheader("Step 4 — Model & Reports")
    st.markdown("Download the trained model or view training reports.")
    model_files = list(MODELS_DIR.glob("*"))
    if model_files:
        for m in model_files:
            st.write("-", m.name)
        if st.button("Download latest model", use_container_width=True):
            latest = max(model_files, key=os.path.getctime)
            with open(latest, "rb") as f:
                bytes_data = f.read()
            st.download_button("Save Model", data=bytes_data, file_name=latest.name, mime="application/octet-stream")
    else:
        st.info("No model files found. Run training first.")

    metrics_file = MODELS_DIR / "metrics.json"
    if metrics_file.exists():
        st.write("Latest training metrics:")
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        st.json(metrics)
    else:
        st.info("No training metrics available yet.")


# ---------------------------
# Main app
# ---------------------------
# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# Top-level: if not logged in -> show auth options
if not st.session_state.logged_in:
    # show large centered header and radio to choose auth option
    st.markdown("<h1 class='title' style='text-align:center'>Welcome to AutoML for Small Business</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#BCAAA4'>Create an account or login to continue.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    # Authentication choice
    cols = st.columns([1, 1, 1])
    with cols[1]:
        auth_choice = st.radio("", ["Login", "Sign Up"], horizontal=True)
    # global for layout
    auth_option = auth_choice

    # Put the auth UI next to side image
    show_auth_page(side_image_path=DEFAULT_SIDE_IMAGE)
    st.markdown("---")
    st.info("Tip: After signing up, login with the same email to enter the AutoML Dashboard.")
else:
    # Sidebar: show user and logout
    st.sidebar.success(f"Logged in as: {st.session_state.user_name}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.user_name = ""
        st.rerun()

    header_after_login()

    # Step navigation in sidebar
    step = st.sidebar.radio("Step Navigation",
                            ["Upload Dataset", "Train Model", "Predict", "Model & Reports", "Settings"])

    if step == "Upload Dataset":
        page_upload_dataset()
    elif step == "Train Model":
        page_train_model()
    elif step == "Predict":
        page_predict()
    elif step == "Model & Reports":
        page_models_and_reports()
    elif step == "Settings":
        st.subheader("Settings")
        st.markdown("Configure app-level options and view local files.")
        st.write("Base URL (for backend):", BASE_URL or "Not configured (local mode)")
        st.write("Uploads folder:", UPLOADS_DIR)
        st.write("Models folder:", MODELS_DIR)
        st.write("Users file:", USERS_FILE)
        st.markdown("---")
        st.markdown("Change side image (optional)")
        img_choice = st.file_uploader("Upload a new side image (will be saved in app folder)",
                                      type=["png", "jpg", "jpeg"])
        if img_choice:
            dest = APP_DIR / "side_uploaded.jpg"
            with open(dest, "wb") as f:
                f.write(img_choice.getvalue())
            st.success(f"Saved side image to {dest}. Edit DEFAULT_SIDE_IMAGE in code to use it.")