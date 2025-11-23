# backend/main.py
"""
FastAPI backend for AutoML for Small Business

Endpoints:
- POST /upload-dataset (file) -> stores dataset and returns metadata
- POST /train (json: { "dataset": "<filename>", "target": "<col>", "engine": "<pycaret|autosklearn|flaml|sklearn>" })
    -> starts training in background, returns job id
- GET  /status -> returns current training status & metrics
- POST /predict (file) -> returns predictions using latest model
- GET  /download-model -> returns model file
- GET  /side-image -> returns the project's side image (local path provided)
"""

import os
import io
import time
import json
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"

# Ensure the folder exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Backend is running"}

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "status": "File uploaded successfully",
        "path": file_path
    }



# ML libs (try optional imports)
try:
    from flaml import AutoML as FLAML_AutoML
    _HAS_FLAML = True
except Exception:
    _HAS_FLAML = False

try:
    import autosklearn.classification as automl_class  # autosklearn
    _HAS_AUTOSKLEARN = True
except Exception:
    _HAS_AUTOSKLEARN = False

try:
    # pycaret may require heavy install; import optional
    from pycaret.classification import setup as py_setup, compare_models as py_compare, pull as py_pull
    _HAS_PYCARET = True
except Exception:
    _HAS_PYCARET = False

# sklearn fallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import joblib

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"
STATUS_FILE = MODELS_DIR / "status.json"
METRICS_FILE = MODELS_DIR / "metrics.json"

# Ensure directories exist
for p in (DATA_DIR, MODELS_DIR, TEMP_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Path to the side image the user uploaded earlier (local path from upload history)
# NOTE: this path includes quotes in the filename per the provided upload.
SIDE_IMAGE_PATH = '/mnt/data/A_flat-style_digital_illustration_showcases_"AutoM.png'

# Basic in-memory status (persisted to STATUS_FILE)
default_status = {
    "state": "IDLE",     # IDLE, TRAINING, COMPLETED, FAILED
    "engine": None,
    "dataset": None,
    "target": None,
    "started_at": None,
    "finished_at": None,
    "metrics": {}
}

if not STATUS_FILE.exists():
    with open(STATUS_FILE, "w") as f:
        json.dump(default_status, f, indent=2)

# Save status helper
def save_status(d):
    with open(STATUS_FILE, "w") as f:
        json.dump(d, f, indent=2)

def load_status():
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return default_status.copy()

# FastAPI app
app = FastAPI(title="AutoML Backend for Small Business (starter)")

# Allow CORS from Streamlit (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers
# ---------------------------
def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Dataset file not found.")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)

def save_uploaded_file_to(path: Path, upload_file: UploadFile):
    with open(path, "wb") as f:
        f.write(upload_file.file.read())

# Simple sklearn training pipeline (fallback)
def train_with_sklearn(dataset_path: Path, target_col: str):
    df = read_dataset(dataset_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Basic preprocessing for numeric data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="weighted"))

    # Save model
    model_path = MODELS_DIR / "model_sklearn.pkl"
    joblib.dump(pipeline, model_path)

    metrics = {"accuracy": acc, "f1_score": f1, "model_path": str(model_path)}
    return metrics

# FLAML wrapper
def train_with_flaml(dataset_path: Path, target_col: str, time_budget: int = 60):
    if not _HAS_FLAML:
        raise RuntimeError("FLAML is not installed.")
    df = read_dataset(dataset_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    automl = FLAML_AutoML()
    automl_settings = {
        "time_budget": time_budget,
        "metric": 'accuracy',
        "task": 'classification',
    }
    automl.fit(X_train=X, y_train=y, **automl_settings)
    # FLAML's automl.model is the best learner — save automl object
    model_path = MODELS_DIR / "model_flaml.pkl"
    joblib.dump(automl, model_path)
    # We can't evaluate here easily without separate holdout; return placeholder
    metrics = {"message": "FLAML ran", "model_path": str(model_path)}
    return metrics

# AutoSklearn wrapper (simplified)
def train_with_autosklearn(dataset_path: Path, target_col: str, time_left_for_this_task: int = 60):
    if not _HAS_AUTOSKLEARN:
        raise RuntimeError("autosklearn is not installed.")
    df = read_dataset(dataset_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    automl = automl_class.AutoSklearnClassifier(time_left_for_this_task=time_left_for_this_task)
    automl.fit(X_train, y_train)
    preds = automl.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="weighted"))
    model_path = MODELS_DIR / "model_autosklearn.pkl"
    joblib.dump(automl, model_path)
    metrics = {"accuracy": acc, "f1_score": f1, "model_path": str(model_path)}
    return metrics

# PyCaret wrapper (simplified)
def train_with_pycaret(dataset_path: Path, target_col: str, session_id: int = 42):
    if not _HAS_PYCARET:
        raise RuntimeError("pycaret is not installed.")
    df = read_dataset(dataset_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")
    # PyCaret expects a dataframe with the target column present
    # Note: pycaret setup creates many artifacts and requires an interactive env
    s = py_setup(data=df, target=target_col, silent=True, verbose=False, session_id=session_id)
    best = py_compare()
    # After compare_models you can pull metrics (simplified here)
    results = py_pull()
    model_path = MODELS_DIR / "model_pycaret.pkl"
    joblib.dump(best, model_path)
    metrics = {"pycaret_summary": results.to_dict(), "model_path": str(model_path)}
    return metrics

# Background training thread
def training_thread(dataset_filename: str, target_col: str, engine: str, extra: dict = None):
    status = load_status()
    status.update({
        "state": "TRAINING",
        "engine": engine,
        "dataset": dataset_filename,
        "target": target_col,
        "started_at": time.time(),
        "finished_at": None,
        "metrics": {}
    })
    save_status(status)

    dataset_path = DATA_DIR / dataset_filename
    try:
        if engine == "flaml":
            metrics = train_with_flaml(dataset_path, target_col, time_budget=extra.get("time_budget", 60))
        elif engine == "autosklearn":
            metrics = train_with_autosklearn(dataset_path, target_col, time_left_for_this_task=extra.get("time_budget", 60))
        elif engine == "pycaret":
            metrics = train_with_pycaret(dataset_path, target_col, session_id=extra.get("session_id", 42))
        else:
            # default sklearn
            metrics = train_with_sklearn(dataset_path, target_col)

        status.update({
            "state": "COMPLETED",
            "finished_at": time.time(),
            "metrics": metrics
        })
        save_status(status)
        # Also persist metrics independently
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        status.update({
            "state": "FAILED",
            "finished_at": time.time(),
            "metrics": {"error": str(e)}
        })
        save_status(status)

# ---------------------------
# API Models
# ---------------------------
class TrainRequest(BaseModel):
    dataset: str
    target: str
    engine: Optional[str] = "sklearn"  # pycaret, autosklearn, flaml, sklearn
    time_budget: Optional[int] = 60

# ---------------------------
# Endpoints
# ---------------------------

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset file (CSV or Excel). Saves to /datasets and returns filename & shape.
    """
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only CSV/XLSX files are accepted.")

    dest = DATA_DIR / file.filename
    # Prevent overwriting: add timestamp if exists
    if dest.exists():
        stamp = int(time.time())
        dest = DATA_DIR / f"{dest.stem}_{stamp}{dest.suffix}"
    save_uploaded_file_to(dest, file)
    # try to read and return basic info
    try:
        df = read_dataset(dest)
        return {"filename": dest.name, "rows": int(df.shape[0]), "columns": int(df.shape[1])}
    except Exception as e:
        return {"filename": dest.name, "warning": f"Saved but could not parse preview: {e}"}

@app.post("/train")

async def train(req: TrainRequest):
    """
    Start training in background. Provide dataset filename (previously uploaded), target column name, and optional engine.
    """
    dataset_file = DATA_DIR / req.dataset
    if not dataset_file.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first via /upload-dataset.")

    engine = req.engine.lower() if req.engine else "sklearn"
    if engine == "flaml" and not _HAS_FLAML:
        raise HTTPException(status_code=400, detail="FLAML not installed on server.")
    if engine == "autosklearn" and not _HAS_AUTOSKLEARN:
        raise HTTPException(status_code=400, detail="autosklearn not installed on server.")
    if engine == "pycaret" and not _HAS_PYCARET:
        raise HTTPException(status_code=400, detail="pycaret not installed on server.")

    # Start background thread
    thr = threading.Thread(target=training_thread, args=(req.dataset, req.target, engine, {"time_budget": req.time_budget}), daemon=True)
    thr.start()
    return {"message": "Training started", "engine": engine, "dataset": req.dataset}

@app.get("/status")
async def status():
    """
    Return current training status and metrics.
    """
    s = load_status()
    return s

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a CSV/XLSX, runs predictions using the latest model, and returns predictions JSON.
    """
    # Load latest model (choose any model file in MODELS_DIR)
    model_files = list(MODELS_DIR.glob("*.pkl"))
    if not model_files:
        raise HTTPException(status_code=404, detail="No trained model available. Train one first.")

    latest = max(model_files, key=os.path.getctime)
    model = joblib.load(latest)

    # Read input
    temp_path = TEMP_DIR / file.filename
    save_uploaded_file_to(temp_path, file)
    try:
        if temp_path.suffix.lower() == ".csv":
            df = pd.read_csv(temp_path)
        else:
            df = pd.read_excel(temp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}")

    # If model is a FLAML AutoML object, use its predict interface
    try:
        if _HAS_FLAML and hasattr(model, "predict"):
            preds = model.predict(df)
        else:
            preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Prepare output
    out = pd.DataFrame({"prediction": preds})
    out_json = out.to_dict(orient="records")
    # Save CSV to temp
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    csv_path = TEMP_DIR / f"predictions_{int(time.time())}.csv"
    with open(csv_path, "wb") as f:
        f.write(csv_bytes)

    return {"predictions": out_json, "download_csv": f"/download-temp/{csv_path.name}"}

@app.get("/download-model")
async def download_model():
    """
    Download the latest model file.
    """
    model_files = list(MODELS_DIR.glob("*.pkl"))
    if not model_files:
        raise HTTPException(status_code=404, detail="No model available. Train a model first.")
    latest = max(model_files, key=os.path.getctime)
    return FileResponse(path=latest, filename=latest.name, media_type="application/octet-stream")

@app.get("/download-temp/{filename}")
async def download_temp(filename: str):
    p = TEMP_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=p, filename=p.name, media_type="text/csv")

@app.get("/side-image")
async def side_image():
    """
    Serve the project's side image from the local path provided earlier.
    """
    p = Path(SIDE_IMAGE_PATH)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Side image not found at {SIDE_IMAGE_PATH}")
    return FileResponse(path=p, filename=p.name, media_type="image/png")
