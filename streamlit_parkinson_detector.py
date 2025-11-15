"""
Notes:
 - The app will try to load these exact filenames from ./models/. If a file is missing or fails to load,
   the app falls back to a safe demo predictor for that modality so the UI never errors.
 - This file is GitHub-ready. After you add the models into models/ and add requirements.txt,
   push to your repo and deploy on Streamlit Cloud (https://share.streamlit.io/).

Dependencies (example requirements.txt contents):
streamlit
tensorflow
torch
timm
pillow
numpy
pandas
scikit-learn
joblib
librosa
soundfile
matplotlib

"""

import streamlit as st
from PIL import Image
import io
import os
import numpy as np
import pandas as pd
import joblib
import tempfile
import base64

# TF for Keras models
import tensorflow as tf

# audio
import librosa

# visual
import matplotlib.pyplot as plt

# Ensure models dir
os.makedirs('models', exist_ok=True)

st.set_page_config(page_title="Ultimate Parkinson's Detector (Final)", layout='wide')

# Model filenames (as confirmed)
MRI_MODEL = 'models/mri_model.h5'
DRAWING_MODEL = 'models/drawing_model.h5'
GAIT_MODEL = 'models/gait_model.pkl'
SPEECH_MODEL = 'models/speech_model.pkl'

# ---------- Utility helpers ----------

@st.cache_resource
def load_keras_model_safe(path):
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            return model
        except Exception as e:
            st.warning(f"Failed to load Keras model at {path}: {e}")
            return None
    return None

@st.cache_resource
def load_joblib_model_safe(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load joblib model at {path}: {e}")
            return None
    return None

# Load models (attempt)
mri_model = load_keras_model_safe(MRI_MODEL)
mri_ann = load_keras_model_safe(MRI_ANN)
drawing_model = load_keras_model_safe(DRAWING_MODEL)
gait_model = load_keras_model_safe(GAIT_MODEL)
speech_model = load_joblib_model_safe(SPEECH_MODEL)

# Demo fallbacks
class DemoImageModel:
    def predict_proba(self, arr):
        mean = float(arr.mean())/255.0
        p = 0.15 + 0.7*mean
        return np.array([[1-p, p]])

class DemoTabularModel:
    def predict_proba(self, X):
        v = X.std(axis=1).mean() if hasattr(X, 'std') else 0.2
        p = 0.2 + 0.6*(1.0/(1.0+np.exp(-(v-0.1)*8)))
        return np.vstack([1-p, p]).T

demo_img = DemoImageModel()
demo_tab = DemoTabularModel()

# Preprocessing helpers
def preprocess_image_for_keras(img: Image.Image, target_shape=(224,224)):
    img = img.convert('RGB')
    img = img.resize((target_shape[1], target_shape[0]))
    arr = np.array(img).astype('float32') / 255.0
    # Keras models sometimes expect shape (1, h, w, c)
    return np.expand_dims(arr, axis=0)

# speech feature extractor
def extract_speech_features_from_bytes(wav_bytes, sr_target=16000, n_mfcc=13):
    # write to temp file and load via librosa for robustness
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
    try:
        y, sr = librosa.load(tmp_path, sr=sr_target)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    # compute mfcc and aggregate
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feats = {}
    for i in range(n_mfcc):
        v = mfcc[i]
        feats[f'mfcc{i+1}_mean'] = float(np.mean(v))
        feats[f'mfcc{i+1}_std'] = float(np.std(v))
        feats[f'mfcc{i+1}_max'] = float(np.max(v))
    return pd.DataFrame([feats])

# gait feature extractor (simple)
def extract_gait_features_from_df(df: pd.DataFrame):
    feats = {}
    for col in df.columns:
        x = pd.to_numeric(df[col], errors='coerce').fillna(0).values
        feats[f'{col}_mean'] = float(np.mean(x))
        feats[f'{col}_std'] = float(np.std(x))
        feats[f'{col}_max'] = float(np.max(x))
        feats[f'{col}_min'] = float(np.min(x))
        feats[f'{col}_energy'] = float(np.sum(x**2)/ (len(x)+1e-9))
    return pd.DataFrame([feats])

# ---------- Streamlit UI ----------
st.title("Ultimate Parkinson's Detector — Final (Multi‑modal)")
st.write("Upload ANY ONE modality file: MRI image (.png/.jpg/.jpeg/.dcm), Spiral/Wave image, Gait CSV/NPY, or Speech audio (.wav/.mp3). The app will pick the right model based on your selection and return a PD probability.")

col1, col2 = st.columns([2,1])
with col1:
    modality = st.selectbox("Select modality to upload", [
        "MRI Image (Keras .h5)",
        "MRI (ANN - parkinsons_ann.h5)",
        "Spiral/Wave Drawing (Keras .h5)",
        "Gait data (CSV / NPY)",
        "Speech audio (WAV / MP3)",
    ])
with col2:
    st.markdown("**Model files expected (place in ./models/):**")
    st.write(f"- MRI image: {MRI_MODEL} (Keras) or {MRI_ANN} (Keras)")
    st.write(f"- Drawing: {DRAWING_MODEL} (Keras)")
    st.write(f"- Gait: {GAIT_MODEL} (Keras)")
    st.write(f"- Speech: {SPEECH_MODEL} (joblib .pkl)")

st.markdown('---')
uploaded = st.file_uploader('Upload file for selected modality', type=None)
run = st.button('Run prediction')

if run:
    if uploaded is None:
        st.error('Please upload a file before predicting.')
    else:
        try:
            if modality.startswith('MRI Image'):
                # read image
                try:
                    img = Image.open(uploaded).convert('RGB')
                except Exception:
                    st.warning('Could not open as image — attempting to read as DICOM or raw bytes.')
                    uploaded.seek(0)
                    data = uploaded.read()
                    # try to open via PIL from bytes
                    try:
                        img = Image.open(io.BytesIO(data)).convert('RGB')
                    except Exception as e:
                        st.error(f'Failed to read MRI image: {e}')
                        st.stop()
                st.image(img, caption='Uploaded MRI image', use_column_width=True)
                x = preprocess_image_for_keras(img, target_shape=(224,224))
                # prefer mri_model first, then mri_ann, else demo
                model_used = None
                preds = None
                if mri_model is not None:
                    try:
                        out = mri_model.predict(x)
                        # handle various output shapes
                        if out.shape[-1] == 1:
                            pd_prob = float(out.squeeze())
                        else:
                            pd_prob = float(out[0,-1])
                        model_used = MRI_MODEL
                    except Exception as e:
                        st.warning(f'Failed to predict with {MRI_MODEL}: {e}')
                if model_used is None and mri_ann is not None:
                    try:
                        out = mri_ann.predict(x)
                        if out.shape[-1] == 1:
                            pd_prob = float(out.squeeze())
                        else:
                            pd_prob = float(out[0,-1])
                        model_used = MRI_ANN
                    except Exception as e:
                        st.warning(f'Failed to predict with {MRI_ANN}: {e}')
                if model_used is None:
                    # demo fallback
                    arr = np.array(img)
                    pd_prob = demo_img.predict_proba(arr)[0,1]
                    model_used = 'demo'
                st.success(f'Predicted PD probability: {pd_prob*100:.1f}% (model: {model_used})')

            elif modality.startswith('MRI (ANN'):
                # similar to above but prefer mri_ann
                try:
                    img = Image.open(uploaded).convert('RGB')
                except Exception:
                    uploaded.seek(0)
                    data = uploaded.read()
                    try:
                        img = Image.open(io.BytesIO(data)).convert('RGB')
                    except Exception as e:
                        st.error(f'Failed to read MRI image: {e}')
                        st.stop()
                st.image(img, caption='Uploaded MRI image (ANN)', use_column_width=True)
                x = preprocess_image_for_keras(img, target_shape=(224,224))
                model_used = None
                pd_prob = None
                if mri_ann is not None:
                    try:
                        out = mri_ann.predict(x)
                        if out.shape[-1] == 1:
                            pd_prob = float(out.squeeze())
                        else:
                            pd_prob = float(out[0,-1])
                        model_used = MRI_ANN
                    except Exception as e:
                        st.warning(f'Failed to predict with {MRI_ANN}: {e}')
                if model_used is None and mri_model is not None:
                    try:
                        out = mri_model.predict(x)
                        if out.shape[-1] == 1:
                            pd_prob = float(out.squeeze())
                        else:
                            pd_prob = float(out[0,-1])
                        model_used = MRI_MODEL
                    except Exception as e:
                        st.warning(f'Failed to predict with {MRI_MODEL}: {e}')
                if model_used is None:
                    arr = np.array(img)
                    pd_prob = demo_img.predict_proba(arr)[0,1]
                    model_used = 'demo'
                st.success(f'Predicted PD probability: {pd_prob*100:.1f}% (model: {model_used})')

            elif modality.startswith('Spiral'):
                try:
                    img = Image.open(uploaded).convert('RGB')
                except Exception as e:
                    st.error(f'Could not read drawing image: {e}')
                    st.stop()
                st.image(img, caption='Uploaded drawing', use_column_width=True)
                x = preprocess_image_for_keras(img, target_shape=(224,224))
                model_used = None
                if drawing_model is not None:
                    try:
                        out = drawing_model.predict(x)
                        if out.shape[-1] == 1:
                            pd_prob = float(out.squeeze())
                        else:
                            pd_prob = float(out[0,-1])
                        model_used = DRAWING_MODEL
                    except Exception as e:
                        st.warning(f'Failed to predict with {DRAWING_MODEL}: {e}')
                if model_used is None:
                    arr = np.array(img)
                    pd_prob = demo_img.predict_proba(arr)[0,1]
                    model_used = 'demo'
                st.success(f'Predicted PD probability from drawing: {pd_prob*100:.1f}% (model: {model_used})')

            elif modality.startswith('Gait'):
                fname = uploaded.name.lower()
                # try csv
                try:
                    if fname.endswith('.csv'):
                        df = pd.read_csv(uploaded)
                    elif fname.endswith('.npy'):
                        arr = np.load(uploaded)
                        if arr.ndim == 1:
                            df = pd.DataFrame({'signal': arr})
                        else:
                            df = pd.DataFrame(arr)
                    else:
                        # try csv
                        uploaded.seek(0)
                        df = pd.read_csv(uploaded)
                except Exception as e:
                    st.error(f'Failed to read gait file: {e}')
                    st.stop()
                st.write('Gait data preview:')
                st.dataframe(df.head())
                feats = extract_gait_features_from_df(df)
                model_used = None
                if gait_model is not None:
                    try:
                        # if keras model expects 2D array
                        try:
                            out = gait_model.predict(feats.values)
                        except Exception:
                            out = gait_model.predict(feats)
                        # interpret output
                        out = np.array(out)
                        if out.ndim==2 and out.shape[-1]>1:
                            pd_prob = float(out[0,-1])
                        else:
                            # scalar
                            pd_prob = float(out.squeeze())
                        model_used = GAIT_MODEL
                    except Exception as e:
                        st.warning(f'Failed to predict with {GAIT_MODEL}: {e}')
                if model_used is None:
                    probs = demo_tab.predict_proba(feats)
                    pd_prob = float(probs[0,1])
                    model_used = 'demo'
                st.success(f'Predicted PD probability from gait: {pd_prob*100:.1f}% (model: {model_used})')

            elif modality.startswith('Speech'):
                bytes_data = uploaded.read()
                st.audio(bytes_data)
                # feature extraction
                with st.spinner('Extracting audio features...'):
                    try:
                        feats = extract_speech_features_from_bytes(bytes_data)
                    except Exception as e:
                        st.error(f'Audio feature extraction failed: {e}')
                        st.stop()
                model_used = None
                if speech_model is not None:
                    try:
                        # some sklearn models expect scaled features; if model includes pipeline, all good
                        probs = speech_model.predict_proba(feats)
                        pd_prob = float(probs[0,1])
                        model_used = SPEECH_MODEL
                    except Exception as e:
                        st.warning(f'Failed to predict with {SPEECH_MODEL}: {e}')
                if model_used is None:
                    probs = demo_tab.predict_proba(feats)
                    pd_prob = float(probs[0,1])
                    model_used = 'demo'
                st.success(f'Predicted PD probability from speech: {pd_prob*100:.1f}% (model: {model_used})')

            else:
                st.error('Unsupported modality')

        except Exception as e:
            st.error(f'Unexpected error during prediction: {e}')
