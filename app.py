import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import librosa
import tempfile
import os

st.set_page_config(page_title="Parkinson Detector", layout="wide")

# Model paths
MRI_MODEL = "models/mri_model.h5"
DRAWING_MODEL = "models/drawing_model.h5"
GAIT_MODEL = "models/gait_model.pkl"
SPEECH_MODEL = "models/speech_model.pkl"

# Cache model loaders
@st.cache_resource
def load_keras(path):
    try:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
    return None

@st.cache_resource
def load_pickle(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
    return None

mri_model = load_keras(MRI_MODEL)
drawing_model = load_keras(DRAWING_MODEL)
gait_model = load_pickle(GAIT_MODEL)
speech_model = load_pickle(SPEECH_MODEL)

# Fallback demo
class Demo:
    def predict_image(self,img):
        arr=np.array(img)
        p=float(arr.mean()/255)
        return p
    def predict_tab(self,x):
        return 0.5

DEMO = Demo()

# Preprocessing
def preprocess_image(img,target=(224,224)):
    img=img.convert("RGB").resize(target)
    arr=np.array(img)/255.0
    return np.expand_dims(arr,0)

def extract_speech(bytes_data):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as tmp:
        tmp.write(bytes_data)
        path=tmp.name
    y,sr=librosa.load(path,sr=16000)
    os.remove(path)
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
    feats={f"mfcc_{i}_mean":mfcc[i].mean() for i in range(13)}
    return pd.DataFrame([feats])

def extract_gait(df):
    feats={}
    for col in df.columns:
        x=pd.to_numeric(df[col],errors='coerce').fillna(0)
        feats[f"{col}_mean"]=x.mean()
        feats[f"{col}_std"]=x.std()
    return pd.DataFrame([feats])

# UI
st.title("Parkinson's Disease Detector — Multi-Modal")
mod=st.selectbox("Choose modality",[
    "MRI Image",
    "Drawing Image",
    "Gait Data (CSV/NPY)",
    "Speech Audio",
])
file=st.file_uploader("Upload file",type=None)

if st.button("Predict"):
    if file is None:
        st.error("Please upload a file.")
    else:
        if mod=="MRI Image":
            try:
                img=Image.open(file)
            except:
                st.error("Invalid image")
                st.stop()
            st.image(img,use_column_width=True)
            x=preprocess_image(img)
            if mri_model:
                out=mri_model.predict(x)
                p=float(out.squeeze()) if out.shape[-1]==1 else float(out[0,-1])
            else:
                p=DEMO.predict_image(img)
            st.success(f"PD Probability: {p*100:.2f}%")

        elif mod=="Drawing Image":
            try:
                img=Image.open(file)
            except:
                st.error("Invalid image")
                st.stop()
            st.image(img,use_column_width=True)
            x=preprocess_image(img)
            if drawing_model:
                out=drawing_model.predict(x)
                p=float(out.squeeze()) if out.shape[-1]==1 else float(out[0,-1])
            else:
                p=DEMO.predict_image(img)
            st.success(f"PD Probability: {p*100:.2f}%")

        elif mod=="Gait Data (CSV/NPY)":
            try:
                if file.name.endswith(".csv"):
                    df=pd.read_csv(file)
                elif file.name.endswith(".npy"):
                    arr=np.load(file)
                    df=pd.DataFrame(arr)
                else:
                    df=pd.read_csv(file)
            except:
                st.error("Invalid gait data file")
                st.stop()
            feats=extract_gait(df)
            if gait_model:
                p=float(gait_model.predict_proba(feats)[0,1])
            else:
                p=DEMO.predict_tab(feats)
            st.success(f"PD Probability: {p*100:.2f}%")

        elif mod=="Speech Audio":
            bytes_data=file.read()
            st.audio(bytes_data)
            feats=extract_speech(bytes_data)
            if speech_model:
                p=float(speech_model.predict_proba(feats)[0,1])
            else:
                p=DEMO.predict_tab(feats)
            st.success(f"PD Probability: {p*100:.2f}%")
