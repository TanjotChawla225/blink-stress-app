import streamlit as st
import pandas as pd
import joblib
import tempfile
from feature_extraction import extract_aperture_from_video, extract_blink_features

st.title("Blink-Based Emotion Detection App(high or low arousal emotions")
st.write(" High Arousal like admiration , amusement, excitement , fear and Low Arousal like boredom , confusion, calmness, relief")
st.write("Upload a 4–5 second video showing your face and eyes.")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())
    temp_path = temp.name

    st.video(temp_path)

    st.write("Extracting blink waveform…")
    ap, fps = extract_aperture_from_video(temp_path)

    st.write("Extracting blink features…")
    feats = extract_blink_features(ap, fps)

    st.write(pd.DataFrame([feats]))

    scaler = joblib.load("scaler.pkl")
    model = joblib.load("stress_model.pkl")

    X = pd.DataFrame([feats])
    X_scaled = scaler.transform(X)

    pred = model.predict(X)[0]

    st.subheader("Prediction")
    if pred == 1:
        st.error(" HIGH AROUSAL !!")
    else:
        st.success("LOW AROUSAL :D")
