import streamlit as st
import pandas as pd
import joblib
import tempfile
from feature_extraction import extract_aperture_from_video, extract_blink_features

st.title("Blink-Based Emotion Detection App(high or low arousal emotions")
st.write("High Arousal and 0 for Low Arousal.

High Arousal (1):

1: admiration
2: adoration
3: aesthetic
4: amusement
5: anger
6: anxiety
7: awe
12: craving
13: disgust
14: empathic pain
15: entrancement
16: excitement
17: fear
18: horror
19: interest
20: joy
23: romance
26: sexual desire
27: surprised
Low Arousal (0):

8: awkwardness
9: boredom
10: calmness
11: confusion
21: nostalgia
22: relief
24: sadness
25: satisfaction)
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
