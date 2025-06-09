import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ---- Mood-based background images ----
mood_backgrounds = {
    "Good": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1500&q=80",
    "Bad": "https://images.unsplash.com/photo-1465101046530-73398c7f28ca?auto=format&fit=crop&w=1500&q=80",
    "Neutral": "https://images.unsplash.com/photo-1465101178521-c1a9136a3fdc?auto=format&fit=crop&w=1500&q=80",
    "Great": "https://images.unsplash.com/photo-1502082553048-f009c37129b9?auto=format&fit=crop&w=1500&q=80",
    "Sad": "https://images.unsplash.com/photo-1465101046530-73398c7f28ca?auto=format&fit=crop&w=1500&q=80",
    "Angry": "https://images.unsplash.com/photo-1504196606672-aef5c9cefc92?auto=format&fit=crop&w=1500&q=80",
    "Cool": "https://images.unsplash.com/photo-1465101178521-c1a9136a3fdc?auto=format&fit=crop&w=1500&q=80",
    "Chill": "https://images.unsplash.com/photo-1465101046530-73398c7f28ca?auto=format&fit=crop&w=1500&q=80",
    # Add more as needed
}
default_bg = "https://images.unsplash.com/photo-1493558103817-58b2924bce98?auto=format&fit=crop&w=1500&q=80"

def set_bg(bg_url):
    import time
    timestamp = int(time.time())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_url}?{timestamp}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .exclusive-title {{
            font-size:3em;
            font-weight:bold;
            color:#fff;
            text-shadow: 2px 2px 8px #000;
            text-align:center;
            margin-bottom: 30px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---- Load model and encoders ----
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(MODEL_DIR, 'best_rf_model.joblib'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
le_sub_mood = joblib.load(os.path.join(MODEL_DIR, 'le_sub_mood.joblib'))
le_weekday = joblib.load(os.path.join(MODEL_DIR, 'le_weekday.joblib'))
mlb = joblib.load(os.path.join(MODEL_DIR, 'mlb_activities.joblib'))
columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.joblib'))

# ---- Emoji mapping for moods ----
mood_emojis = {
    "Good": "😃",
    "Bad": "😞",
    "Neutral": "😐",
    "Great": "🤩",
    "Sad": "😢",
    "Angry": "😡",
    "Cool": "😎",
    "Chill": "🧘",
    # Add more mappings as needed
}

# ---- Session state for background ----
if "bg_url" not in st.session_state:
    st.session_state.bg_url = default_bg

# Always set background at the top using the current value
set_bg(st.session_state.bg_url)

# ...existing code...

st.markdown("""
    <style>
    .highlight-label {
        font-size: 1.5em;
        font-weight: bold;
        color: #000;  /* Changed from pink to black */
        text-shadow: 1px 1px 4px #fff;
        margin-bottom: 0.2em;
        margin-top: 1em;
    }
    .exclusive-title {
        font-size: 3em;
        font-weight: bold;
        color: #000;
        text-align: center;
        margin-bottom: 30px;
        /* Glowing effect */
        text-shadow:
            0 0 10px #fff,
            0 0 20px #fff,
            0 0 30px #ff00de,
            0 0 40px #ff00de,
            0 0 50px #ff00de;
        animation: swing 0s infinite;
        display: inline-block;
        transform-origin: top center;
    }
    @keyframes swing {
        0%, 100% { transform: rotate(0deg);}
        50% { transform: rotate(3deg);}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="exclusive-title"> Mood Prediction App </div>', unsafe_allow_html=True)

st.markdown('<div class="highlight-label">Sub Mood</div>', unsafe_allow_html=True)
sub_mood = st.selectbox("", le_sub_mood.classes_)

st.markdown('<div class="highlight-label">Weekday</div>', unsafe_allow_html=True)
weekday = st.selectbox("", le_weekday.classes_)

st.markdown('<div class="highlight-label">Activities</div>', unsafe_allow_html=True)
activities = st.multiselect("", mlb.classes_)

# ...rest of your code...

if st.button("Predict Mood"):
    # Prepare input
    sample = pd.DataFrame(columns=columns)
    sample.loc[0] = 0
    sample['sub_mood'] = le_sub_mood.transform([sub_mood])[0]
    sample['weekday'] = le_weekday.transform([weekday])[0]
    for activity in mlb.classes_:
        sample[activity] = 1 if activity in activities else 0
    sample_scaled = scaler.transform(sample.values)
    prediction = model.predict(sample_scaled)[0]
    emoji = mood_emojis.get(str(prediction), "🙂")
    bg_url = mood_backgrounds.get(str(prediction), default_bg)
    st.session_state.bg_url = bg_url  # Update background for next rerun
    st.markdown(
        f"<h2 style='text-align:center; color:#fff; text-shadow:2px 2px 8px #000;'>Predicted mood: {prediction} {emoji}</h2>",
        unsafe_allow_html=True
    )