import streamlit as st
import sys
import os
import joblib
import numpy as np
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.questions import questions

# Initialize session state for predictions
if 'dt_prediction' not in st.session_state:
    st.session_state.dt_prediction = None
if 'rf_prediction' not in st.session_state:
    st.session_state.rf_prediction = None

# Load models
model_dir = os.path.join(os.path.dirname(__file__), '../models')
dt_model = joblib.load(os.path.join(model_dir, 'dtmodel.pkl'))
rf_model = joblib.load(os.path.join(model_dir, 'rfmodel.pkl'))

def parse_int_safely(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

# Encoding dictionaries
encoding_question7 = {
    'R Programming': 0,
    'Information Security': 1,
    'Shell Programming': 2,
    'Machine Learning': 3,
    'Full Stack': 4,
    'Hadoop': 5,
    'Python': 6,
    'Distro Making': 7,
    'App Development': 8
}

encoding_question8 = {
    'Database Security': 0,
    'System Designing': 1,
    'Web Technologies': 2,
    'Machine Learning': 3,
    'Hacking': 4,
    'Testing': 5,
    'Data Science': 6,
    'Game Development': 7,
    'Cloud Computing': 8
}

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .top-prediction {
        background-color: #76D77EFF;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        position: relative;
        overflow: hidden;
    }
    .other-prediction {
        background-color: #f5f5f5;
        padding: 0.8rem;
        border-radius: 4px;
        margin: 0.3rem 0;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2 {
        color: #2c3e50;
        margin-top: 1.5rem;
    }
    h3 {
        color: #34495e;
    }
    .mindmap-button {
        background-color: #9c27b0 !important;
    }
    .mindmap-button:hover {
        background-color: #7b1fa2 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Career Path Recommender")

# Add a description
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #666;'>
            Answer the following questions to discover your ideal career path. 
            Our AI models will analyze your responses and provide personalized recommendations.
        </p>
    </div>
""", unsafe_allow_html=True)

user_answers = {}
validation_errors = []

# Create a container for questions
with st.container():
    for i, q in enumerate(questions):
        st.markdown(f"### Question {i+1}", unsafe_allow_html=True)
        st.markdown("<style>h3 {color: white !important;}</style>", unsafe_allow_html=True)
        if q["type"] == "number":
            user_answers[q["id"]] = st.slider(
                q["text"], 
                min_value=q["min"], 
                max_value=q["max"], 
                step=1,
                help=f"Rate from {q['min']} to {q['max']}"
            )
        elif q["type"] == "select":
            options = {opt["label"]: opt["value"] for opt in q["options"]}
            options = {"Select an option": None, **options}
            user_answers[q["id"]] = st.selectbox(
                q["text"], 
                list(options.keys()), 
                key=q["id"],
                help="Choose the option that best describes you"
            )
            user_answers[q["id"]] = options[user_answers[q["id"]]]
            if user_answers[q["id"]] is None:
                validation_errors.append(f"Please select an option for: {q['text']}")

# Prediction button with custom styling
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
predict_button = st.button("🔮 Get Career Predictions", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if predict_button:
    if validation_errors:
        for error in validation_errors:
            st.error(error)
            break
    else:
        # Prepare data for prediction
        raw_data = [user_answers[q["id"]] for q in questions]
        
        # Encode the data
        encoded_data = [
            min(raw_data[0], 9),
            raw_data[1],
            min(raw_data[2], 9),
            min(raw_data[3], 9),
            parse_int_safely(raw_data[4]),
            parse_int_safely(raw_data[5]),
            encoding_question7[raw_data[6]],
            encoding_question8[raw_data[7]],
            parse_int_safely(raw_data[8]),
            parse_int_safely(raw_data[9]),
            parse_int_safely(raw_data[10]),
            parse_int_safely(raw_data[11]),
            parse_int_safely(raw_data[12]),
            parse_int_safely(raw_data[13]),
            parse_int_safely(raw_data[14]),
            parse_int_safely(raw_data[15]),
            parse_int_safely(raw_data[16]),
            parse_int_safely(raw_data[17]),
            parse_int_safely(raw_data[18]),
        ]
        print("encoded_data : ", encoded_data)

        # Make predictions using both models
        dt_prediction = dt_model.predict([encoded_data])[0]
        dt_proba = dt_model.predict_proba([encoded_data])[0]
        dt_probability = float(dt_proba[np.where(dt_model.classes_ == dt_prediction)[0][0]])

        rf_prediction = rf_model.predict([encoded_data])[0]
        rf_proba = rf_model.predict_proba([encoded_data])[0]
        rf_probability = float(rf_proba[np.where(rf_model.classes_ == rf_prediction)[0][0]])

        # Store predictions in session state
        st.session_state.dt_prediction = dt_prediction
        st.session_state.rf_prediction = rf_prediction

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌳 Decision Tree Model")
            st.markdown(f"""
                <div class="top-prediction">
                    <div style="
                        width: {dt_probability*100}%;
                        height: 100%;
                        position: absolute;
                        top: 0;
                        left: 0;
                        background-color: rgba(118, 215, 126, 0.5);
                        z-index: 1;
                    "></div>
                    <div style="position: relative; z-index: 2;">
                        <strong>Primary Recommendation:</strong> {dt_prediction}<br>
                        <strong>Confidence:</strong> {dt_probability:.2%}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Other Potential Careers")
            top_3_indices = np.argsort(dt_proba)[-3:][::-1]
            for idx in top_3_indices:
                career = dt_model.classes_[idx]
                prob = dt_proba[idx]
                st.markdown(f'<div class="other-prediction">', unsafe_allow_html=True)
                st.markdown(f"**{career}:** {prob:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### 🌲 Random Forest Model")
            st.markdown(f"""
                <div class="top-prediction">
                    <div style="
                        width: {rf_probability*100}%;
                        height: 100%;
                        position: absolute;
                        top: 0;
                        left: 0;
                        background-color: rgba(118, 215, 126, 0.5);
                        z-index: 1;
                    "></div>
                    <div style="position: relative; z-index: 2;">
                        <strong>Primary Recommendation:</strong> {rf_prediction}<br>
                        <strong>Confidence:</strong> {rf_probability:.2%}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Other Potential Careers")
            top_3_indices = np.argsort(rf_proba)[-3:][::-1]
            for idx in top_3_indices:
                career = rf_model.classes_[idx]
                prob = rf_proba[idx]
                st.markdown(f'<div class="other-prediction">', unsafe_allow_html=True)
                st.markdown(f"**{career}:** {prob:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)

        # # Add mind map generation button
        # st.markdown("<div style='text-align: center; margin-top: 2rem;'>", unsafe_allow_html=True)
        # if st.button("🧠 Generate Career Mind Map", use_container_width=True, key="mindmap_button", 
        #             help="Create a detailed mind map for your career path"):
        #     st.switch_page("mindmap.py")
        # st.markdown("</div>", unsafe_allow_html=True)
