import streamlit as st
import google.generativeai as genai
import os
from pathlib import Path
import json
from dotenv import load_dotenv
import logging
import datetime

print("\n=== Starting Mind Map Generation ===")

# Configure logging
def setup_logger():
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    print(f"Log directory created/verified at: {log_dir}")
    
    # Create a logger
    logger = logging.getLogger('career_recommender')
    logger.setLevel(logging.INFO)
    
    # Create a file handler for logging to a file
    log_file = log_dir / f'mindmap_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    print(f"Logger initialized with log file: {log_file}")
    return logger

# Initialize logger
logger = setup_logger()

# Hide Streamlit sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# Load environment variables from .env file
print("\nLoading environment variables...")
load_dotenv()
logger.info("Environment variables loaded")

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in environment variables")
    logger.error("GOOGLE_API_KEY not found in environment variables")
    st.error("Please set the GOOGLE_API_KEY in your .env file")
    st.stop()

print("Configuring Gemini API...")
genai.configure(api_key=GOOGLE_API_KEY)
logger.info("Gemini API configured")

# Initialize Gemini model
print("Initializing Gemini model...")
model = genai.GenerativeModel('models/gemini-2.5-flash-preview-04-17')
logger.info("Gemini model initialized")

st.title("🧠 Career Path Mind Map")

# Get predictions from session state
print("\nRetrieving predictions from session state...")
dt_prediction = st.session_state.get('dt_prediction')
rf_prediction = st.session_state.get('rf_prediction')

print(f"Predictions retrieved - DT: {dt_prediction}, RF: {rf_prediction}")
logger.info(f"Retrieved predictions - DT: {dt_prediction}, RF: {rf_prediction}")

if not dt_prediction or not rf_prediction:
    print("WARNING: No predictions found in session state")
    logger.warning("No predictions found in session state")
    st.warning("Please go back and get career predictions first!")
    if st.button("Go Back"):
        st.switch_page("app.py")
    st.stop()

# Create a prompt for Gemini
print("\nGenerating prompt for Gemini...")
prompt = f"""
Create a detailed mind map for a career path in {dt_prediction} (Decision Tree prediction) and {rf_prediction} (Random Forest prediction).
Include the following aspects:
1. Required Skills
2. Educational Path
3. Career Progression
4. Industry Applications
5. Tools and Technologies
6. Certifications
7. Salary Range
8. Job Market Trends

Format the response as a JSON object with the following structure:
{{
    "career_path": {{
        "title": "Career Title",
        "description": "Brief description",
        "skills": ["skill1", "skill2", ...],
        "education": ["degree1", "degree2", ...],
        "progression": ["entry", "mid", "senior", "expert"],
        "applications": ["industry1", "industry2", ...],
        "tools": ["tool1", "tool2", ...],
        "certifications": ["cert1", "cert2", ...],
        "salary_range": {{
            "entry": "range",
            "mid": "range",
            "senior": "range"
        }},
        "market_trends": ["trend1", "trend2", ...]
    }}
}}
"""

print("Prompt generated successfully")
logger.info("Generated prompt for Gemini")

# Generate mind map
with st.spinner("Generating mind map..."):
    try:
        print("\nSending request to Gemini API...")
        logger.info("Sending request to Gemini API")
        response = model.generate_content(prompt)
        print("Response received from Gemini API")
        logger.info("Received response from Gemini API")
        
        print("\nParsing JSON response...")
        logger.info("Parsing JSON response")
        mind_map_data = json.loads(response.text.strip()
                .replace('```json', '')
                .replace('```', '')
                .strip())
        print("JSON parsed successfully")
        logger.info("Successfully parsed JSON response")
        
        # Display the mind map
        st.markdown("## Career Path Analysis")
        
        # Main career info
        career_title = mind_map_data['career_path']['title']
        print(f"\nDisplaying career path for: {career_title}")
        st.markdown(f"### {career_title}")
        st.markdown(mind_map_data['career_path']['description'])
        logger.info(f"Displaying career path for: {career_title}")
        
        # Create columns for different sections
        col1, col2 = st.columns(2)
        
        with col1:
            print("\nRendering left column sections...")
            st.markdown("#### 🎯 Required Skills")
            for skill in mind_map_data['career_path']['skills']:
                st.markdown(f"- {skill}")
            
            st.markdown("#### 📚 Educational Path")
            for degree in mind_map_data['career_path']['education']:
                st.markdown(f"- {degree}")
            
            st.markdown("#### 📈 Career Progression")
            for level in mind_map_data['career_path']['progression']:
                st.markdown(f"- {level}")
            
            st.markdown("#### 🏢 Industry Applications")
            for industry in mind_map_data['career_path']['applications']:
                st.markdown(f"- {industry}")
        
        with col2:
            print("Rendering right column sections...")
            st.markdown("#### 🛠️ Tools and Technologies")
            for tool in mind_map_data['career_path']['tools']:
                st.markdown(f"- {tool}")
            
            st.markdown("#### 📜 Certifications")
            for cert in mind_map_data['career_path']['certifications']:
                st.markdown(f"- {cert}")
            
            st.markdown("#### 💰 Salary Range")
            salary = mind_map_data['career_path']['salary_range']
            st.markdown(f"- Entry Level: {salary['entry']}")
            st.markdown(f"- Mid Level: {salary['mid']}")
            st.markdown(f"- Senior Level: {salary['senior']}")
            
            st.markdown("#### 📊 Market Trends")
            for trend in mind_map_data['career_path']['market_trends']:
                st.markdown(f"- {trend}")
        
        print("All mind map sections displayed successfully")
        logger.info("Successfully displayed all mind map sections")
        
        # Add a button to go back
        st.markdown("<div style='text-align: center; margin-top: 2rem;'>", unsafe_allow_html=True)
        if st.button("← Go Back to Predictions"):
            print("\nUser clicked 'Go Back' button")
            logger.info("User clicked 'Go Back' button")
            st.switch_page("app.py")
        st.markdown("</div>", unsafe_allow_html=True)
        
    except json.JSONDecodeError as e:
        print(f"\nERROR: JSON parsing error: {str(e)}")
        print(f"Raw response: {response.text}")
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Raw response: {response.text}")
        st.error("Error parsing the mind map data. Please try again.")
        if st.button("Go Back"):
            st.switch_page("app.py")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        st.error(f"Error generating mind map: {str(e)}")
        if st.button("Go Back"):
            st.switch_page("app.py")

print("\n=== Mind Map Generation Complete ===") 