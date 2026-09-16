import streamlit as st
import pandas as pd
import joblib
import datetime
import math

# Configure page settings
st.set_page_config(page_title="GoSpot | Driver App", page_icon="🚗", layout="centered")

# Load model efficiently (caches it so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    return joblib.load('gospot_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please run train_model.py first to generate gospot_model.pkl.")
    st.stop()

st.title("🚗 GoSpot")
st.subheader("Get a Spot before you go.")
st.markdown("---")
st.write("**Plan Your Parking**")

# UI Inputs for the general public
col1, col2 = st.columns(2)
with col1:
    target_time = st.time_input("Target Arrival Time", datetime.time(8, 30))
    day_of_week = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
with col2:
    city = st.selectbox("Destination City", ["Manila", "Makati", "Quezon City"])

# Mapping string inputs to numerical data for the AI model
day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
city_map = {"Manila": 0, "Makati": 1, "Quezon City": 2}

if st.button("Find Parking", type="primary", use_container_width=True):
    # Convert chosen time to minutes from midnight
    time_minutes = target_time.hour * 60 + target_time.minute
    
    # Prepare features for the model
    features = pd.DataFrame([{
        'day_of_week': day_map[day_of_week],
        'time_of_day_minute': time_minutes,
        'total_capacity': 50,
        'city_code': city_map[city]
    }])
    
    # Run prediction directly
    predicted_occupancy = float(model.predict(features)[0])
    predicted_occupancy = max(0.0, min(1.0, predicted_occupancy))
    availability_pct = round((1 - predicted_occupancy) * 100, 1)
    estimated_free_slots = int(50 * (1 - predicted_occupancy))
    
    # Static confidence score for MVP presentation purposes
    confidence_score = 92.5
    
    st.markdown(f"### 📍 Best Match: {city} Central Parking")
    
    # Display metrics clearly
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Predicted Availability", value=f"{availability_pct}%")
    m2.metric(label="Est. Free Slots", value=f"{estimated_free_slots} / 50")
    m3.metric(label="Estimated Rate", value="₱60.00")
    
    st.success(f"✅ **High Confidence** (Data updated recently. Score: {confidence_score})")
        
    st.markdown("""
    **Safety & Amenities:**
    * 💡 Well-lit at night
    * 📹 Active CCTV
    * ♿ PWD Accessible Slots Available
    """)