import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GoSpot | Smart Parking Platform", page_icon="🚗", layout="wide")

# --- LOAD TRAINED ML MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('gospot_model.pkl')
    except Exception:
        return None

model = load_model()

# --- DISTANCE UTILITY (HAVERSINE FORMULA) ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two coordinates in kilometers."""
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)

CITY_COORDINATES = {
    "Manila": (14.5905, 120.9842),
    "Makati": (14.5547, 121.0244),
    "Quezon City": (14.6488, 121.0509)
}

# --- INITIALIZE SESSION STATE DATA ---
if "role" not in st.session_state:
    st.session_state.role = None  # None = Landing Page, 'driver' = Driver UI, 'owner' = Owner UI

if "parking_lots" not in st.session_state:
    st.session_state.parking_lots = [
        {
            "id": 1,
            "name": "Intramuros Surface & Basement Lot",
            "city": "Manila",
            "address": "Muralla St, Intramuros, Manila",
            "lat": 14.5898,
            "lon": 120.9754,
            "price": 50.0,
            "total_capacity": 45,
            "current_free_slots": 12,
            "lighting": True,
            "cctv": True,
            "pwd": True,
            "commends": 18,
            "reviews": [
                {"user": "StudentCommuter", "comment": "Mabilis mag-park pero puno na bandang 7:30 AM."},
                {"user": "Alex_M", "comment": "Good lighting at night, very safe."}
            ]
        },
        {
            "id": 2,
            "name": "Legazpi Village Commercial Parking",
            "city": "Makati",
            "address": "Salcedo St, Legazpi Village, Makati",
            "lat": 14.5532,
            "lon": 121.0185,
            "price": 80.0,
            "total_capacity": 60,
            "current_free_slots": 22,
            "lighting": True,
            "cctv": True,
            "pwd": True,
            "commends": 34,
            "reviews": [
                {"user": "AnthonyUy", "comment": "Spacious slots and reliable security guards."}
            ]
        },
        {
            "id": 3,
            "name": "Timog Avenue Secure Lot",
            "city": "Quezon City",
            "address": "Timog Ave cor. Tomas Morato, Quezon City",
            "lat": 14.6360,
            "lon": 121.0345,
            "price": 60.0,
            "total_capacity": 30,
            "current_free_slots": 5,
            "lighting": True,
            "cctv": True,
            "pwd": False,
            "commends": 9,
            "reviews": [
                {"user": "QC_Driver", "comment": "Tight slots, but convenient for restaurants nearby."}
            ]
        }
    ]

# --- APP HEADER (VISIBLE ON ALL PAGES) ---
col_left, col_center, col_right = st.columns([2, 1, 2])

with col_center:
    try:
        st.image("1.png", use_container_width=True) 
    except FileNotFoundError:
        st.markdown("<h1 style='text-align: center;'>🚗 GoSpot</h1>", unsafe_allow_html=True)

# Moved OUTSIDE the columns to perfectly center across the entire screen width
st.markdown("<p style='text-align: center; color: gray;'>AI-Powered Parking Availability & Prediction Platform</p>", unsafe_allow_html=True)

# ==========================================
# PAGE 0: LANDING PAGE (ROLE SELECTION)
# ==========================================
if st.session_state.role is None:
    st.markdown("<h2 style='text-align: center;'>Welcome to GoSpot!</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please select your role to continue.</p>", unsafe_allow_html=True)
    
    st.write("") # Spacing
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    
    with col2:
        if st.button("🚘 I am a Driver", use_container_width=True):
            st.session_state.role = 'driver'
            st.rerun()
            
    with col3:
        if st.button("🏢 I am a Parking Owner", use_container_width=True):
            st.session_state.role = 'owner'
            st.rerun()

# ==========================================
# PAGE 1: DRIVER INTERFACE
# ==========================================
elif st.session_state.role == 'driver':
# Back Button
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state.role = None
            st.rerun()

    st.subheader("Find & Compare Nearby Parking Spots")

    # Search and Parameter Inputs
    col1, col2 = st.columns([2, 1])
    with col1:
        destination_address = st.text_input(
            "📍 Enter your destination address:",
            placeholder="e.g. Intramuros Manila, Greenbelt Makati, or Tomas Morato QC"
        )
    with col2:
        target_city = st.selectbox("Select Target City:", ["Manila", "Makati", "Quezon City"])

    col3, col4 = st.columns(2)
    with col3:
        target_time = st.time_input("Expected Arrival Time", datetime.time(8, 0))
    with col4:
        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )

    search_button = st.button("Search Nearest Parking", type="primary", use_container_width=True)

    if search_button or destination_address:
        # Determine driver anchor coordinates
        base_lat, base_lon = CITY_COORDINATES[target_city]
        driver_lat, driver_lon = base_lat, base_lon

        # Calculate distances to all lots
        lots_display = []
        city_code_map = {"Manila": 0, "Makati": 1, "Quezon City": 2}
        day_code_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        time_minutes = target_time.hour * 60 + target_time.minute

        for lot in st.session_state.parking_lots:
            # Calculate distance using coordinates
            dist_km = haversine(driver_lat, driver_lon, lot["lat"], lot["lon"])
            
            # Predict availability using XGBoost model
            if model:
                features = pd.DataFrame([{
                    'day_of_week': day_code_map[day_of_week],
                    'time_of_day_minute': time_minutes,
                    'total_capacity': lot["total_capacity"],
                    'city_code': city_code_map.get(lot["city"], 0)
                }])
                pred_occ = float(model.predict(features)[0])
                pred_occ = max(0.0, min(1.0, pred_occ))
                availability_pct = round((1 - pred_occ) * 100, 1)
            else:
                availability_pct = round((lot["current_free_slots"] / lot["total_capacity"]) * 100, 1)

            lots_display.append({**lot, "distance_km": dist_km, "predicted_avail": availability_pct})

        # Sort lots: nearest to farthest
        lots_display = sorted(lots_display, key=lambda x: x["distance_km"])

        st.markdown(f"### 🎯 Results Near: **{destination_address if destination_address else target_city}**")
        st.write(f"Showing **{len(lots_display)}** parking options sorted by proximity:")

        for lot in lots_display:
            with st.container():
                st.markdown(f"#### 🏢 {lot['name']}")
                st.caption(f"📍 {lot['address']} ({lot['city']})")

                # Metrics row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Distance", f"{lot['distance_km']} km away")
                c2.metric("Predicted Availability", f"{lot['predicted_avail']}%")
                c3.metric("Live Free Slots", f"{lot['current_free_slots']} / {lot['total_capacity']}")
                c4.metric("Rate", f"₱{lot['price']:.2f}")

                # Amenities icons
                amenities = []
                if lot["lighting"]: amenities.append("💡 Well-lit")
                if lot["cctv"]: amenities.append("📹 Active CCTV")
                if lot["pwd"]: amenities.append("♿ PWD Accessible")
                st.write("**Amenities:** " + " • ".join(amenities) if amenities else "Standard Lot")

                # Commend & Reviews Section
                col_btn, col_count = st.columns([1, 4])
                with col_btn:
                    if st.button(f"👍 Commend ({lot['commends']})", key=f"commend_{lot['id']}"):
                        for item in st.session_state.parking_lots:
                            if item["id"] == lot["id"]:
                                item["commends"] += 1
                                st.rerun()

                # Expandable Community Reviews
                with st.expander(f"💬 Driver Reviews & Notes ({len(lot['reviews'])})"):
                    if lot["reviews"]:
                        for r in lot["reviews"]:
                            st.markdown(f"- **@{r['user']}**: {r['comment']}")
                    else:
                        st.info("No reviews yet. Be the first to leave one!")

                    with st.form(key=f"review_form_{lot['id']}"):
                        user_handle = st.text_input("Your Name / Handle", key=f"user_{lot['id']}")
                        user_comment = st.text_area("Write a review or parking tip:", key=f"comment_{lot['id']}")
                        submit_review = st.form_submit_button("Post Review")

                        if submit_review:
                            if user_handle and user_comment:
                                for item in st.session_state.parking_lots:
                                    if item["id"] == lot["id"]:
                                        item["reviews"].append({"user": user_handle, "comment": user_comment})
                                        st.success("Review posted!")
                                        st.rerun()
                            else:
                                st.error("Please provide both your name and a comment.")

                st.markdown("---")

# ==========================================
# PAGE 2: PARKING OWNER INTERFACE
# ==========================================
elif st.session_state.role == 'owner':
    # Back Button
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state.role = None
            st.rerun()

    st.subheader("Manage & List Parking Spaces")
    st.write("Post your available parking slots to reach drivers before they leave.")

    with st.expander("➕ List a New Parking Spot / Lot", expanded=True):
        with st.form("add_lot_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                lot_name = st.text_input("Parking Lot / Facility Name", placeholder="e.g. Sunshine 100 Basement Parking")
                city = st.selectbox("City", ["Manila", "Makati", "Quezon City"])
                specific_address = st.text_input("Specific Address / Street", placeholder="e.g. 123 Pioneer St, Mandaluyong")
            with col_b:
                price = st.number_input("Parking Rate (₱ Flat or Base)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
                total_capacity = st.number_input("Total Parking Capacity", min_value=1, max_value=500, value=20, step=1)
                free_slots = st.number_input("Currently Available Slots", min_value=0, max_value=500, value=10, step=1)

            st.write("**Amenities & Security Features:**")
            col_c, col_d, col_e = st.columns(3)
            with col_c:
                has_light = st.checkbox("Well-lit at night", value=True)
            with col_d:
                has_cctv = st.checkbox("Active CCTV Coverage", value=True)
            with col_e:
                has_pwd = st.checkbox("PWD Accessible Slots", value=False)

            submitted = st.form_submit_button("Publish Parking Spot", type="primary", use_container_width=True)

            if submitted:
                if not lot_name or not specific_address:
                    st.error("Please fill in both the parking facility name and specific address.")
                else:
                    base_lat, base_lon = CITY_COORDINATES[city]
                    simulated_lat = base_lat + np.random.uniform(-0.01, 0.01)
                    simulated_lon = base_lon + np.random.uniform(-0.01, 0.01)

                    new_lot = {
                        "id": len(st.session_state.parking_lots) + 1,
                        "name": lot_name,
                        "city": city,
                        "address": specific_address,
                        "lat": simulated_lat,
                        "lon": simulated_lon,
                        "price": float(price),
                        "total_capacity": int(total_capacity),
                        "current_free_slots": int(free_slots),
                        "lighting": has_light,
                        "cctv": has_cctv,
                        "pwd": has_pwd,
                        "commends": 0,
                        "reviews": []
                    }
                    st.session_state.parking_lots.append(new_lot)
                    st.success(f"'{lot_name}' has been successfully listed!")

    st.subheader("Your Active Parking Inventory")
    inventory_df = pd.DataFrame([
        {
            "Name": lot["name"],
            "City": lot["city"],
            "Address": lot["address"],
            "Rate": f"₱{lot['price']:.2f}",
            "Free / Total": f"{lot['current_free_slots']} / {lot['total_capacity']}",
            "Commends": lot["commends"],
            "Reviews": len(lot["reviews"])
        }
        for lot in st.session_state.parking_lots
    ])
    st.dataframe(inventory_df, use_container_width=True)
