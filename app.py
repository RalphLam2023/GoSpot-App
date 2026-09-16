import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GoSpot | Smart Parking Platform", page_icon="🚗", layout="wide")

# --- CUSTOM CSS: BIGGER BUTTONS & GREEN THEME ---
st.markdown("""
    <style>
    /* Global Primary Green Theme Colors */
    :root {
        --primary-color: #16a34a;
    }

    /* Target all buttons across the app to make them bigger */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        padding: 12px 24px !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid #16a34a !important;
    }

    /* Primary buttons (Driver / Owner buttons & Main Action buttons) */
    div.stButton > button[kind="primary"] {
        background-color: #16a34a !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.3) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #15803d !important;
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.4) !important;
    }

    /* Extra-large styling specifically for the Landing Page selection buttons */
    .landing-btn div.stButton > button {
        height: 100px !important;
        font-size: 28px !important;
        letter-spacing: 0.5px;
    }

    /* Secondary / standard buttons hover */
    div.stButton > button[kind="secondary"]:hover {
        border-color: #16a34a !important;
        color: #16a34a !important;
        background-color: #f0fdf4 !important;
    }

    /* Green accent highlights for titles and headers */
    h1, h2, h3, h4 {
        color: #14532d;
    }

    /* Metric value color */
    div[data-testid="stMetricValue"] {
        color: #16a34a !important;
    }
    </style>
""", unsafe_allow_html=True)

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
    st.session_state.role = None  

if "parking_lots" not in st.session_state:
    st.session_state.parking_lots = [
        {
            "id": 1,
            "owner_name": "Bayleaf Hotel",
            "name": "Bayleaf Hotel Parking",
            "city": "Manila",
            "address": "Muralla St corner Victoria St, Intramuros, Manila",
            "lat": 14.5898,
            "lon": 120.9754,
            "price": 150.0,
            "total_capacity": 80,
            "current_free_slots": 25,
            "lighting": True,
            "cctv": True,
            "pwd": True,
            "commends": 42,
            "reviews": [{"user": "StudentCommuter", "comment": "Very secure and safe hotel basement, though pricier than street parking."}]
        },
        {
            "id": 2,
            "owner_name": "Ayala Property Mgmt",
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
            "reviews": [{"user": "AnthonyUy", "comment": "Spacious slots and reliable security guards."}]
        },
        {
            "id": 3,
            "owner_name": "QC LGU Admin",
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
            "reviews": [{"user": "QC_Driver", "comment": "Tight slots, but convenient for restaurants nearby."}]
        }
    ]

# --- APP HEADER (VISIBLE ON ALL PAGES) ---
col_left, col_center, col_right = st.columns([2, 1, 2])
with col_center:
    try:
        st.image("1.png", use_container_width=True) 
    except FileNotFoundError:
        st.markdown("<h1 style='text-align: center; color: #16a34a;'>🚗 GoSpot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-Powered Parking Availability & Prediction Platform</p>", unsafe_allow_html=True)
st.markdown("---")


# ==========================================
# PAGE 0: LANDING PAGE
# ==========================================
if st.session_state.role is None:
    st.markdown("<h2 style='text-align: center;'>I am a:</h2>", unsafe_allow_html=True)
    st.write("") 
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="landing-btn">', unsafe_allow_html=True)
        if st.button("Driver", use_container_width=True, type="primary"):
            st.session_state.role = 'driver'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div class="landing-btn">', unsafe_allow_html=True)
        if st.button("Parking Owner", use_container_width=True, type="primary"):
            st.session_state.role = 'owner'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 1: DRIVER INTERFACE
# ==========================================
elif st.session_state.role == 'driver':
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state.role = None
            st.rerun()

    st.subheader("Find & Compare Nearby Parking Spots")

    col1, col2 = st.columns([2, 1])
    with col1:
        destination_address = st.text_input("📍 Enter your destination address:", placeholder="e.g. Intramuros Manila, Greenbelt Makati")
    with col2:
        target_city = st.selectbox("Select Target City:", ["Manila", "Makati", "Quezon City"])

    col3, col4 = st.columns(2)
    with col3:
        target_time = st.time_input("Expected Arrival Time", datetime.time(8, 0))
    with col4:
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    search_button = st.button("Search Nearest Parking", type="primary", use_container_width=True)

    if search_button or destination_address:
        base_lat, base_lon = CITY_COORDINATES[target_city]
        driver_lat, driver_lon = base_lat, base_lon
        lots_display = []
        city_code_map = {"Manila": 0, "Makati": 1, "Quezon City": 2}
        day_code_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        time_minutes = target_time.hour * 60 + target_time.minute

        for lot in st.session_state.parking_lots:
            dist_km = haversine(driver_lat, driver_lon, lot["lat"], lot["lon"])
            if model:
                features = pd.DataFrame([{'day_of_week': day_code_map[day_of_week], 'time_of_day_minute': time_minutes, 'total_capacity': lot["total_capacity"], 'city_code': city_code_map.get(lot["city"], 0)}])
                pred_occ = float(model.predict(features)[0])
                pred_occ = max(0.0, min(1.0, pred_occ))
                availability_pct = round((1 - pred_occ) * 100, 1)
            else:
                availability_pct = round((lot["current_free_slots"] / lot["total_capacity"]) * 100, 1)

            lots_display.append({**lot, "distance_km": dist_km, "predicted_avail": availability_pct})

        lots_display = sorted(lots_display, key=lambda x: x["distance_km"])
        st.markdown(f"### 🎯 Results Near: **{destination_address if destination_address else target_city}**")

        for lot in lots_display:
            with st.container():
                st.markdown(f"#### 🏢 {lot['name']}")
                st.caption(f"📍 {lot['address']} ({lot['city']}) | **Managed by: {lot['owner_name']}**")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Distance", f"{lot['distance_km']} km away")
                c2.metric("Predicted Availability", f"{lot['predicted_avail']}%")
                c3.metric("Live Free Slots", f"{lot['current_free_slots']} / {lot['total_capacity']}")
                c4.metric("Rate", f"₱{lot['price']:.2f}")

                col_btn, col_count = st.columns([1, 4])
                with col_btn:
                    if st.button(f"👍 Commend ({lot['commends']})", key=f"commend_{lot['id']}"):
                        for item in st.session_state.parking_lots:
                            if item["id"] == lot["id"]:
                                item["commends"] += 1
                                st.rerun()
                st.markdown("---")

# ==========================================
# PAGE 2: PARKING OWNER INTERFACE
# ==========================================
elif st.session_state.role == 'owner':
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state.role = None
            st.rerun()

    st.subheader("Manage & List Parking Spaces")
    
    current_owner = st.text_input("Enter your Name or Company to manage your listings:", placeholder="e.g. Bayleaf Hotel")
    
    if current_owner:
        with st.expander(f"➕ Publish a New Parking Spot as '{current_owner}'", expanded=False):
            with st.form("add_lot_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    lot_name = st.text_input("Parking Facility Name", placeholder="e.g. Sunshine 100 Basement")
                    city = st.selectbox("City", ["Manila", "Makati", "Quezon City"])
                    specific_address = st.text_input("Specific Address", placeholder="e.g. 123 Pioneer St")
                with col_b:
                    price = st.number_input("Parking Rate (₱ Flat)", min_value=10.0, value=50.0, step=5.0)
                    total_capacity = st.number_input("Total Capacity", min_value=1, value=20, step=1)
                    free_slots = st.number_input("Currently Available", min_value=0, value=10, step=1)

                submitted = st.form_submit_button("Publish Parking Spot", type="primary", use_container_width=True)

                if submitted:
                    if not lot_name or not specific_address:
                        st.error("Please fill in both the parking facility name and specific address.")
                    else:
                        base_lat, base_lon = CITY_COORDINATES[city]
                        new_lot = {
                            "id": len(st.session_state.parking_lots) + 1,
                            "owner_name": current_owner,
                            "name": lot_name,
                            "city": city,
                            "address": specific_address,
                            "lat": base_lat + np.random.uniform(-0.01, 0.01),
                            "lon": base_lon + np.random.uniform(-0.01, 0.01),
                            "price": float(price),
                            "total_capacity": int(total_capacity),
                            "current_free_slots": int(free_slots),
                            "lighting": True,
                            "cctv": True,
                            "pwd": False,
                            "commends": 0,
                            "reviews": []
                        }
                        st.session_state.parking_lots.append(new_lot)
                        st.success(f"'{lot_name}' has been successfully listed under {current_owner}!")
                        st.rerun()

    st.markdown("---")
    st.subheader("Global Platform Inventory")
    st.write("All parking spaces currently published on GoSpot, grouped by Owner.")
    
    inventory_df = pd.DataFrame([
        {
            "Owner Name": lot["owner_name"],
            "Facility Name": lot["name"],
            "City": lot["city"],
            "Address": lot["address"],
            "Rate": f"₱{lot['price']:.2f}",
            "Free / Total": f"{lot['current_free_slots']} / {lot['total_capacity']}"
        }
        for lot in st.session_state.parking_lots
    ])
    
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)
