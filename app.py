import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import math
import json
import os
import base64
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GoSpot | Smart Parking Platform", page_icon="🚗", layout="wide")

# --- IMAGE TO BASE64 HELPER (FOR LOCAL BACKGROUND) ---
def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode()}"
    except Exception:
        pass
    return ""

map_bg = get_base64_image("map.jpg")

# --- CUSTOM CSS: BIGGER BUTTONS, GREEN THEME, TEXT COLOR FIX ---
st.markdown(f"""
    <style>
    :root {{ --primary-color: #16a34a; }}
    
    /* Force dark text for readability on light backgrounds */
    .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: #1e293b !important;
    }}
    
    /* Headers Green Override */
    h1, h2, h3, h4 {{ color: #16a34a !important; font-weight: bold !important; }}

    /* FIX: Force white text specifically inside input boxes and dropdowns */
    div[data-baseweb="input"] input, 
    div[data-baseweb="select"] div,
    input {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}
    
    /* Ensure metric values retain primary color */
    div[data-testid="stMetricValue"] > div {{
        color: #16a34a !important;
        -webkit-text-fill-color: #16a34a !important;
    }}

    /* Real Local Map Background with a frosted glass overlay */
    .stApp {{
        background-image: linear-gradient(rgba(248, 250, 252, 0.85), rgba(248, 250, 252, 0.95)), 
                          url("{map_bg}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Make the main content area stand out against the map background */
    [data-testid="stAppViewBlockContainer"] {{
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-top: 1rem;
        margin-bottom: 2rem;
    }}

    /* Button Styling */
    div.stButton > button {{
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        padding: 12px 24px !important;
        transition: all 0.2s ease-in-out !important;
        border: 2px solid #16a34a !important;
        color: #16a34a !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #16a34a !important;
    }}
    div.stButton > button[kind="primary"], div.stButton > button[kind="primary"] * {{
        background-color: #16a34a !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.3) !important;
        -webkit-text-fill-color: #ffffff !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: #15803d !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.4) !important;
    }}
    
    .landing-btn div.stButton > button {{
        height: 100px !important;
        font-size: 28px !important;
        letter-spacing: 0.5px;
    }}
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

# --- DISTANCE UTILITY (HAVERSINE) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 
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

# --- DATABASE MANAGEMENT (JSON) ---
DB_FILE = "gospot_db.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    else:
        default_db = {
            "lots": [
                {
                    "id": 1, "owner_name": "Bayleaf Hotel", "name": "Bayleaf Hotel Parking",
                    "city": "Manila", "address": "Muralla St corner Victoria St, Intramuros, Manila",
                    "lat": 14.5898, "lon": 120.9754, "price": 150.0, "total_capacity": 80, 
                    "current_free_slots": 25, "lighting": True, "cctv": True, "pwd": True, "commends": 42,
                    "reviews": []
                },
                {
                    "id": 2, "owner_name": "SM Prime Holdings", "name": "SM City Manila Multi-Level",
                    "city": "Manila", "address": "Natividad Almeda-Lopez St, Ermita, Manila",
                    "lat": 14.5874, "lon": 120.9816, "price": 50.0, "total_capacity": 300, 
                    "current_free_slots": 85, "lighting": True, "cctv": True, "pwd": True, "commends": 112,
                    "reviews": []
                },
                {
                    "id": 3, "owner_name": "Ayala Property Mgmt", "name": "Legazpi Village Commercial Parking",
                    "city": "Makati", "address": "Salcedo St, Legazpi Village, Makati",
                    "lat": 14.5532, "lon": 121.0185, "price": 80.0, "total_capacity": 60, 
                    "current_free_slots": 22, "lighting": True, "cctv": True, "pwd": True, "commends": 34,
                    "reviews": []
                },
                {
                    "id": 4, "owner_name": "Ayala Property Mgmt", "name": "Greenbelt 2 Basement Parking",
                    "city": "Makati", "address": "Esperanza St, Greenbelt, Makati",
                    "lat": 14.5524, "lon": 121.0189, "price": 75.0, "total_capacity": 150, 
                    "current_free_slots": 40, "lighting": True, "cctv": True, "pwd": True, "commends": 89,
                    "reviews": []
                },
                {
                    "id": 5, "owner_name": "QC LGU Admin", "name": "Timog Avenue Secure Lot",
                    "city": "Quezon City", "address": "Timog Ave cor. Tomas Morato, Quezon City",
                    "lat": 14.6360, "lon": 121.0345, "price": 60.0, "total_capacity": 30, 
                    "current_free_slots": 5, "lighting": True, "cctv": True, "pwd": False, "commends": 9,
                    "reviews": []
                },
                {
                    "id": 6, "owner_name": "Ayala Malls", "name": "Trinoma Mindanao Ave Open Parking",
                    "city": "Quezon City", "address": "Mindanao Ave, Triangle Park, Quezon City",
                    "lat": 14.6534, "lon": 121.0345, "price": 50.0, "total_capacity": 200, 
                    "current_free_slots": 120, "lighting": True, "cctv": True, "pwd": True, "commends": 55,
                    "reviews": []
                }
            ],
            "history": {}
        }
        save_db(default_db)
        return default_db

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=4)

# --- GEOCODING HELPER ---
@st.cache_data
def get_coordinates_from_address(address, city):
    try:
        geolocator = Nominatim(user_agent="gospot_prototype")
        location = geolocator.geocode(f"{address}, {city}, Metro Manila, Philippines")
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return None

# --- SESSION STATE INITIALIZATION ---
if "role" not in st.session_state:
    st.session_state.role = None  
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "db" not in st.session_state:
    st.session_state.db = load_db()

# --- APP HEADER ---
col_left, col_center, col_right = st.columns([2, 1, 2])
with col_center:
    try:
        if os.path.exists("logo.png"):
            st.image("logo.png", use_container_width=True) 
        else:
            st.markdown("<h1 style='text-align: center;'>🚗 GoSpot</h1>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<h1 style='text-align: center;'>🚗 GoSpot</h1>", unsafe_allow_html=True)
        
st.markdown("<p style='text-align: center; font-weight: bold;'>AI-Powered Parking Availability Platform</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# PAGE 0: LANDING PAGE
# ==========================================
if st.session_state.role is None:
    st.markdown("<h2 style='text-align: center; background: white; padding: 10px 20px; border-radius: 10px; display: inline-block; margin: 0 auto; border: 2px solid #16a34a;'>I am a:</h2>", unsafe_allow_html=True)
    st.write("") 
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="landing-btn">', unsafe_allow_html=True)
        if st.button("Driver", use_container_width=True, type="primary"):
            st.session_state.role = 'driver'
            st.session_state.db = load_db()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div class="landing-btn">', unsafe_allow_html=True)
        if st.button("Parking Owner", use_container_width=True, type="primary"):
            st.session_state.role = 'owner'
            st.session_state.db = load_db()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 1: LOGIN SCREEN
# ==========================================
elif st.session_state.role is not None and st.session_state.user_name is None:
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back"):
            st.session_state.role = None
            st.rerun()

    st.markdown(f"<h3 style='text-align: center;'>{st.session_state.role.capitalize()} Login</h3>", unsafe_allow_html=True)
    
    col_log1, col_log2, col_log3 = st.columns([1, 2, 1])
    with col_log2:
        st.write("Please enter your name to access your dashboard:")
        entered_name = st.text_input("Full Name or Handle", placeholder="e.g. Juan Dela Cruz")
        if st.button("Log In", type="primary", use_container_width=True):
            if entered_name.strip():
                st.session_state.user_name = entered_name.strip()
                st.rerun()
            else:
                st.error("Please enter a valid name.")

# ==========================================
# PAGE 2: DRIVER DASHBOARD
# ==========================================
elif st.session_state.role == 'driver' and st.session_state.user_name is not None:
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Log Out", use_container_width=True):
            st.session_state.role = None
            st.session_state.user_name = None
            st.rerun()

    st.subheader(f"👋 Welcome, {st.session_state.user_name}!")
    
    tab1, tab2 = st.tabs(["🔍 Find Parking", "🕒 Recent Parkings"])
    
    with tab1:
        st.write("### 1. Set Your Destination")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            destination_address = st.text_input("📍 Enter destination address manually:", placeholder="e.g. Rizal Park")
        with col2:
            target_city = st.selectbox("Destination City:", ["Manila", "Makati", "Quezon City"])

        # Determine the base map coordinates dynamically
        base_lat, base_lon = CITY_COORDINATES[target_city]
        has_custom_pin = False

        if destination_address:
            coords = get_coordinates_from_address(destination_address, target_city)
            if coords:
                base_lat, base_lon = coords
                has_custom_pin = True

        st.info("The map syncs with your typed address automatically. You can also click directly on the map to drop a pin.")
        
        # INTERACTIVE MAP WIDGET
        m = folium.Map(location=[base_lat, base_lon], zoom_start=15)
        
        # Add target marker if user typed a valid address
        if has_custom_pin:
            folium.Marker(
                [base_lat, base_lon], 
                popup=destination_address,
                tooltip="📍 Your Target Destination",
                icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
            ).add_to(m)

        # Map existing lots
        for lot in st.session_state.db["lots"]:
            folium.Marker(
                [lot["lat"], lot["lon"]], 
                popup=f"{lot['name']} - ₱{lot['price']}",
                tooltip="🅿️ " + lot["name"],
                icon=folium.Icon(color="green", icon="info-sign")
            ).add_to(m)

        map_data = st_folium(m, height=400, use_container_width=True)
        
        # If the map is clicked, it overrides the search base
        if map_data and map_data.get("last_clicked"):
            base_lat = map_data["last_clicked"]["lat"]
            base_lon = map_data["last_clicked"]["lng"]
            st.success(f"📍 Manual Map Pin Dropped at: {base_lat:.4f}, {base_lon:.4f}")

        st.write("### 2. Set Arrival Details")
        col3, col4 = st.columns(2)
        with col3:
            target_time = st.time_input("Expected Arrival Time", datetime.time(8, 0))
        with col4:
            day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

        search_button = st.button("Search Nearest Parking", type="primary", use_container_width=True)

        if search_button or destination_address or (map_data and map_data.get("last_clicked")):
            lots_display = []
            day_code_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            city_code_map = {"Manila": 0, "Makati": 1, "Quezon City": 2}
            time_minutes = target_time.hour * 60 + target_time.minute

            for lot in st.session_state.db["lots"]:
                dist_km = haversine(base_lat, base_lon, lot["lat"], lot["lon"])
                if model:
                    features = pd.DataFrame([{'day_of_week': day_code_map[day_of_week], 'time_of_day_minute': time_minutes, 'total_capacity': lot["total_capacity"], 'city_code': city_code_map.get(lot["city"], 0)}])
                    pred_occ = float(model.predict(features)[0])
                    pred_occ = max(0.0, min(1.0, pred_occ))
                    availability_pct = round((1 - pred_occ) * 100, 1)
                else:
                    availability_pct = round((lot["current_free_slots"] / lot["total_capacity"]) * 100, 1)

                lots_display.append({**lot, "distance_km": dist_km, "predicted_avail": availability_pct})

            lots_display = sorted(lots_display, key=lambda x: x["distance_km"])
            
            location_label = destination_address if destination_address else "Pinned Location"
            st.markdown(f"### 🎯 Results Near: **{location_label}**")

            for lot in lots_display:
                with st.container():
                    st.markdown(f"#### 🏢 {lot['name']}")
                    st.caption(f"📍 {lot['address']} ({lot['city']}) | **Managed by: {lot.get('owner_name', 'Independent')}**")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Distance", f"{lot['distance_km']} km away")
                    c2.metric("Predicted Availability", f"{lot['predicted_avail']}%")
                    c3.metric("Live Free Slots", f"{lot['current_free_slots']} / {lot['total_capacity']}")
                    c4.metric("Rate", f"₱{lot['price']:.2f}")

                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.button(f"🚙 Park Here", key=f"park_{lot['id']}", type="primary"):
                            user_history = st.session_state.db["history"].get(st.session_state.user_name, [])
                            user_history.insert(0, {
                                "lot_id": lot["id"], 
                                "lot_name": lot["name"],
                                "date": datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
                            })
                            st.session_state.db["history"][st.session_state.user_name] = user_history
                            save_db(st.session_state.db)
                            st.success("Parking location saved to your Recent History!")
                    with col_act2:
                        if st.button(f"👍 Commend ({lot['commends']})", key=f"commend_{lot['id']}"):
                            for item in st.session_state.db["lots"]:
                                if item["id"] == lot["id"]:
                                    item["commends"] += 1
                            save_db(st.session_state.db)
                            st.rerun()
                    st.markdown("---")

    with tab2:
        st.write("### Your Parking History")
        user_history = st.session_state.db["history"].get(st.session_state.user_name, [])
        
        if not user_history:
            st.info("You haven't parked anywhere recently. Search and click 'Park Here' to build your history!")
        else:
            for record in user_history:
                st.markdown(f"**🏢 {record['lot_name']}**")
                st.caption(f"📅 Visited on: {record['date']}")
                st.markdown("---")

# ==========================================
# PAGE 3: PARKING OWNER DASHBOARD
# ==========================================
elif st.session_state.role == 'owner' and st.session_state.user_name is not None:
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Log Out", use_container_width=True):
            st.session_state.role = None
            st.session_state.user_name = None
            st.rerun()

    st.subheader(f"🏢 Owner Dashboard: {st.session_state.user_name}")
    
    with st.expander(f"➕ Publish a New Parking Spot", expanded=False):
        with st.form("add_lot_form", clear_on_submit=True):
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
                    new_id = max([L["id"] for L in st.session_state.db["lots"]] + [0]) + 1
                    new_lot = {
                        "id": new_id,
                        "owner_name": st.session_state.user_name,
                        "name": lot_name,
                        "city": city,
                        "address": specific_address,
                        "lat": base_lat + np.random.uniform(-0.01, 0.01),
                        "lon": base_lon + np.random.uniform(-0.01, 0.01),
                        "price": float(price),
                        "total_capacity": int(total_capacity),
                        "current_free_slots": int(free_slots),
                        "lighting": True, "cctv": True, "pwd": False,
                        "commends": 0, "reviews": []
                    }
                    st.session_state.db["lots"].append(new_lot)
                    save_db(st.session_state.db)
                    st.success(f"'{lot_name}' has been successfully listed!")
                    st.rerun()

    st.markdown("---")
    st.subheader("Your Managed Listings")
    
    my_lots = [lot for lot in st.session_state.db["lots"] if lot.get("owner_name") == st.session_state.user_name]
    
    if not my_lots:
        st.info("You haven't published any parking facilities yet. Use the form above to add your first lot!")
    else:
        for lot in my_lots:
            with st.container():
                st.markdown(f"#### 🏢 {lot['name']}")
                st.caption(f"📍 {lot['address']} | Capacity: {lot['total_capacity']} | Rate: ₱{lot['price']:.2f}")
                
                with st.expander("✏️ Edit or Delete Listing"):
                    with st.form(f"edit_form_{lot['id']}"):
                        new_price = st.number_input("Update Rate (₱)", value=float(lot['price']), step=5.0)
                        new_capacity = st.number_input("Update Total Capacity", value=int(lot['total_capacity']), step=1)
                        new_free = st.number_input("Update Live Available Slots", value=int(lot['current_free_slots']), step=1)
                        
                        update_submit = st.form_submit_button("Save Changes")
                        if update_submit:
                            lot['price'] = new_price
                            lot['total_capacity'] = new_capacity
                            lot['current_free_slots'] = new_free
                            save_db(st.session_state.db)
                            st.success("Listing updated successfully!")
                            st.rerun()
                    
                    if st.button(f"🗑️ Delete '{lot['name']}'", key=f"del_{lot['id']}"):
                        st.session_state.db["lots"] = [L for L in st.session_state.db["lots"] if L["id"] != lot["id"]]
                        save_db(st.session_state.db)
                        st.warning("Listing deleted.")
                        st.rerun()
            st.write("")
