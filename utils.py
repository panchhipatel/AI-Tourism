"""
utils.py — Helper functions: data loading, preprocessing, formatting
"""

import pandas as pd
import numpy as np
import streamlit as st
import os

DATASET_PATH = "dataset/travel_data.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def save_uploaded_dataset(uploaded_file) -> str:
    os.makedirs("dataset", exist_ok=True)
    path = f"dataset/{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Formatting / display helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_number(n) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))


def risk_badge(label: str) -> str:
    icons = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
    return icons.get(label, label)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

SEASON_MAP   = {"Winter": 0, "Spring": 1, "Summer": 2, "Monsoon": 3, "Autumn": 4}
WEATHER_MAP  = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Snowy": 3, "Windy": 4, "Foggy": 5}
DOW_MAP      = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                "Friday": 4, "Saturday": 5, "Sunday": 6}
BOOL_MAP     = {"Yes": 1, "No": 0, "TRUE": 1, "FALSE": 0}


def get_season(date) -> str:
    month = date.month
    if month in [11, 12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    else:
        return "Festive"  # October


def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """Light encoding for ML — returns a copy."""
    d = df.copy()
    for col in d.select_dtypes(include="object").columns:
        d[col] = d[col].astype("category").cat.codes
    return d


def get_crowd_label(visitors: int, df: pd.DataFrame) -> str:
    """Convert raw visitor count to Low/Medium/High relative to dataset."""
    q33 = df["Visitors_Count"].quantile(0.33)
    q66 = df["Visitors_Count"].quantile(0.66)
    if visitors <= q33:
        return "Low"
    elif visitors <= q66:
        return "Medium"
    return "High"


# ─────────────────────────────────────────────────────────────────────────────
# Place & destination helpers
# ─────────────────────────────────────────────────────────────────────────────

DESTINATION_INFO: dict = {
    "Goa": {
        "desc": "Sun, sand & sea — India's party capital.",
        "peak": ["Winter"], "moderate": ["Festive", "Summer"], "off": ["Monsoon"],
        "off_perks": "Low costs, less crowded beaches, lush scenery.",
        "off_warnings": "Heavy rain, non-operational water sports, rough seas."
    },
    "Rajasthan": {
        "desc": "Land of kings, forts and golden deserts.",
        "peak": ["Winter", "Festive"], "moderate": ["Summer"], "off": ["Monsoon"],
        "off_perks": "Emerald green forts, luxury hotels at low prices.",
        "off_warnings": "Humidity and sudden downpours, high heat in early summer."
    },
    "Kerala": {
        "desc": "God's Own Country — backwaters & spices.",
        "peak": ["Winter"], "moderate": ["Festive", "Summer"], "off": ["Monsoon"],
        "off_perks": "Ayurveda healing, waterfalls in full glory.",
        "off_warnings": "Possibility of travel disruptions due to heavy rains."
    },
    "Himachal Pradesh": {
        "desc": "Snow peaks, trekking & hill stations.",
        "peak": ["Summer", "Winter"], "moderate": ["Festive"], "off": ["Monsoon"],
        "off_perks": "Cloud-covered peaks, mist, peaceful ambience.",
        "off_warnings": "Landslide risks during heavy rains, road blockages."
    },
    "Uttarakhand": {
        "desc": "Yoga capital, Char Dham & Rishikesh.",
        "peak": ["Summer", "Festive"], "moderate": ["Winter"], "off": ["Monsoon"],
        "off_perks": "Raging rivers, mystic foggy mountains, budget stays.",
        "off_warnings": "Heavy rainfall, river floods, landslide-prone areas."
    },
    "Agra": {
        "desc": "Home of the iconic Taj Mahal.",
        "peak": ["Winter", "Festive"], "moderate": ["Monsoon"], "off": ["Summer"],
        "off_perks": "Easy access to monuments with no long queues.",
        "off_warnings": "Extremely high temperatures (up to 45°C), dehydrating heat."
    },
    "Ladakh": {
        "desc": "High-altitude desert with stunning landscapes.",
        "peak": ["Summer"], "moderate": ["Festive"], "off": ["Winter", "Monsoon"],
        "off_perks": "Unique photography opportunities of snowy terrain.",
        "off_warnings": "Freezing temperatures (-20°C), closed mountain passes."
    },
    "Andaman": {
        "desc": "Crystal clear waters & coral reefs.",
        "peak": ["Winter"], "moderate": ["Festive", "Summer"], "off": ["Monsoon"],
        "off_perks": "Extremely calm island vibe, rainforest beauty.",
        "off_warnings": "High ferry cancellation risks due to weather, limited sea sports."
    },
    "Varanasi": {
        "desc": "One of the world's oldest living cities.",
        "peak": ["Winter", "Festive"], "moderate": ["Monsoon"], "off": ["Summer"],
        "off_perks": "Peaceful river banks, late-night spiritual calm.",
        "off_warnings": "Intense heat and humidity, exhausting daytime travel."
    },
    "Mysore": {
        "desc": "City of palaces, sandalwood & silk.",
        "peak": ["Festive", "Winter"], "moderate": ["Summer"], "off": ["Monsoon"],
        "off_perks": "Lush royal gardens, rain-washed heritage structures.",
        "off_warnings": "Moderate rainfall, potential humidity."
    },
}

TRAVEL_TIPS: list = [
    "📅 Book tickets 4-6 weeks in advance for peak season.",
    "💊 Carry a basic first-aid kit and any prescription medicines.",
    "📲 Download offline maps before travelling to remote areas.",
    "🧴 Apply sunscreen and stay hydrated, especially in summer.",
    "🔒 Keep digital and physical copies of all ID documents.",
    "💰 Carry some local cash; many places don't accept cards.",
    "🌦 Check weather forecasts 3-4 days before your trip.",
    "🚗 Book transportation ahead for holiday weekends.",
    "🍽 Try local street food at hygienic, popular stalls.",
    "📸 Respect photography rules at religious sites.",
]

BUDGET_PLANS: dict = {
    "Beach":  {
        "places": ["Goa", "Andaman", "Lakshadweep", "Varkala", "Puri"],
        "tips": "Travel off-peak (Apr–Jun) for cheaper hotels. Use local buses.",
    },
    "Hill":   {
        "places": ["Shimla", "Manali", "Ooty", "Munnar", "Darjeeling"],
        "tips": "Book accommodation near the ridge for best views.",
    },
    "City":   {
        "places": ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai"],
        "tips": "Use metro rail to cut transport costs significantly.",
    },
    "Heritage": {
        "places": ["Rajasthan", "Agra", "Hampi", "Khajuraho", "Pattadakal"],
        "tips": "Combo tickets for multiple monuments save 20-30%.",
    },
    "Adventure": {
        "places": ["Rishikesh", "Ladakh", "Spiti", "Coorg", "Meghalaya"],
        "tips": "Book group packages — costs drop by 40% vs solo.",
    },
    "Wildlife": {
        "places": ["Jim Corbett", "Ranthambore", "Kaziranga", "Bandipur", "Sundarbans"],
        "tips": "Early morning safaris have better sightings.",
    },
}

CHATBOT_KB: dict = {
    "best time goa":         "The best time to visit Goa is November to February — cool, dry, and festive!",
    "goa":                   "Goa is famous for its beaches, nightlife, and Portuguese heritage. Best visited Oct–Feb.",
    "rajasthan":             "Rajasthan is the land of forts and deserts. Visit Oct–Mar for pleasant weather.",
    "kerala":                "Kerala offers backwaters, Ayurveda, and lush greenery. Best from Sep–Mar.",
    "himachal":              "Himachal Pradesh has Manali, Shimla, and Spiti. Snow lovers should visit Dec–Feb.",
    "budget":                "Start with a clear budget. Split costs: 40% stay, 30% transport, 20% food, 10% experiences.",
    "overcrowding":          "To avoid crowds, travel on weekdays, visit in shoulder seasons, and reach attractions early.",
    "packing":               "Pack light! Carry basics: ID proof, weather-appropriate clothes, power bank, and a first-aid kit.",
    "visa":                  "For international tourists, India offers e-Visa for 167+ countries. Apply at indianvisaonline.gov.in.",
    "food":                  "India has diverse cuisine. Try local dhabas for authentic food. Always drink bottled water.",
    "transport":             "India has trains, buses, flights, and cabs. IRCTC is the best for train booking.",
    "safety":                "India is generally safe. Keep your belongings secure, use registered taxis, and share your itinerary.",
    "hello":                 "Hello! 👋 I'm your AI travel assistant. Ask me about destinations, budgets, tips, or planning!",
    "hi":                    "Hi there! 🌍 How can I help you plan your next trip?",
    "help":                  "I can help with: destination info, best time to visit, budget planning, packing tips, safety advice, and more!",
}
