"""
model.py — ML models: Regression-based Crowd Level Determination
Ensures crowd levels are relative to each specific place's average.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import mean_absolute_error, r2_score
from utils import load_data, get_season, DESTINATION_INFO

# ─────────────────────────────────────────────────────────────────────────────
# Advanced Travel Intelligence Logic
# ─────────────────────────────────────────────────────────────────────────────

def predict_intelligence(destination: str, travel_date):
    """
    Advanced AI Travel Intelligence System logic:
    Predicts demand and overcrowding risk based on season-destination matching.
    """
    if not travel_date:
        return None
        
    season = get_season(travel_date)
    info = DESTINATION_INFO.get(destination)
    
    # Base structure
    result = {
        "destination": destination,
        "month": travel_date.strftime("%B"),
        "season": season,
    }
    
    if not info:
        result.update({
            "demand": "Medium", "risk": "Medium",
            "reason": f"No specific seasonal patterns found for {destination}",
            "suggestion": f"This destination is generally good to visit! Proceed with your travel plans and enjoy exploring {destination}."
        })
        return result
        
    peak = info.get("peak", [])
    moderate = info.get("moderate", [])
    off = info.get("off", [])
    
    if season in peak:
        demand, risk = "High", "High"
        reason = f"{season} is peak season for {destination}"
        suggestion = f"⚠️ HIGH OVERCROWDING RISK! Expect large crowds and higher prices. Consider booking everything well in advance or visiting early morning."
    elif season in moderate:
        demand, risk = "Medium", "Medium"
        reason = f"{season} is a moderate/balanced season for {destination}"
        suggestion = f"Balanced trip expected. Good time to visit for a mix of experience and manageable crowds."
    else:
        demand, risk = "Low", "Low"
        reason = f"{season} is off-season for {destination} tourism"
        perks = info.get("off_perks", "low cost, less crowd")
        warn = info.get("off_warnings", "limited activities")
        suggestion = f"Good for budget travel, but {warn}"

    result.update({
        "demand": demand,
        "risk": risk,
        "reason": reason,
        "suggestion": suggestion
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame):
    """Return (X, y_demand, feature_cols, encoders, place_means)."""
    d = df.copy()

    # Pre-calculate place means for relative comparison
    place_means = d.groupby("Place_Name")["Visitors_Count"].mean().to_dict()
    global_mean = d["Visitors_Count"].mean()

    # Categorical Encoding
    cat_cols = ["Place_Name", "Location_State", "Place_Type", "Season", 
                "Day_of_Week", "Weather_Type", "Is_Weekend"]
    
    encoders = {}
    for col in cat_cols:
        if col in d.columns:
            le = LabelEncoder()
            # Handle unknown values by adding a dedicated label
            unique_vals = list(d[col].astype(str).unique()) + ["Unknown"]
            le.fit(unique_vals)
            d[col] = le.transform(d[col].astype(str))
            encoders[col] = le

    # Numeric Features
    num_cols = ["Google_Rating", "Ticket_Price", "Review_Count_Lakhs"]
    for col in num_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)

    feature_cols = cat_cols + num_cols
    feature_cols = [c for c in feature_cols if c in d.columns]

    d = d.dropna(subset=feature_cols + ["Visitors_Count"])

    X = d[feature_cols]
    y = d["Visitors_Count"]

    return X, y, feature_cols, encoders, place_means, global_mean


# ─────────────────────────────────────────────────────────────────────────────
# Model cache
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_models():
    """Train Regression model for demand; use it to derive crowd labels."""
    df = load_data()
    if df.empty:
        return None

    X, y, feature_cols, encoders, place_means, global_mean = _prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Use a strong Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train)
    
    y_pred = rf_reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return {
        "rf_reg":       rf_reg,
        "encoders":     encoders,
        "feature_cols": feature_cols,
        "place_means":  place_means,
        "global_mean":  global_mean,
        "metrics": {
            "r2": round(r2, 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 2)
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction API
# ─────────────────────────────────────────────────────────────────────────────

def predict(state: str, place_type: str, season: str,
            weather: str, tourists: int, is_weekend: bool,
            ticket_price: int, rating: float, travel_date=None,
            place_name: str = "Unknown"):
    
    models = train_models()
    if models is None:
        return None

    # Precise Day name
    day_name = travel_date.strftime("%A") if travel_date else ("Saturday" if is_weekend else "Wednesday")
    
    # Feature Input
    raw = {
        "Place_Name":           place_name,
        "Location_State":       state,
        "Place_Type":           place_type,
        "Season":               season,
        "Day_of_Week":          day_name,
        "Weather_Type":         weather,
        "Is_Weekend":           "Yes" if is_weekend else "No",
        "Google_Rating":        rating,
        "Ticket_Price":         ticket_price,
        "Review_Count_Lakhs":   1.0,
    }

    enc = models["encoders"]
    row = {}
    for col in models["feature_cols"]:
        val = raw.get(col, "Unknown")
        if col in enc:
            if str(val) not in enc[col].classes_:
                val = "Unknown"
            val = enc[col].transform([str(val)])[0]
        row[col] = val

    X_input = pd.DataFrame([row])[models["feature_cols"]]

    # Predict Demand
    demand = int(models["rf_reg"].predict(X_input)[0])
    
    # Compare with Place Mean to determine Crowd Level
    avg = models["place_means"].get(place_name, models["global_mean"])
    
    # Thresholds for relative labeling (Refined and balanced)
    # High: > 20% above average
    # Low:  < 15% below average
    if demand > 1.20 * avg:
        label = "High"
    elif demand < 0.85 * avg:
        label = "Low"
    else:
        label = "Medium"

    return {
        "demand":       demand,
        "crowd_label":  label,
        "avg_visitors": int(avg)
    }
