"""
app.py — AI-Driven Tourism: Main Streamlit entry point
Run:  streamlit run app.py
"""

import os, io, random
from datetime import date
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express       as px
import plotly.graph_objects as go
from streamlit_option_menu  import option_menu

from auth  import show_login_page, logout, update_profile
from model import train_models, predict, predict_intelligence
from utils import (load_data, save_uploaded_dataset, fmt_number, risk_badge,
                   DESTINATION_INFO, TRAVEL_TIPS, BUDGET_PLANS, CHATBOT_KB,
                   get_crowd_label, get_season)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Driven Tourism",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

def chatbot_response(query: str) -> str:
    q = query.lower().strip()
    for key, reply in CHATBOT_KB.items():
        if key in q:
            return reply
    # Fallback generic
    df = load_data()
    if not df.empty and any(s.lower() in q for s in df["Location_State"].unique()):
        for s in df["Location_State"].unique():
            if s.lower() in q:
                sub = df[df["Location_State"] == s]
                avg_v = int(sub["Visitors_Count"].mean())
                places = sub["Place_Name"].unique()[:3]
                return (f"**{s}** sees ~{fmt_number(avg_v)} visitors on average. "
                        f"Top spots: {', '.join(places)}. "
                        f"Best season: {sub['Season'].mode()[0]}.")
    tips = random.choice(TRAVEL_TIPS)
    return (f"Great question about **{query}**! Here's a quick tip: {tips}\n\n"
            "Try asking: 'best time Goa', 'budget planning', 'Rajasthan', 'safety tips', etc.")


def show_chatbot():
    st.subheader("🤖 AI Travel Chatbot")
    st.caption("Ask about destinations, best times, budgets, packing, safety, and more!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("assistant", "Hello! 👋 I'm your AI travel assistant. Ask me anything about travel in India!")
        ]

    # render history using native chat_message
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_msg = st.chat_input("Ask me anything about your trip...")

    if user_msg:
        st.session_state.chat_history.append(("user", user_msg))
        with st.chat_message("user"):
            st.write(user_msg)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = chatbot_response(user_msg)
                st.write(reply)
                st.session_state.chat_history.append(("assistant", reply))

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = [("assistant", "Hello! 👋 Starting fresh — what would you like to know?")]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  BUDGET TRIP PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

def show_budget_planner():
    st.subheader("💰 Budget Trip Planner")
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input("Total Budget (₹)", min_value=1000, max_value=500000,
                                  value=15000, step=1000)
    with col2:
        loc_type = st.selectbox("Destination Type", list(BUDGET_PLANS.keys()))
    with col3:
        days = st.number_input("Number of Days", min_value=1, max_value=30, value=5)

    travel_date = st.date_input("Travel Date")

    if st.button("🗺 Plan My Trip", use_container_width=True):
        plan = BUDGET_PLANS[loc_type]
        dest = random.choice(plan["places"])
        per_day = budget / days
        stay_budget    = round(budget * 0.40)
        transport_budget = round(budget * 0.30)
        food_budget    = round(budget * 0.20)
        exp_budget     = round(budget * 0.10)

        st.success(f"✅ Plan ready for **{dest}** ({days} days, ₹{budget:,} budget)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏨 Stay", f"₹{stay_budget:,}")
        col2.metric("🚗 Transport", f"₹{transport_budget:,}")
        col3.metric("🍽 Food", f"₹{food_budget:,}")
        col4.metric("🎭 Activities", f"₹{exp_budget:,}")

        st.divider()
        st.subheader("📅 Day-wise Itinerary")
        activities = [
            "Explore local market & street food",
            "Visit top tourist attractions",
            "Nature walk / adventure activity",
            "Local cultural experience / museum",
            "Shopping & rest day",
            "Day trip to nearby attraction",
            "Sunset point / viewpoint visit",
            "Explore local heritage & history",
        ]
        for d in range(1, int(days) + 1):
            act = activities[(d - 1) % len(activities)]
            st.write(f"**Day {d}** — {act}  |  Budget: ₹{int(per_day):,}")

        st.divider()
        st.info(f"💡 **Pro Tip:** {plan['tips']}")
        for tip in random.sample(TRAVEL_TIPS, 3):
            st.write(f"• {tip}")

        # cost pie using Plotly
        fig = px.pie(
            values=[stay_budget, transport_budget, food_budget, exp_budget],
            names=["Stay", "Transport", "Food", "Activities"],
            title="Budget Breakdown",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLACE SEARCH (User)
# ═══════════════════════════════════════════════════════════════════════════════

def show_place_search():
    st.subheader("🔍 Place Search & Details")
    df = load_data()
    if df.empty:
        st.warning("Dataset not loaded.")
        return

    query = st.text_input("Search destination / state / place name", placeholder="e.g. Goa, Temple, Beach…")
    if query:
        mask = (
            df["Location_State"].str.contains(query, case=False, na=False) |
            df["Place_Name"].str.contains(query, case=False, na=False) |
            df["Place_Type"].str.contains(query, case=False, na=False)
        )
        results = df[mask].drop_duplicates("Place_Name").head(10)
        if results.empty:
            st.warning("No places found. Try a different search term.")
        else:
            for _, row in results.iterrows():
                with st.expander(f"📍 {row['Place_Name']}  —  {row['Location_State']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("⭐ Google Rating", row["Google_Rating"])
                    c2.metric("👥 Avg Visitors", fmt_number(row["Visitors_Count"]))
                    c3.metric("🎟 Ticket Price", f"₹{row['Ticket_Price']}")
                    st.write(f"**Type:** {row['Place_Type']}  |  **Zone:** {row['Zone']}  |  **Best Season:** {row['Season']}")
                    crowd = get_crowd_label(row["Visitors_Count"], df)
                    st.write(f"**Crowd Level:** {crowd}")
                    info = DESTINATION_INFO.get(row["Location_State"], {})
                    if info:
                        st.write(f"**Best Time:** {info.get('best_time', '-')}")
                        st.info(info.get("tips", ""))


# ═══════════════════════════════════════════════════════════════════════════════
#  PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def show_profile():
    st.subheader("👤 Profile & Settings")
    u = st.session_state.user
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(f"**{u.get('full_name','—')}**\n\nRole: {u.get('role','—')}")
    with col2:
        st.write(f"**Username:** {u.get('username','—')}")
        st.write(f"**Email:** {u.get('email','—')}")
        st.write(f"**Role:** {u.get('role','—')}")

    st.divider()
    st.subheader("✏️ Edit Profile")
    with st.form("edit_profile"):
        c1, c2 = st.columns(2)
        with c1:
            new_username  = st.text_input("Username",  value=u.get("username",""))
            new_full_name = st.text_input("Full Name", value=u.get("full_name",""))
        with c2:
            new_email    = st.text_input("Email",        value=u.get("email",""))
            new_password = st.text_input("New Password (leave blank to keep)", type="password")
        save = st.form_submit_button("💾 Save Changes", use_container_width=True)

    if save:
        ok, msg = update_profile(u["username"], new_username, new_password, new_email, new_full_name)
        if ok:
            st.success(msg)
        else:
            st.error(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAVEL AGENT — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def show_overview():
    st.subheader("📊 Dataset Overview")
    df = load_data()
    if df.empty:
        st.warning("No dataset loaded. Go to **Dataset** tab to upload one.")
        return

    c1, c2, c3, c4 = st.columns(4)
    total_visitors = df["Visitors_Count"].sum()
    total_revenue  = df["Revenue"].sum() if "Revenue" in df.columns else (df["Visitors_Count"] * df["Ticket_Price"]).sum()
    avg_rating     = df["Google_Rating"].mean()
    unique_places  = df["Place_Name"].nunique()

    c1.metric("👥 Total Visitors", fmt_number(total_visitors))
    c2.metric("💰 Total Revenue", f"₹{fmt_number(total_revenue)}")
    c3.metric("⭐ Avg Rating", f"{avg_rating:.2f}")
    c4.metric("📍 Destinations", unique_places)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visitor Distribution")
        fig = px.histogram(df, x="Visitors_Count", nbins=40, title="Histogram of Visitor Counts")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Season Distribution")
        fig = px.pie(df, names="Season", title="Records by Season")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Data")
    st.dataframe(df.sample(min(10, len(df))), use_container_width=True)

    st.subheader("Column Summary")
    summary = pd.DataFrame({
        "Column": df.columns,
        "Type":   df.dtypes.values,
        "Non-Null": df.count().values,
        "Missing":  df.isnull().sum().values,
        "Unique":   df.nunique().values,
    })
    st.dataframe(summary, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAVEL AGENT — DATASET TAB
# ═══════════════════════════════════════════════════════════════════════════════

def show_dataset_tab():
    st.subheader("📂 Dataset Management")
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
    if uploaded:
        with st.spinner("Saving…"):
            path = save_uploaded_dataset(uploaded)
            load_data.clear()
        st.success(f"Dataset saved: `{path}`")

    df = load_data()
    if df.empty:
        st.info("No dataset loaded yet.")
        return

    st.divider()
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        states = ["All"] + sorted(df["Location_State"].dropna().unique().tolist())
        state_filter = st.selectbox("State", states)
    with col2:
        seasons = ["All"] + sorted(df["Season"].dropna().unique().tolist())
        season_filter = st.selectbox("Season", seasons)
    with col3:
        min_v, max_v = int(df["Visitors_Count"].min()), int(df["Visitors_Count"].max())
        visitor_range = st.slider("Visitors Count", min_v, max_v, (min_v, max_v))

    fdf = df.copy()
    if state_filter  != "All": fdf = fdf[fdf["Location_State"] == state_filter]
    if season_filter != "All": fdf = fdf[fdf["Season"]         == season_filter]
    fdf = fdf[(fdf["Visitors_Count"] >= visitor_range[0]) &
              (fdf["Visitors_Count"] <= visitor_range[1])]

    st.info(f"Showing {len(fdf):,} of {len(df):,} records")
    st.dataframe(fdf, use_container_width=True)

    csv_bytes = fdf.to_csv(index=False).encode()
    st.download_button("⬇ Download Filtered CSV", csv_bytes, "filtered_data.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAVEL AGENT — DASHBOARD / CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def show_dashboard():
    st.subheader("📈 Visual Dashboard")
    df = load_data()
    if df.empty:
        st.warning("Dataset not available.")
        return

    if "Date" in df.columns and df["Date"].notna().any():
        trend = df.groupby(df["Date"].dt.to_period("M").astype(str))["Visitors_Count"].sum().reset_index()
        trend.columns = ["Month", "Visitors"]
        fig = px.line(trend, x="Month", y="Visitors", title="Monthly Tourist Trends", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top = df.groupby("Location_State")["Visitors_Count"].sum().nlargest(10).reset_index()
        fig = px.bar(top, x="Visitors_Count", y="Location_State", orientation="h",
                     title="Top 10 States by Total Visitors", color="Visitors_Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        season_df = df["Season"].value_counts().reset_index()
        season_df.columns = ["Season", "Count"]
        fig = px.pie(season_df, names="Season", values="Count", title="Season-wise Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔥 Correlation Heatmap")
    num_df = df[["Visitors_Count", "Google_Rating", "Review_Count_Lakhs",
                 "Ticket_Price", "Revenue"]].dropna()
    corr = num_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation Matrix of Numeric Features")
    st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        tt = df["Tourist_Type"].value_counts().reset_index()
        tt.columns = ["Type", "Count"]
        fig = px.bar(tt, x="Type", y="Count", title="Tourist Type Breakdown", color="Type")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        zone_df = df["Zone"].value_counts().reset_index()
        zone_df.columns = ["Zone", "Count"]
        fig = px.bar(zone_df, x="Zone", y="Count", title="Zone-wise Records", color="Zone")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION TAB
# ═══════════════════════════════════════════════════════════════════════════════

def show_prediction():
    st.subheader("🔮 Trip Overcrowding & Experience Predictor")
    df = load_data()
    if df.empty:
        st.warning("Dataset not available.")
        return


    st.divider()
    st.subheader("Plan Your Trip")
    col1, col2, col3 = st.columns(3)
    with col1:
        places_list = sorted(df["Place_Name"].dropna().unique().tolist())
        place_input = st.selectbox("Where do you want to go?", places_list)
        
    with col2:
        num_users = st.number_input("How many people are going?", 1, 1000, 2)
        
    with col3:
        travel_date = st.date_input("When do you plan to travel?", min_value=date.today())

    if st.button("🔍 Predict Experience", use_container_width=True):
        match = df[df["Place_Name"] == place_input].iloc[0]
        state = match["Location_State"]
        p_type = match["Place_Type"]
        ticket_price = match["Ticket_Price"]
        rating = match["Google_Rating"]
        
        season = get_season(travel_date)
        is_weekend = travel_date.weekday() >= 5
        weather = "Sunny"
        
        with st.spinner("Analyzing data…"):
            # ML Model Prediction
            result = predict(state, p_type, season, weather,
                             num_users, is_weekend, ticket_price, rating, 
                             travel_date=travel_date, place_name=place_input)
            
            # Rule-based Travel Intelligence Prediction
            intel = predict_intelligence(state, travel_date)

        if result is None:
            st.error("Prediction failed.")
            return

        st.divider()
        
        if intel:
            st.subheader(f"Results for {place_input}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("📊 Predicted Visitors", fmt_number(result["demand"]))
            c2.metric("📅 Typical Average", fmt_number(result.get("avg_visitors", 0)))
            
            with c3:
                # Emoji map for Risk
                e_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                risk_emoji = e_map.get(intel['risk'], "")
                st.write(f"### Crowd Risk")
                st.write(f"**{intel['risk']} {risk_emoji}**")
                
            st.divider()
            st.subheader("AI Suggestion")
            
            # Banner based on risk level
            msg = f"{intel['reason'].upper()}!"
            if intel['risk'] == "High":
                st.error(f"⚠️ {msg}")
            elif intel['risk'] == "Medium":
                st.warning(f"🟡 {msg}")
            else:
                st.success(f"✅ {msg}")
                
            st.write(intel['suggestion'])
            st.caption(f"**Season:** {intel['season']}  |  **Month:** {intel['month']}")

        st.divider()
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("💡 Travel Tips")
            info = DESTINATION_INFO.get(state, {})
            if info:
                st.write(f"**Best Time:** {info.get('best_time', 'N/A')}")
                st.info(info.get('tips', 'No specific tips for this state.'))
            else:
                st.write("• Research local customs before visiting.")
                st.write("• Keep a copy of your ID handy.")
            
            st.subheader("🎒 What to Carry")
            items = ["Water bottle", "Power bank", "ID Proof", "Comfortable shoes"]
            if season == "Winter": items += ["Light jacket", "Moisturizer"]
            elif season == "Summer": items += ["Sunscreen", "Sunglasses", "Cotton clothes"]
            elif season == "Monsoon": items += ["Umbrella", "Raincoat", "Waterproof bag"]
            
            if p_type in ["Beach", "Island"]: items += ["Swimwear", "Flip flops"]
            elif p_type in ["Temple", "Heritage Site"]: items += ["Modest clothing", "Hat"]
            
            for item in items:
                st.write(f"• {item}")

        with colB:
            if intel and intel['risk'] in ["High", "Medium"]:
                st.subheader("🔄 Recommended Alternatives")
                st.write(f"Since {place_input} might be crowded, consider these nearby or similar spots in {state}:")
                alts = df[(df["Location_State"] == state) & (df["Place_Name"] != place_input)].head(3)
                if alts.empty:
                    alts = df[df["Place_Name"] != place_input].sample(3)
                
                for _, alt in alts.iterrows():
                    alt_crowd = get_crowd_label(alt["Visitors_Count"], df)
                    st.write(f"📍 **{alt['Place_Name']}** ({alt['Place_Type']})")
                    st.write(f"Rating: ⭐ {alt['Google_Rating']} | Typical Crowd: {alt_crowd}")
            else:
                st.subheader("🌟 Must-See Nearby")
                nearby = df[(df["Location_State"] == state) & (df["Place_Name"] != place_input)].head(2)
                for _, n in nearby.iterrows():
                    st.write(f"• **{n['Place_Name']}** — just a short trip away!")


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARISON TAB
# ═══════════════════════════════════════════════════════════════════════════════

def show_comparison():
    st.subheader("⚖️ Place Comparison")
    df = load_data()
    if df.empty:
        st.warning("Dataset not available.")
        return

    places = sorted(df["Place_Name"].dropna().unique().tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        place1 = st.selectbox("Select First Place", places, index=0)
    with col2:
        place2 = st.selectbox("Select Second Place", places, index=min(1, len(places)-1))

    if place1 == place2:
        st.warning("Please select two different places to compare.")
        return

    d1 = df[df["Place_Name"] == place1].iloc[0]
    d2 = df[df["Place_Name"] == place2].iloc[0]

    st.divider()
    comparison_data = {
        "Feature": ["State", "Type", "Rating", "Avg Visitors", "Ticket Price", "Best Season"],
        place1: [d1['Location_State'], d1['Place_Type'], d1['Google_Rating'], fmt_number(d1['Visitors_Count']), f"₹{d1['Ticket_Price']}", d1['Season']],
        place2: [d2['Location_State'], d2['Place_Type'], d2['Google_Rating'], fmt_number(d2['Visitors_Count']), f"₹{d2['Ticket_Price']}", d2['Season']]
    }
    st.table(pd.DataFrame(comparison_data))

    col1, col2 = st.columns(2)
    def get_pros_cons(row):
        pros = []
        cons = []
        if row['Google_Rating'] >= 4.5: pros.append("Excellent rating")
        if row['Ticket_Price'] == 0: pros.append("Free entry")
        elif row['Ticket_Price'] < 100: pros.append("Budget-friendly")
        else: cons.append("Higher entry fee")
        crowd = get_crowd_label(row['Visitors_Count'], df)
        if crowd == "Low": pros.append("Peaceful/Low crowd")
        elif crowd == "High": cons.append("Often overcrowded")
        return pros, cons

    with col1:
        st.write(f"### 📍 {place1}")
        p1, c1 = get_pros_cons(d1)
        st.write("**Advantages:**")
        for p in p1: st.write(f"✅ {p}")
        st.write("**Disadvantages:**")
        for c in c1: st.write(f"❌ {c}")

    with col2:
        st.write(f"### 📍 {place2}")
        p2, c2 = get_pros_cons(d2)
        st.write("**Advantages:**")
        for p in p2: st.write(f"✅ {p}")
        st.write("**Disadvantages:**")
        for c in c2: st.write(f"❌ {c}")

    st.divider()
    st.subheader("💡 AI Suggestion")
    if d1['Google_Rating'] > d2['Google_Rating']:
        st.write(f"If you prioritize **quality and experience**, **{place1}** is better with a higher rating of {d1['Google_Rating']}.")
    else:
        st.write(f"If you prioritize **quality and experience**, **{place2}** is better with a higher rating of {d2['Google_Rating']}.")
    
    if d1['Visitors_Count'] < d2['Visitors_Count']:
        st.write(f"If you prefer **peace and quiet**, choose **{place1}** as it attracts fewer visitors.")
    else:
        st.write(f"If you prefer **peace and quiet**, choose **{place2}** as it attracts fewer visitors.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAVEL AGENT — MAP EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

STATE_COORDS = {
    "Goa": (15.2993, 74.1240), "Rajasthan": (27.0238, 74.2179),
    "Kerala": (10.8505, 76.2711), "Himachal Pradesh": (31.1048, 77.1734),
    "Karnataka": (15.3173, 75.7139), "Uttar Pradesh": (26.8467, 80.9462),
    "Assam": (26.2006, 92.9376), "Odisha": (20.9517, 85.0985),
    "Telangana": (18.1124, 79.0193), "Nagaland": (26.1584, 94.5624),
    "Tamil Nadu": (11.1271, 78.6569), "Maharashtra": (19.7515, 75.7139),
    "West Bengal": (22.9868, 87.8550), "Gujarat": (22.2587, 71.1924),
    "Madhya Pradesh": (22.9734, 78.6569), "Punjab": (31.1471, 75.3412),
}

def show_map_explorer():
    st.subheader("🗺 Map Explorer")
    df = load_data()
    if df.empty:
        st.warning("Dataset not available.")
        return
    states = sorted(df["Location_State"].dropna().unique().tolist())
    selected = st.selectbox("Select State", states)
    sub = df[df["Location_State"] == selected]
    coords = STATE_COORDS.get(selected, (20.5937, 78.9629))
    map_df = sub.drop_duplicates("Place_Name").head(30).copy()
    rng = np.random.default_rng(42)
    map_df["lat"] = coords[0] + rng.uniform(-0.5, 0.5, len(map_df))
    map_df["lon"] = coords[1] + rng.uniform(-0.5, 0.5, len(map_df))
    st.map(map_df[["lat", "lon"]])
    st.divider()
    st.subheader(f"Places in {selected}")
    cols = ["Place_Name", "Place_Type", "Google_Rating", "Visitors_Count", "Season", "Ticket_Price"]
    cols = [c for c in cols if c in sub.columns]
    top_places = sub[cols].drop_duplicates("Place_Name").sort_values("Google_Rating", ascending=False).head(15)
    for _, row in top_places.iterrows():
        with st.expander(f"📍 {row['Place_Name']} ({row.get('Place_Type','')})"):
            c1, c2, c3 = st.columns(3)
            c1.metric("⭐ Rating",  row.get("Google_Rating", "-"))
            c2.metric("👥 Avg Visitors", fmt_number(row.get("Visitors_Count", 0)))
            c3.metric("🎟 Ticket", f"₹{row.get('Ticket_Price', 0)}")
            st.write(f"**Best Season:** {row.get('Season', '-')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAVEL AGENT — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

def show_alerts():
    st.subheader("🚨 Alerts & Recommendations")
    df = load_data()
    if df.empty:
        st.warning("Dataset not available.")
        return
    q66 = df["Visitors_Count"].quantile(0.66)
    high_risk = (df[df["Visitors_Count"] >= q66]
                 .groupby("Location_State")["Visitors_Count"]
                 .mean().nlargest(8).reset_index())
    high_risk.columns = ["State", "Avg Visitors"]
    st.error(f"⚠️ {len(high_risk)} states have HIGH overcrowding risk!")
    for _, row in high_risk.iterrows():
        with st.expander(f"🔴 {row['State']} — Avg {fmt_number(row['Avg Visitors'])} visitors"):
            st.write("**Crowd Management Strategies:**")
            strategies = [
                "Implement timed entry slots and advance booking only.",
                "Promote off-peak visiting hours with discounts.",
                "Develop alternate tourist circuits in the same region.",
                "Deploy digital crowd monitoring at hotspots.",
                "Coordinate with local authorities on traffic management.",
            ]
            for s in random.sample(strategies, 3):
                st.write(f"• {s}")


def show_home():
    st.title("🌟 AI-Driven Tourism Intelligence Platform")
    st.markdown("""
    ### Welcome to the Future of Travel Planning
    This platform leverages **Advanced AI** and **Machine Learning** to provide real-time insights into tourism demand, 
    overcrowding risks, and smart travel recommendations across India's top destinations.
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔮 Intelligence Engine")
        st.write("Our rule-based intelligence system analyzes specific seasonal patterns (Winter, Summer, Monsoon, Festive) to predict peak demand and overcrowding risk for your selected destination.")
        
        st.subheader("🤖 ML Forecaster")
        st.write("A trained Machine Learning model predicts expected visitor counts based on historical data, weather conditions, and travel trends, giving you a data-driven edge.")
        
    with col2:
        st.subheader("📊 BI Dashboard")
        st.write("Dynamic visualizations and correlation heatmaps help travel agents and enthusiasts understand the underlying factors driving tourism revenue and popularity.")
        
        st.subheader("💬 Smart AI Assistant")
        st.write("Our integrated chatbot is ready to answer your questions about destinations, budget planning, and travel tips, providing a seamless planning experience.")

    st.info("💡 **Pro Tip:** Use the sidebar to navigate through the different tools. As a Travel Agent, you have exclusive access to the Advanced Prediction and Dataset management tools.")


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARDS
# ═══════════════════════════════════════════════════════════════════════════════

def travel_agent_dashboard():
    with st.sidebar:
        st.title("🌍 AI-Driven Tourism")
        st.write(f"Welcome, {st.session_state.user.get('full_name','Agent')}!")
        selected = option_menu(
            menu_title=None,
            options=["Home", "Overview", "Dataset", "Dashboard", "Prediction", "Comparison",
                     "Budget Planner", "Map Explorer", "Chatbot",
                     "Alerts", "Profile", "Logout"],
            icons=["house", "bar-chart-line", "database", "graph-up", "cpu", "arrow-left-right",
                   "wallet2", "map", "robot",
                   "bell", "person-circle", "box-arrow-right"],
            default_index=0,
        )

    if selected == "Home":             show_home()
    elif selected == "Overview":        show_overview()
    elif selected == "Dataset":       show_dataset_tab()
    elif selected == "Dashboard":     show_dashboard()
    elif selected == "Prediction":    show_prediction()
    elif selected == "Comparison":    show_comparison()
    elif selected == "Budget Planner":show_budget_planner()
    elif selected == "Map Explorer":  show_map_explorer()
    elif selected == "Chatbot":       show_chatbot()
    elif selected == "Alerts":        show_alerts()
    elif selected == "Profile":       show_profile()
    elif selected == "Logout":
        logout()
        st.rerun()

def user_dashboard():
    with st.sidebar:
        st.title("🌍 AI-Driven Tourism")
        st.write(f"Hi, {st.session_state.user.get('full_name','Traveller')}!")
        selected = option_menu(
            menu_title=None,
            options=["Home", "Chatbot", "Budget Planner", "Place Search", "Comparison", "Profile", "Logout"],
            icons=["house", "robot", "wallet2", "search", "arrow-left-right", "person-circle", "box-arrow-right"],
            default_index=0,
        )

    if selected == "Home":             show_home()
    elif selected == "Chatbot":          show_chatbot()
    elif selected == "Budget Planner": show_budget_planner()
    elif selected == "Place Search":   show_place_search()
    elif selected == "Comparison":     show_comparison()
    elif selected == "Profile":        show_profile()
    elif selected == "Logout":
        logout()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not st.session_state.get("logged_in"):
        show_login_page()
    else:
        role = st.session_state.user.get("role", "User")
        if role == "Travel Agent":
            travel_agent_dashboard()
        else:
            user_dashboard()

if __name__ == "__main__":
    main()
