"""
auth.py — Authentication: login, signup, session management
"""

import pandas as pd
import streamlit as st
import os

USERS_FILE = "users.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_users() -> pd.DataFrame:
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(columns=["username", "password", "role", "email", "full_name"])
        df.to_csv(USERS_FILE, index=False)
        return df
    return pd.read_csv(USERS_FILE, dtype=str)


def _save_users(df: pd.DataFrame):
    df.to_csv(USERS_FILE, index=False)


# ── public API ────────────────────────────────────────────────────────────────

def authenticate(username: str, password: str):
    """Return user dict on success, None on failure."""
    df = _load_users()
    row = df[(df["username"] == username) & (df["password"] == password)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def register_user(username: str, password: str, role: str,
                  email: str, full_name: str) -> tuple[bool, str]:
    """Register new user. Returns (success, message)."""
    df = _load_users()
    if username in df["username"].values:
        return False, "Username already exists."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    new_row = pd.DataFrame([{
        "username": username,
        "password": password,
        "role": role,
        "email": email,
        "full_name": full_name,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    _save_users(df)
    return True, "Account created successfully!"


def update_profile(username: str, new_username: str, new_password: str,
                   email: str, full_name: str) -> tuple[bool, str]:
    """Update user profile. Returns (success, message)."""
    df = _load_users()
    idx = df[df["username"] == username].index
    if idx.empty:
        return False, "User not found."
    if new_username != username and new_username in df["username"].values:
        return False, "New username already taken."
    df.loc[idx, "username"]  = new_username
    df.loc[idx, "email"]     = email
    df.loc[idx, "full_name"] = full_name
    if new_password:
        if len(new_password) < 6:
            return False, "Password must be at least 6 characters."
        df.loc[idx, "password"] = new_password
    _save_users(df)
    # update session
    st.session_state.user["username"]  = new_username
    st.session_state.user["email"]     = email
    st.session_state.user["full_name"] = full_name
    return True, "Profile updated!"


def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ── UI pages ──────────────────────────────────────────────────────────────────

def show_login_page():
    st.title("🌍 AI-Driven Tourism")
    st.write("AI-Driven Tourism Demand Forecasting")

    tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])

    # ── Login ──
    with tab_login:
        st.subheader("Welcome back!")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please fill in all fields.")
            else:
                user = authenticate(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.success(f"Welcome, {user['full_name'] or username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.markdown("---")
        st.caption("Demo accounts — Agent: `admin / admin123`  |  User: `user1 / user123`")

    # ── Sign Up ──
    with tab_signup:
        st.subheader("Create an account")
        with st.form("signup_form"):
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Name")
                new_username = st.text_input("Choose Username")
                role = st.selectbox("Role", ["User", "Travel Agent"])
            with col2:
                email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                confirm_pw = st.text_input("Confirm Password", type="password")
            reg_btn = st.form_submit_button("Create Account", use_container_width=True)

        if reg_btn:
            if not all([full_name, new_username, email, new_password, confirm_pw]):
                st.error("Please fill in all fields.")
            elif new_password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(new_username, new_password, role, email, full_name)
                if ok:
                    st.success(msg + " Please login.")
                else:
                    st.error(msg)
