"""2026 World Cup Score Predictions — Streamlit Frontend

Start: streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import sys

import httpx
import streamlit as st
import pandas as pd

API_BASE = os.environ.get("QUANTBET_API", "http://localhost:8000")
API_TOKEN = os.environ.get("QUANTBET_TOKEN", "")
if not API_TOKEN:
    st.warning("QUANTBET_TOKEN not set. API requests will fail. Set this environment variable in production.")

_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
_CLIENT = httpx.Client(base_url=API_BASE, headers=_HEADERS, timeout=30)

st.set_page_config(
    page_title="2026 World Cup Score Predictions",
    page_icon="⚽",
    layout="wide",
)

# ---- Sidebar ----
st.sidebar.title("⚽ 2026 World Cup Score Predictions")
st.sidebar.markdown("Data Scientist Workbench")
mode = st.sidebar.radio("Mode", ["\U0001f4ac Chat Predict", "\U0001f4ca Momentum", "\U0001f52c Custom Backtest"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"API: `{API_BASE}`")


# ---- Helpers ----
def call_api(method: str, path: str, json: dict | None = None) -> dict | None:
    try:
        if method == "GET":
            r = _CLIENT.get(path)
        else:
            r = _CLIENT.post(path, json=json or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ---- Mode 1: Chat Prediction ----
if mode == "\U0001f4ac Chat Predict":
    st.title("\U0001f4ac Natural Language Prediction")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("data") and msg["data"].get("type") == "prediction":
                d = msg["data"]
                col1, col2, col3 = st.columns(3)
                col1.metric(f"🏠 {d['home_team']}", f"{d['home_win_prob']:.1%}")
                col2.metric("\U0001f91d Draw", f"{d['draw_prob']:.1%}")
                col3.metric(f"✈️ {d['away_team']}", f"{d['away_win_prob']:.1%}")
                if d.get("expected_home_goals"):
                    st.caption(
                        f"Expected goals {d['expected_home_goals']:.1f}-{d['expected_away_goals']:.1f}"
                        f"  |  O2.5 {d.get('over_25', 0):.0%}  BTTS {d.get('btts', 0):.0%}"
                    )

    prompt = st.chat_input('e.g. "Spain vs Cape Verde" or "momentum"')
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = call_api("POST", "/api/v1/chat", {"message": prompt})
        if result:
            reply = result["reply"]
            data = result.get("data")
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "data": data,
            })
            with st.chat_message("assistant"):
                st.markdown(reply)
                if data and data.get("type") == "prediction":
                    d = data
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"🏠 {d['home_team']}", f"{d['home_win_prob']:.1%}")
                    col2.metric("\U0001f91d Draw", f"{d['draw_prob']:.1%}")
                    col3.metric(f"✈️ {d['away_team']}", f"{d['away_win_prob']:.1%}")
                    if d.get("expected_home_goals"):
                        st.caption(
                            f"Expected goals {d['expected_home_goals']:.1f}-{d['expected_away_goals']:.1f}"
                            f"  |  O2.5 {d.get('over_25', 0):.0%}  BTTS {d.get('btts', 0):.0%}"
                        )

# ---- Mode 2: Momentum ----
elif mode == "\U0001f4ca Momentum":
    st.title("\U0001f4ca GAS Real-time Momentum")
    st.caption("How much each team currently deviates from its Elo anchor (positive = stronger than anchor, negative = weaker)")

    result = call_api("GET", "/api/v1/model/momentum")
    if result:
        df = pd.DataFrame(result["teams"])
        df["momentum"] = df["momentum"].round(3)
        df["d_att"] = df["d_att"].round(3)
        df["d_def"] = df["d_def"].round(3)
        df = df.rename(columns={
            "team": "Team", "momentum": "Momentum",
            "d_att": "δ_att", "d_def": "δ_def",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Team")["Momentum"])

# ---- Mode 3: Custom Backtest ----
elif mode == "\U0001f52c Custom Backtest":
    st.title("\U0001f52c Batch Predict")
    st.markdown("Enter team names one match at a time, run batch predictions.")

    if "backtest_rows" not in st.session_state:
        st.session_state.backtest_rows = []

    col1, col2, col3 = st.columns(3)
    with col1:
        home = st.text_input("Home", key="bt_home")
    with col2:
        away = st.text_input("Away", key="bt_away")
    with col3:
        venue = st.selectbox("Venue", ["neutral", "home", "away"], key="bt_venue")

    if st.button("\U00002795 Add") and home and away:
        st.session_state.backtest_rows.append({
            "home": home.strip(), "away": away.strip(), "venue": venue
        })

    if st.button("\U0001f50d Batch Predict") and st.session_state.backtest_rows:
        results = []
        for row in st.session_state.backtest_rows:
            r = call_api("POST", "/api/v1/predict", row)
            if r:
                results.append({
                    "Home": r["home_team"], "Away": r["away_team"],
                    "Home Win": f"{r['home_win_prob']:.1%}",
                    "Draw": f"{r['draw_prob']:.1%}",
                    "Away Win": f"{r['away_win_prob']:.1%}",
                    "λh": r["expected_home_goals"],
                    "λa": r["expected_away_goals"],
                })
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    if st.session_state.backtest_rows:
        st.markdown("**Added matches:**")
        for row in st.session_state.backtest_rows:
            st.text(f"{row['home']} vs {row['away']} ({row['venue']})")
