import os
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("ASSISTFLOW_API_URL", "http://127.0.0.1:8000/submit-ticket")
REQUEST_TIMEOUT_SECONDS = 20

st.set_page_config(page_title="AssistFlow AI", layout="wide", page_icon="🤖")

if "tickets" not in st.session_state:
    st.session_state["tickets"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

PRIORITY_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def preprocess_text_frontend(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def submit_ticket(cleaned_text: str) -> dict:
    response = requests.post(
        API_URL,
        json={"ticket_text": cleaned_text},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def user_portal() -> None:
    st.title("💬 AssistFlow Customer Support")
    st.caption(f"Connected endpoint: {API_URL}")
    st.markdown("Describe your issue below. The assistant will triage and respond immediately.")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("My internet is slow..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        cleaned_text = preprocess_text_frontend(prompt)
        if not cleaned_text:
            st.warning("Please enter a valid issue description.")
            return

        with st.spinner("Analyzing issue..."):
            try:
                data = submit_ticket(cleaned_text)
            except requests.exceptions.RequestException as exc:
                st.error(f"Backend request failed: {exc}")
                return

        ai_reply = data.get("user_response", "I'm not sure, please contact support.")
        priority = data.get("final_priority", "Low")
        sentiment = data.get("final_sentiment", "Neutral")
        action = data.get("action", "AUTO_RESOLVE")
        status = "Open" if action == "ESCALATE_TO_AGENT" else "Resolved"

        new_ticket = {
            "id": len(st.session_state["tickets"]) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customer_text": prompt,
            "priority": priority,
            "sentiment": sentiment,
            "action": action,
            "status": status,
            "ai_solution": ai_reply,
            "agent_explanation": data.get("agent_explanation", ""),
        }
        st.session_state["tickets"].append(new_ticket)

        st.session_state["messages"].append({"role": "assistant", "content": ai_reply})
        with st.chat_message("assistant"):
            st.markdown(ai_reply)
            if action == "ESCALATE_TO_AGENT":
                st.error(f"🚨 Escalated to agent ({priority}).")
            else:
                st.success("✅ Auto-resolved.")


def agent_dashboard() -> None:
    st.title("🛡️ Agent Command Center")
    st.markdown("Live queue with triage actions and trend insights.")

    if not st.session_state["tickets"]:
        st.info("No tickets submitted yet.")
        return

    df = pd.DataFrame(st.session_state["tickets"])
    df["priority_rank"] = df["priority"].map(PRIORITY_ORDER).fillna(-1)
    df = df.sort_values(by=["status", "priority_rank", "id"], ascending=[True, False, False])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickets", len(df))
    c2.metric("Open Queue", len(df[df["status"] == "Open"]))
    c3.metric("Negative Sentiment", len(df[df["sentiment"] == "Negative"]))
    c4.metric("Resolved", len(df[df["status"] == "Resolved"]))

    st.divider()
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📨 Ticket Feed")
        selected_priorities = st.multiselect(
            "Priority",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
        selected_status = st.multiselect(
            "Status",
            ["Open", "Resolved"],
            default=["Open", "Resolved"],
        )

        filtered_df = df[df["priority"].isin(selected_priorities) & df["status"].isin(selected_status)]

        if filtered_df.empty:
            st.warning("No tickets match current filters.")
            return

        for _, row in filtered_df.iterrows():
            with st.container(border=True):
                top_a, top_b, top_c = st.columns([1, 4, 2])
                top_a.write(f"**#{int(row['id'])}**")
                top_b.write(f"**{row['priority']}** | {row['sentiment']} | **{row['status']}**")
                top_c.caption(row["timestamp"])

                st.write(f"🗣️ *{row['customer_text']}*")

                with st.expander("View AI response and agent notes"):
                    st.info(row["ai_solution"])
                    if row["agent_explanation"]:
                        st.caption(f"Agent note: {row['agent_explanation']}")

                if row["status"] == "Open":
                    if st.button(f"Mark Resolved #{int(row['id'])}", key=f"resolve_{int(row['id'])}"):
                        ticket_id = int(row["id"])
                        for ticket in st.session_state["tickets"]:
                            if ticket["id"] == ticket_id:
                                ticket["status"] = "Resolved"
                                break
                        st.rerun()

    with col_right:
        st.subheader("📈 Support Trends")
        fig_pie = px.pie(
            df,
            names="priority",
            title="Ticket Priority Mix",
            hole=0.45,
            color="priority",
            color_discrete_map={"High": "orangered", "Medium": "gold", "Low": "seagreen"},
        )
        fig_pie.update_layout(height=300, margin=dict(t=32, b=8, l=8, r=8))
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_bar = px.histogram(
            df,
            x="sentiment",
            title="Sentiment Distribution",
            color="sentiment",
            color_discrete_map={"Negative": "crimson", "Neutral": "gray", "Positive": "seagreen"},
        )
        fig_bar.update_layout(height=300, margin=dict(t=32, b=8, l=8, r=8))
        st.plotly_chart(fig_bar, use_container_width=True)


st.sidebar.title("AssistFlow Navigation")
sidebar_nav = st.sidebar.radio("Go to", ["👤 User Portal", "🛡️ Agent Dashboard"])

if sidebar_nav == "👤 User Portal":
    user_portal()
else:
    agent_dashboard()
