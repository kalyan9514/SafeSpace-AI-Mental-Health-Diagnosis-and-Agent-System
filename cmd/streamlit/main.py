"""
cmd/streamlit/main.py

Streamlit app entry point for SafeSpace AI.
Provides a conversational interface powered by the LangChain agent,
with a Folium map for nearby clinic results.
"""

import logging
import streamlit as st
import folium
from streamlit_folium import st_folium
from internal.agent.agent import build_agent, find_clinics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def render_clinic_map(clinics: list[dict]):
    """Render a Folium map with markers for each clinic."""
    if not clinics:
        st.warning("No clinics found.")
        return

    first = clinics[0]
    m = folium.Map(location=[first["lat"], first["lng"]], zoom_start=13)

    for clinic in clinics:
        if clinic["lat"] and clinic["lng"]:
            folium.Marker(
                location=[clinic["lat"], clinic["lng"]],
                popup=f"{clinic['name']}\n{clinic['address']}\n{clinic['phone']}",
                tooltip=clinic["name"],
            ).add_to(m)

    st_folium(m, width=700, height=450)


def main():
    st.set_page_config(page_title="SafeSpace AI Agent", page_icon="🧠")
    st.title("🧠 SafeSpace AI — Mental Health Agent")

    # Build agent once and cache in session state
    if "agent" not in st.session_state:
        with st.spinner("Loading agent..."):
            st.session_state.agent = build_agent()
        st.success("Agent ready.")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Chat input
    user_input = st.chat_input("How are you feeling today?")

    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})

        # Check if user is asking about clinics
        if any(word in user_input.lower() for word in ["clinic", "hospital", "center", "nearby"]):
            clinics = find_clinics(user_input)
            render_clinic_map(clinics)
            response = f"Found {len(clinics)} mental health clinics near you."
        else:
            with st.spinner("Thinking..."):
                response = st.session_state.agent.run(user_input)

        st.session_state.history.append({"role": "assistant", "content": response})

    # Render chat history
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if __name__ == "__main__":
    main()