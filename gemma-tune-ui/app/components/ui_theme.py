import streamlit as st

def load_custom_ui():
    with open("assets/css/custom.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
