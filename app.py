import streamlit as st
from lifecycle_page import render_lifecycle_app
from direct_page import render_direct_prediction_app

st.set_page_config(
    page_title="Seismic Failure Mode Probability Assessment",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page_mode" not in st.session_state:
    st.session_state.page_mode = "lifecycle"

if st.session_state.page_mode == "lifecycle":
    render_lifecycle_app()
else:
    render_direct_prediction_app()