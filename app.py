# -*- coding: utf-8 -*-

import streamlit as st
import joblib
import os

from lifecycle_page import render_lifecycle_app
from direct_page import render_direct_prediction_app


st.set_page_config(
    page_title="Seismic Failure Mode Probability Assessment",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== Green style for page-switching buttons ==================
st.markdown("""
<style>
/* Make Streamlit secondary buttons green */
button[data-testid="baseButton-secondary"] {
    background-color: #008000 !important;
    color: white !important;
    border: 1px solid #006400 !important;
    font-weight: bold !important;
    font-family: "Times New Roman", serif !important;
}

/* Hover effect */
button[data-testid="baseButton-secondary"]:hover {
    background-color: #006400 !important;
    color: white !important;
    border: 1px solid #004d00 !important;
}

/* Active/click effect */
button[data-testid="baseButton-secondary"]:active {
    background-color: #004d00 !important;
    color: white !important;
    border: 1px solid #003300 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_numpy_assets():
    assets_path = "model_assets_numpy.pkl"
    if not os.path.exists(assets_path):
        return None
    return joblib.load(assets_path)


assets = load_numpy_assets()


if "page_mode" not in st.session_state:
    st.session_state.page_mode = "lifecycle"


if st.session_state.page_mode == "lifecycle":
    render_lifecycle_app(assets=assets)
else:
    render_direct_prediction_app(assets=assets, show_back_button=True)
