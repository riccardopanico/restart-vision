import streamlit as st
from views.interface import run_app

if __name__ == "__main__":
    st.set_page_config(page_title="YOLOv8 Streamlit Inference", layout="wide")
    run_app()
