import streamlit as st
   
def global_page_style(file_path):  
    with open(file_path) as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)