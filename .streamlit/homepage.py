import streamlit as st

#st.logo("lscft_logo.jpg")

with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.title("ADHD Pathway Simulation App")

st.write("Welcome to the ADHD pathway simulation app! This simulation is " 
         "designed to model the flow of CYP through our new diagnostic "
         "Pathway")

st.write("Head to the 'Run Pathway Simulation' page to get started.")