import streamlit as st

# Initialise session state variable
# I've opted to do this in the app.py file here (though this wasn't demonstrated in the
# session!)
# By doing it in app.py, these defaults will be set regardless of which page the user enters the
# app on - so e.g. if they jump straight to the des page via a bookmark, it will still pull in
# these defaults because behind the scenes the app.py file is still running.
# Alternatively, we could run this same block of code on both the des.py and lsoa_map.py pages
# (but we could get away without including it in the homepage.py file because that page neither
# displays nor changes the value of demand)
#if 'walk_in_demand' not in st.session_state:
#    st.session_state.walk_in_demand = 150
#if 'calls_demand' not in st.session_state:
#   st.session_state.calls_demand = 50

# Notice that here I've put the lsoa_map in between the homepage and des pages as it makes more sense
# for the user to go to the lsoa map (to choose their region for demand) rather than going to the
# des page first
# i.e. the order of pages in the list will affect the order they appear in the sidebar for the user

pg = st.navigation(
    [st.Page("homepage.py", title="Welcome!", icon=":material/add_circle:"),
     st.Page("des.py", title="Run Pathway Simulation", icon=":material/public:")
     ,]
     )

pg.run()