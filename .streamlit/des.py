import simpy
import random
import numpy as np
import pandas as pd
import streamlit as st
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import kaleido
import io

from des_classes_v5 import g, Trial
#from app_style import global_page_style

########## Streamlit App ##########
st.set_page_config(layout="wide")

st.logo("https://lancsvp.org.uk/wp-content/uploads/2021/08/nhs-logo-300x189.png")

# Import custom css for using a Google font
# with open("style.css") as css:
#    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

#global_page_style('static/css/style.css')

st.title("ADHD Pathway Simulation")

with st.sidebar:

    st.subheader("Model Inputs")

    with st.expander("Referrals"):

        # Referral Inputs
        st.markdown("#### Referrals")
        referral_input = st.slider("Number of Referrals Per Week", 1, 100, 50)
        referral_reject_input = st.number_input("Referral Rejection Rate (%)",
                        min_value=0.0, max_value=20.0, step=0.25, value=4.25)
           
    with st.expander("Triage"):
        
        # Triage Inputs
        st.divider()
        st.markdown("#### Triage")
        triage_rejection_input = st.number_input("Triage Rejection Rate (%)",
                        min_value=0.0, max_value=20.0, step=0.25, value=7.0)
        triage_wl_input = st.number_input("Current Triage Waiting List", min_value=0, max_value=2500, step=1, value=0)
        if triage_wl_input > 0:
            triage_wait_input = st.number_input("Current Average Triage Waiting Time (weeks)", min_value=0, max_value=52, step=1, value=0)
        else:
            triage_wait_input = 0
        triage_target_input = st.slider("Number of Weeks to Triage", 1, 10, 4)
        triage_clin_time_input =  st.slider("Avg Clinical Time per Triage (mins)", 20, 60, 48)
        triage_admin_time_input =  st.slider("Avg Admin Time per Triage (mins)", 20, 60, 48)

    with st.expander("Pack & Observations"):

        # School/Home Assessment Packs
        st.divider()
        st.markdown("#### School/Home Assessment Packs")
        target_pack_input = st.slider("Number of Weeks to Return Information Pack"
                                                                        ,2, 6, 3)
        pack_rejection_input = st.number_input("Pack Rejection Rate (%)",
                        min_value=0.0, max_value=20.0, step=0.25, value=3.0)
        # Observations
        st.divider()
        st.markdown("#### QB and Observations")
        target_obs_input = st.slider("Number of Weeks to Return Observations"
                                                                        ,2, 6, 4)
        obs_rejection_input = st.number_input("Observations Rejection Rate (%)",
                        min_value=0.0, max_value=20.0, step=0.25, value=1.0)
    with st.expander("MDT"):
   
        # MDT Inputs
        st.divider()
        st.markdown("#### MDT")
        mdt_rejection_input = st.number_input("MDT Rejection Rate (%)",
                        min_value=0.0, max_value=20.0, step=0.25, value=5.0)
        mdt_target_input = st.slider("Number of Weeks to MDT", 0, 5, 1)
        mdt_resource_input =  st.slider("Number of MDT Slots p/w", 0, 100, 60)

    with st.expander("Assessment"):
    
        # Assessment Inputs
        st.divider()
        st.markdown("#### Assessment")
        asst_rejection_input = st.number_input("Referral Rejection Rate (%)",
                        min_value=0.0, max_value=20.0, step=0.25, value=3.0)
        asst_target_input = st.slider("Number of Weeks to Assess", 0, 5, 4)
        asst_wl_input = st.number_input("Current Assessment Waiting List", min_value=0, max_value=2500, step=1, value=0)
        if asst_wl_input > 0:
            asst_wait_input = st.number_input("Current Average Assessment Waiting Time (weeks)", min_value=0, max_value=156, step=1, value=0)
        else:
            asst_wait_input = 0
        asst_clin_time_input =  st.slider("Avg Clinical Time per Asst (mins)", 60, 120, 90)
        asst_admin_time_input =  st.slider("Avg Admin Time per Asst (mins)", 60, 120, 90)

    with st.expander("Job Plans"):
   
        # MDT Inputs
        st.divider()
        st.markdown("#### Job Plans")
        b6_prac_avail_input = st.number_input(label="Starting Number of B6 Practitioner WTE",min_value=0.5,max_value=20.0, step=0.5,value = g.number_staff_b6_prac)
        b6_prac_hours_input = st.slider(label="Number of B6 Hours per WTE", min_value=0.0, max_value=25.0, value=g.hours_avail_b6_prac)
        b6_prac_add_input = st.number_input("Additional Number of B6 Practitioners WTE",
                        min_value=0.0, max_value=20.0, step=0.5, value=0.0)
        asst_resource_input =  st.slider("Number of Assessment Slots per B6", 0, 25, 3)
        triage_resource_input =  st.slider("Number of Triage Slots per B6 WTE", 0, 25, 8)
        b4_prac_avail_input = st.number_input(label="Starting Number of B4 Practitioner WTE",min_value=0.5,max_value=20.0, step=0.5,value = g.number_staff_b4_prac)
        b4_prac_hours_input = st.slider(label="Number of B4 Hours per WTE", min_value=0.0, max_value=25.0, value=g.hours_avail_b4_prac)
        b4_prac_add_input = st.number_input("Additional Number of B4 Practitioners WTE",
                        min_value=0.0, max_value=20.0, step=0.5, value=0.0)
        weeks_lost_input = st.number_input("Weeks Lost to Leave/Sickness etc.",
                        min_value=0.0, max_value=20.0, step=0.25, value=10.0)
            
    with st.expander("Simulation Parameters"):
    
        st.divider()
        st.markdown("#### Simulation Parameters")
        sim_duration_input =  st.slider("Simulation Duration (weeks)", min_value=26, max_value=520, value=52, step=26)
        st.write(f"The service is running for {sim_duration_input} weeks")
        number_of_runs_input = st.slider("Number of Simulation Runs", 1, 20, 10)

g.mean_referrals_pw = referral_input
g.base_waiting_list = 2741
g.referral_rejection_rate = referral_reject_input/100
g.triage_rejection_rate = triage_rejection_input/100
g.target_triage_wait = triage_target_input
g.triage_waiting_list = triage_wl_input
g.triage_average_wait = triage_wait_input
g.triage_resource = int(triage_resource_input * (b6_prac_avail_input+b6_prac_add_input))
g.triage_time_clin = triage_clin_time_input
g.triage_time_admin = triage_admin_time_input
g.target_pack_wait = target_pack_input
g.pack_rejection_rate = pack_rejection_input/100
g.target_obs_wait = target_obs_input
g.obs_rejection_rate = obs_rejection_input/100
g.mdt_rejection_rate = mdt_rejection_input/100
g.target_mdt_wait = mdt_target_input
g.mdt_resource = mdt_resource_input
g.staff_weeks_lost = weeks_lost_input

g.asst_rejection_rate = asst_rejection_input/100
g.target_asst_wait = asst_target_input
g.asst_waiting_list = asst_wl_input
g.asst_average_wait = asst_wait_input
g.asst_resource = int(asst_resource_input * (b6_prac_avail_input+b6_prac_add_input))
g.asst_time_clin = asst_clin_time_input
g.asst_time_admin = asst_admin_time_input

g.number_staff_b6_prac = b6_prac_avail_input
g.number_staff_b4_prac = b4_prac_avail_input
g.hours_avail_b6_prac = b6_prac_hours_input
g.hours_avail_b4_prac = b4_prac_hours_input

# calculate total hours for job plans
total_b6_prac_hours = b6_prac_avail_input*b6_prac_hours_input
total_b4_prac_hours = b4_prac_avail_input*b4_prac_hours_input

g.sim_duration = sim_duration_input
g.number_of_runs = number_of_runs_input

###########################################################
# Run a trial using the parameters from the g class and   #
# print the results                                       #
###########################################################

button_run_pressed = st.button("Run simulation")

if button_run_pressed:
    with st.spinner('Simulating the system...'):

# Create an instance of the Trial class
        my_trial = Trial()
        pd.set_option('display.max_rows', 1000)
        # Call the run_trial method of our Trial class object
        
        df_trial_results, df_weekly_stats = my_trial.run_trial()

        st.subheader(f'Summary of all {g.number_of_runs} Simulation Runs over {g.sim_duration}'
                     f' Weeks with {b6_prac_add_input} Additional B6 and {b4_prac_add_input} Additional B4')
        
        # turn mins values from running total to weekly total in hours
        df_weekly_stats['Referral Screen Hrs'] = (df_weekly_stats['Referral Screen Mins']-df_weekly_stats['Referral Screen Mins'].shift(1))/60
        df_weekly_stats['Triage Clin Hrs'] = (df_weekly_stats['Triage Clin Mins']-df_weekly_stats['Triage Clin Mins'].shift(1))/60
        df_weekly_stats['Triage Admin Hrs'] = (df_weekly_stats['Triage Admin Mins']-df_weekly_stats['Triage Admin Mins'].shift(1))/60
        df_weekly_stats['Triage Reject Hrs'] = (df_weekly_stats['Triage Reject Mins']-df_weekly_stats['Triage Reject Mins'].shift(1))/60
        df_weekly_stats['Pack Send Hrs'] = (df_weekly_stats['Pack Send Mins']-df_weekly_stats['Pack Send Mins'].shift(1))/60
        df_weekly_stats['Pack Reject Hrs'] = (df_weekly_stats['Pack Reject Mins']-df_weekly_stats['Pack Reject Mins'].shift(1))/60
        df_weekly_stats['Obs Visit Hrs'] = (df_weekly_stats['Obs Visit Mins']-df_weekly_stats['Obs Visit Mins'].shift(1))/60
        df_weekly_stats['Obs Reject Hrs'] = (df_weekly_stats['Obs Reject Mins']-df_weekly_stats['Obs Reject Mins'].shift(1))/60
        df_weekly_stats['MDT Prep Hrs'] = (df_weekly_stats['MDT Prep Mins']-df_weekly_stats['MDT Prep Mins'].shift(1))/60
        df_weekly_stats['MDT Meet Hrs'] = (df_weekly_stats['MDT Meet Mins']-df_weekly_stats['MDT Meet Mins'].shift(1))/60
        df_weekly_stats['MDT Reject Hrs'] = (df_weekly_stats['MDT Reject Mins']-df_weekly_stats['MDT Reject Mins'].shift(1))/60
        df_weekly_stats['Asst Clin Hrs'] = (df_weekly_stats['Asst Clin Mins']-df_weekly_stats['Asst Clin Mins'].shift(1))/60
        df_weekly_stats['Asst Admin Hrs'] = (df_weekly_stats['Asst Admin Mins']-df_weekly_stats['Asst Admin Mins'].shift(1))/60
        df_weekly_stats['Diag Reject Hrs'] = (df_weekly_stats['Diag Reject Mins']-df_weekly_stats['Diag Reject Mins'].shift(1))/60
        df_weekly_stats['Diag Accept Hrs'] = (df_weekly_stats['Diag Accept Mins']-df_weekly_stats['Diag Accept Mins'].shift(1))/60
        # get rid of negative values
        num = df_weekly_stats._get_numeric_data()

        num[num < 0] = 0

        #st.write(df_weekly_stats)

        ########## Waiting List Tab ##########

        ##### get all data structured correctly #####

        with st.expander("See explanation"):
            st.write('This ADHD Simulation is designed to replicate the flow '
                     'of patients through the new ADHD Diagnostic Pathway. Each '
                     'week referrals (patients) are generated by taking the '
                     'input of Referrals per Week and randomising this to '
                     'better reflect real life variability in referral rates. '
                     'It is randomised using a Poisson Distribution. Each week '
                     'a set amount of resource is allocated based upon the '
                     'inputs for the number of Triage, MDT and Assessment '
                     'slots per week. Each patient that is put through the '
                     'pathway takes a unit of resource for that part of the '
                     'pathway. If no resources for that specific part of the '
                     'pathway is available as they have all been used for that '
                     'week the patient is added to a waiting list and waits '
                     'until a resource becomes available for that part of the '
                     'pathway. Once a resource is available the patient is put '
                     'through that part of the pathway, and how long they '
                     'waited is recorded along with how long it took to '
                     'complete that part of the pathway and how much staff '
                     'resource (in minutes by taking the average clinical and '
                     'admin time input and randomising this, again using a '
                     'Poisson Distribution) was used to complete this. Once '
                     'they start that part of the pathway they are removed '
                     'from the waiting list and the next patient in the queue '
                     'starts that part of the pathway, if a resource is '
                     'available.')
            
            st.write('At any point along the pathway a patient can be rejected '
                     '. The decision to reject is based upon the rejection '
                     'rate input set for that part of the pathway. For '
                     'example, if the rejection rate is set to zero then no '
                     'patients will be rejected, but if it is set to 5% then '
                     '5% of patients will be rejected by being randomly '
                     'selected for rejection. Once a patient is rejected at '
                     'any stage they will no longer progress to any subsequent '
                     'parts of the pathway. Once a patient completes that part '
                     'of the pathway they are taken off that waiting list.')
        
        df_weekly_wl = df_weekly_stats[['Run','Week Number','Triage WL',
                                        'MDT WL','Asst WL']]

        df_weekly_wl_unpivot = pd.melt(df_weekly_wl, value_vars=['Triage WL',
                                                                 'MDT WL',
                                                                 'Asst WL'],
                                                                 id_vars=['Run',
                                                                'Week Number'])
        
        df_weekly_rej = df_weekly_stats[['Run','Week Number','Triage Rejects',
                                         'MDT Rejects','Asst Rejects']]

        df_weekly_rej_unpivot = pd.melt(df_weekly_rej, 
                                        value_vars=['Triage Rejects',
                                                    'MDT Rejects',
                                                    'Asst Rejects'],
                                                    id_vars=['Run',
                                                    'Week Number'])

        df_weekly_wt = df_weekly_stats[['Run','Week Number','Triage Wait',
                                        'MDT Wait','Asst Wait']]

        df_weekly_wt_unpivot = pd.melt(df_weekly_wt, value_vars=['Triage Wait',
                                        'MDT Wait','Asst Wait'], id_vars=['Run',
                                        'Week Number'])
        
        ########## Clinical & Admin Tab ##########
        
        ##### get all data structured correctly #####

        ##### Top - 1 chart #####

        df_weekly_ref_screen = df_weekly_stats[['Run','Week Number',
                                                'Referral Screen Hrs']]
                  
        ##### Upper Middle - 3 columns 1 row #####

        df_weekly_triage_clin = df_weekly_stats[['Run','Week Number',
                                                'Triage Clin Hrs']]
        
        df_weekly_triage_admin = df_weekly_stats[['Run','Week Number',
                                                'Triage Admin Hrs']]
        
        df_weekly_triage_rej = df_weekly_stats[['Run','Week Number',
                                                'Triage Reject Hrs']]
        ##### Middle - 2 columns 2 rows #####

        df_weekly_col4 = df_weekly_stats[['Run','Week Number','Pack Send Hrs',
                                        'Obs Visit Hrs']]
        
        df_weekly_col4_unpivot = pd.melt(df_weekly_col4, value_vars=[
                                        'Pack Send Hrs',
                                        'Obs Visit Hrs'],
                                        id_vars=['Run','Week Number'])
        
        df_weekly_col5 = df_weekly_stats[['Run','Week Number','Pack Reject Hrs',
                                        'Obs Reject Hrs']]
        
        df_weekly_col5_unpivot = pd.melt(df_weekly_col5, value_vars=[
                                        'Pack Reject Hrs',
                                        'Obs Reject Hrs'],
                                        id_vars=['Run','Week Number'])
        
        ##### Lower Middle - 3 columns 1 row #####

        df_weekly_mdt_prep = df_weekly_stats[['Run','Week Number',
                                                'MDT Prep Hrs']]
        
        df_weekly_mdt_meet = df_weekly_stats[['Run','Week Number',
                                                'MDT Meet Hrs']]
        
        df_weekly_mdt_rej = df_weekly_stats[['Run','Week Number',
                                                'MDT Reject Hrs']]

        ##### Bottom - 2 columns 2 rows #####
        
        df_weekly_col9 = df_weekly_stats[['Run','Week Number',
                                        'Asst Clin Hrs','Diag Reject Hrs']]
        
        df_weekly_col9_unpivot = pd.melt(df_weekly_col9, value_vars=[
                                        'Asst Clin Hrs','Diag Reject Hrs'],
                                        id_vars=['Run','Week Number'])
        
        df_weekly_col10 = df_weekly_stats[['Run','Week Number',
                                        'Asst Admin Hrs','Diag Accept Hrs']]
        
        df_weekly_col10_unpivot = pd.melt(df_weekly_col10, value_vars=[
                                        'Asst Admin Hrs','Diag Accept Hrs'],
                                        id_vars=['Run','Week Number'])
        
        ########## Job Plans Tab ##########

        ##### Band 6 Practitioner

        df_weekly_b6 = df_weekly_stats[['Run','Week Number',
                                        'Referral Screen Hrs','Triage Clin Hrs',
                                        'Triage Admin Hrs','Triage Reject Hrs',
                                        'Pack Reject Hrs','Obs Reject Hrs',
                                        'MDT Reject Hrs',
                                        'Asst Clin Hrs','Asst Admin Hrs',
                                        'Diag Accept Hrs','Diag Reject Hrs']]

        df_weekly_b6_avg = df_weekly_b6.groupby(['Week Number'], as_index=False).mean()
        
        df_weekly_b6_unpivot = pd.melt(df_weekly_b6_avg, value_vars=[
                                        'Referral Screen Hrs','Triage Clin Hrs',
                                        'Triage Admin Hrs','Triage Reject Hrs',
                                        'Pack Reject Hrs','Obs Reject Hrs',
                                        'MDT Reject Hrs',
                                        'Asst Clin Hrs','Asst Admin Hrs',
                                        'Diag Accept Hrs','Diag Reject Hrs'],
                                        id_vars=['Week Number'])
        
        ##### Band 4 Practitioner

        df_weekly_b4 = df_weekly_stats[['Run','Week Number',
                                        'Obs Visit Hrs','Obs Reject Hrs',
                                        'MDT Prep Hrs','MDT Meet Hrs']]

        df_weekly_b4_avg = df_weekly_b4.groupby(['Week Number'], as_index=False).mean()

        df_weekly_b4_unpivot = pd.melt(df_weekly_b4_avg, value_vars=[
                                        'Obs Visit Hrs','Obs Reject Hrs',
                                        'MDT Prep Hrs','MDT Meet Hrs'],
                                        id_vars=['Week Number'])
        
                       
        tab1, tab2, tab3 = st.tabs(["Waiting Lists", "Clinical & Admin","Job Plans"])
        
        ########## Waiting Lists Tab ##########
        
        with tab1:    

            col1, col2, col3 = st.columns(3)

            with col1:
            
                for i, list_name in enumerate(df_weekly_wl_unpivot['variable']
                                            .unique()):

                    if list_name == 'Triage WL':
                        section_title = 'Triage'
                    elif list_name == 'MDT WL':
                        section_title = 'MDT'
                    elif list_name == 'Asst WL':
                        section_title = 'Assessment'

                    st.subheader(section_title)

                    df_weekly_wl_filtered = df_weekly_wl_unpivot[
                                        df_weekly_wl_unpivot["variable"]==list_name]
                    
                    fig = px.line(
                                df_weekly_wl_filtered,
                                x="Week Number",
                                color="Run",
                                #line_dash="Run",
                                y="value",
                                labels={
                                        "value": "Waiters",
                                        #"sepal_width": "Sepal Width (cm)",
                                        #"species": "Species of Iris"
                                        },
                                #facet_row="variable", # show each facet as a row
                                #facet_col="variable", # show each facet as a column
                                height=500,
                                width=350,
                                title=f'{list_name} by Week'
                                )
                    
                    fig.update_traces(line=dict(dash='dot'))
                    
                    # get the average waiting list across all the runs
                    weekly_avg_wl = df_weekly_wl_filtered.groupby(['Week Number',
                                                    'variable'])['value'].mean(
                                                    ).reset_index()
                    
                    fig.add_trace(
                                go.Scatter(x=weekly_avg_wl["Week Number"],
                                        y=weekly_avg_wl["value"], name='Average',
                                        line=dict(width=3,color='blue')))
        
                    
                    # get rid of 'variable' prefix resulting from df.melt
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split
                                                            ("=")[1]))
                    #fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))

                    # fig.update_layout(
                    #     title=dict(text=f'ADHD {'variable'} Waiting Lists by Week, 
                    #               font=dict(size=20), automargin=True, yref='paper')
                    #     ))
                    fig.update_layout(title_x=0.3,font=dict(size=10))
                    #fig.

                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

            with col2:
            
                for i, list_name in enumerate(df_weekly_rej_unpivot['variable']
                                            .unique()):
                
                    df_weekly_rej_filtered = df_weekly_rej_unpivot[
                                    df_weekly_rej_unpivot["variable"]==list_name]
                    
                    st.subheader('')

                    fig2 = px.line(
                                df_weekly_rej_filtered,
                                x="Week Number",
                                color="Run",
                                #line_dash="Run",
                                y="value",
                                labels={
                                        "value": "Rejections",
                                        #"sepal_width": "Sepal Width (cm)",
                                        #"species": "Species of Iris"
                                        },
                                #facet_row="variable", # show each facet as a row
                                #facet_col="variable", # show each facet as a column
                                height=500,
                                width=350,
                                title=f'{list_name} by Week'
                                )
                    
                    fig2.update_traces(line=dict(dash='dot'))
                    
                    # get the average waiting list across all the runs
                    weekly_avg_rej = df_weekly_rej_filtered.groupby(['Week Number',
                                                    'variable'])['value'].mean(
                                                    ).reset_index()
                    
                    fig2.add_trace(
                                go.Scatter(x=weekly_avg_rej["Week Number"],y=
                                        weekly_avg_rej["value"], name='Average',
                                        line=dict(width=3,color='blue')))
        
                    
                    # get rid of 'variable' prefix resulting from df.melt
                    fig2.for_each_annotation(lambda a: a.update(text=a.text.split
                                                                        ("=")[1]))
                    #fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))

                    # fig.update_layout(
                    #     title=dict(text=f'ADHD {'variable'} Waiting Lists by Week, 
                    #       font=dict(size=20), automargin=True, yref='paper')
                    #     ))
                    fig2.update_layout(title_x=0.3,font=dict(size=10))
                    #fig.

                    st.plotly_chart(fig2, use_container_width=True)

                    st.divider()

        with col3:
            
                for i, list_name in enumerate(df_weekly_wt_unpivot['variable']
                                            .unique()):
                
                    df_weekly_wt_filtered = df_weekly_wt_unpivot[
                                        df_weekly_wt_unpivot["variable"]==list_name]

                    st.subheader('')
                    
                    if list_name == 'Triage Wait':
                        y_var_targ = triage_target_input
                    elif list_name == 'MDT Wait':
                        y_var_targ = mdt_target_input
                    elif list_name == 'Asst Wait':
                        y_var_targ = asst_target_input
                
                    fig3 = px.line(
                                df_weekly_wt_filtered,
                                x="Week Number",
                                color="Run",
                                #line_dash="Run",
                                y="value",
                                labels={
                                        "value": "Avg Wait(weeks)",
                                        #"sepal_width": "Sepal Width (cm)",
                                        #"species": "Species of Iris"
                                        },
                                #facet_row="variable", # show each facet as a row
                                #facet_col="variable", # show each facet as a column
                                height=500,
                                width=350,
                                title=f'{list_name} by Week'
                                )
                    
                    fig3.update_traces(line=dict(dash='dot'))
                    
                    weekly_avg_wt = df_weekly_wt_filtered.groupby(['Week Number',
                                                    'variable'])['value'
                                                    ].mean().reset_index()

                                
                    fig3.add_trace(
                                go.Scatter(x=weekly_avg_wt["Week Number"],
                                        y=weekly_avg_wt["value"],
                                        name='Average',line=dict(width=3,
                                        color='blue')))
        
                    fig3.add_trace(
                                go.Scatter(x=weekly_avg_wt["Week Number"],
                                        y=np.repeat(y_var_targ,g.sim_duration),
                                        name='Target',line=dict(width=3,
                                        color='green')))
                    
                    # get rid of 'variable' prefix resulting from df.melt
                    fig3.for_each_annotation(lambda a: a.update(text=a.text.split(
                                                                        "=")[1]))
                    #fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))

                    # fig.update_layout(
                    #     title=dict(text=f'ADHD {'variable'} Waiting Lists by Week,
                    #       font=dict(size=20), automargin=True, yref='paper')
                    #     ))
                    fig3.update_layout(title_x=0.3,font=dict(size=10))

                    ##fig3.add_hline(y=y_var_targ, annotation_text="mean")
                    
                    st.plotly_chart(fig3, use_container_width=True)

                    st.divider()


            # fig2 = px.line(
            #                 df_weekly_wl_unpivot,
            #                 x="Week Number",
            #                 color="Run",
            #                 y="value",
            #                 labels={
            #                         "value": "Waiters",
            #                         #"sepal_width": "Sepal Width (cm)",
            #                         #"species": "Species of Iris"
            #                         },
            #                 #facet_row="variable", # show each facet as a row
            #                 #facet_col="variable", # show each facet as a column
            #                 height=800,
                            
            #                 )
            # # get rid of 'variable' prefix resulting from df.melt
            # fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
            # #fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))

            # # fig.update_layout(
            # #     title=dict(text=f'ADHD {'variable'} Waiting Lists by Week, 
            #                   font=dict(size=20), automargin=True, yref='paper')
            # #     ))
            # fig2.update_layout(title_x=0.4)
            # #fig.

            # st.plotly_chart(fig2, use_container_width=True)
            # # with col1:
            # #     st.write(df_trial_results)

            #     @st.fragment
            #     def download_1():
            #         st.download_button(
            #             "Click here to download the data in a csv format",
            #             df_trial_results.to_csv().encode('utf-8'),
            #             f"trial_summary_{g.number_of_clinicians}_clinicians_{
            #             g.mean_referrals}_referrals.csv","text/csv")
            #     download_1()

            # fig = px.line(weekly_wl_position,x="Week Number" ,y="Waiting List",
            #               color="Run Number",
            #               title='ADHD Diagnosis Waiting List by Week')

            # fig.update_traces(line=dict(color="Blue", width=0.5))

            # fig.update_layout(title_x=0.4)

            # fig.add_trace(
            #     go.Scatter(x=weekly_avg_wl["Week Number"],y=weekly_avg_wl[
            #                               "Waiting List"], name='Average'))

            # fig.update_layout(xaxis_title='Week Number',
            #                 yaxis_title='Patients Waiting')

            # with col2:
            #     st.plotly_chart(fig, use_container_width=True)

            #     @st.fragment
            #     def download_2():
            #         # Create an in-memory buffer
            #         buffer = io.BytesIO()
            #         fig.write_image(file=buffer,format='pdf')
            #         st.download_button(label='Click here to Download Chart as PDF'
            #         ,data=buffer, file_name='waiting_list',
            #         mime='application/octet-stream')
            #     download_2()


        ########## Clinical & Admin Tab ##########

        with tab2:

            st.subheader('Referrals')

            ##### Referral Screening #####

            df_ref_screen_avg = df_weekly_ref_screen.groupby(['Week Number'])['Referral Screen Hrs'].mean().reset_index()
            
            fig = px.histogram(df_ref_screen_avg, 
                                x='Week Number',
                                y='Referral Screen Hrs',
                                nbins=sim_duration_input,
                                labels={'Referral Screen Hrs': 'Hours'},
                                color_discrete_sequence=['green'],
                                title=f'Referral Screening Hours by Week')
            
            fig.update_layout(title_x=0.4,font=dict(size=10),bargap=0.2)
            
            fig.update_traces(marker_line_color='black', marker_line_width=1)
            #fig.

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            ##### Triage #####

            col1, col2, col3 = st.columns(3)

            with col1:

                st.subheader('Triage')

                df_triage_clin_avg = df_weekly_triage_clin.groupby(['Week Number'])['Triage Clin Hrs'].mean().reset_index()
                
                fig = px.histogram(df_triage_clin_avg, 
                                    x='Week Number',
                                    y='Triage Clin Hrs',
                                    nbins=sim_duration_input,
                                    labels={'Triage Clin Hrs': 'Hours'},
                                    color_discrete_sequence=['green'],
                                    title=f'Triage Clinical Hours by Week')
                
                fig.update_layout(title_x=0.3,font=dict(size=10),bargap=0.2)
                
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                #fig.

                st.plotly_chart(fig, use_container_width=True)

                st.divider()

            with col2:

                st.subheader('')

                df_triage_admin_avg = df_weekly_triage_admin.groupby(['Week Number'])['Triage Admin Hrs'].mean().reset_index()
                
                fig = px.histogram(df_triage_admin_avg, 
                                    x='Week Number',
                                    y='Triage Admin Hrs',
                                    nbins=sim_duration_input,
                                    labels={'Triage Admin Hrs': 'Hours'},
                                    color_discrete_sequence=['blue'],
                                    title=f'Triage Admin Hours by Week')
                
                fig.update_layout(title_x=0.3,font=dict(size=10),bargap=0.2)
                
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                #fig.

                st.plotly_chart(fig, use_container_width=True)

                st.divider()

            with col3:

                st.subheader('')

                df_triage_rej_avg = df_weekly_triage_rej.groupby(['Week Number'])['Triage Reject Hrs'].mean().reset_index()
                
                fig = px.histogram(df_triage_rej_avg, 
                                    x='Week Number',
                                    y='Triage Reject Hrs',
                                    nbins=sim_duration_input,
                                    labels={'Triage Reject Hrs': 'Hours'},
                                    color_discrete_sequence=['red'],
                                    title=f'Triage Rejection Hours by Week')
                
                fig.update_layout(title_x=0.3,font=dict(size=10),bargap=0.2)
                
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                #fig.

                st.plotly_chart(fig, use_container_width=True)

                st.divider()

            ##### Pack & Obs #####

            col4, col5 = st.columns(2)
            
            with col4:
            
                for i, list_name in enumerate(df_weekly_col4_unpivot['variable']
                                            .unique()):

                    if list_name == 'Pack Send Hrs':
                        section_title = 'Information Packs'
                    elif list_name == 'Obs Visit Hrs':
                        section_title = 'Observations'

                    if list_name == 'Pack Send Hrs':
                        chart_colour = 'blue'
                    elif list_name == 'Obs Visit Hrs':
                        chart_colour = 'green'
                    
                    st.subheader(section_title)

                    df_weekly_col4_filtered = df_weekly_col4_unpivot[
                                        df_weekly_col4_unpivot["variable"]==list_name]
                    
                    weekly_avg_hrs_col4 = df_weekly_col4_filtered.groupby(['variable',
                                                    'Week Number'])['value'
                                                    ].mean().reset_index()
                    
                    fig = px.histogram(weekly_avg_hrs_col4, 
                                       x="Week Number",
                                       y='value',
                                       nbins=sim_duration_input,
                                       labels={"value": "Hours"},
                                       color_discrete_sequence=[chart_colour],
                                       title=f'{list_name} by Week')
                   
                    # get rid of 'variable' prefix resulting from df.melt
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split
                                                            ("=")[1]))
                    fig.update_layout(title_x=0.4,font=dict(size=10),bargap=0.2)
                    fig.update_traces(marker_line_color='black', marker_line_width=1)
                    #fig.

                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

            with col5:
            
                for i, list_name in enumerate(df_weekly_col5_unpivot['variable']
                                            .unique()):

                    st.subheader('')
                    
                    df_weekly_col5_filtered = df_weekly_col5_unpivot[
                                        df_weekly_col5_unpivot["variable"]==list_name]
                    
                    weekly_avg_hrs_col5 = df_weekly_col5_filtered.groupby(['variable',
                                                    'Week Number'])['value'
                                                    ].mean().reset_index()
                    
                    fig = px.histogram(weekly_avg_hrs_col5, 
                                       x="Week Number",
                                       y='value',
                                       nbins=sim_duration_input,
                                       labels={"value": "Hours"},
                                       color_discrete_sequence=["red"],
                                       title=f'{list_name} by Week')
                   
                    # get rid of 'variable' prefix resulting from df.melt
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split
                                                            ("=")[1]))
                    fig.update_layout(title_x=0.4,font=dict(size=10),bargap=0.2)
                    fig.update_traces(marker_line_color='black', marker_line_width=1)
                    #fig.

                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

            ##### MDT #####
            
            col6, col7, col8 = st.columns(3)

            with col6:

                st.subheader('MDT')

                df_mdt_prep_avg = df_weekly_mdt_prep.groupby(['Week Number'])['MDT Prep Hrs'].mean().reset_index()
                
                fig = px.histogram(df_mdt_prep_avg, 
                                    x='Week Number',
                                    y='MDT Prep Hrs',
                                    nbins=sim_duration_input,
                                    labels={'MDT Prep Hrs': 'Hours'},
                                    color_discrete_sequence=['blue'],
                                    title=f'MDT Prep Hours by Week')
                
                fig.update_layout(title_x=0.3,font=dict(size=10),bargap=0.2)
                
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                #fig.

                st.plotly_chart(fig, use_container_width=True)

                st.divider()

            with col7:

                st.subheader('')

                df_mdt_meet_avg = df_weekly_mdt_meet.groupby(['Week Number'])['MDT Meet Hrs'].mean().reset_index()
                
                fig = px.histogram(df_mdt_meet_avg, 
                                    x='Week Number',
                                    y='MDT Meet Hrs',
                                    nbins=sim_duration_input,
                                    labels={'MDT Meet Hrs': 'Hours'},
                                    color_discrete_sequence=['goldenrod'],
                                    title=f'MDT Meeting Hours by Week')
                
                fig.update_layout(title_x=0.3,font=dict(size=10),bargap=0.2)
                
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                #fig.

                st.plotly_chart(fig, use_container_width=True)

                st.divider()

            with col8:

                st.subheader('')

                df_mdt_rej_avg = df_weekly_mdt_rej.groupby(['Week Number'])['MDT Reject Hrs'].mean().reset_index()
                
                fig = px.histogram(df_mdt_rej_avg, 
                                    x='Week Number',
                                    y='MDT Reject Hrs',
                                    nbins=sim_duration_input,
                                    labels={'MDT Reject Hrs': 'Hours'},
                                    color_discrete_sequence=['red'],
                                    title=f'MDT Rejection Hours by Week')
                
                fig.update_layout(title_x=0.3,font=dict(size=10),bargap=0.2)
                
                fig.update_traces(marker_line_color='black', marker_line_width=1)
                #fig.

                st.plotly_chart(fig, use_container_width=True)

                st.divider()

            ##### Asst & Diagnosis #####

            col9, col10 = st.columns(2)
            
            with col9:
            
                for i, list_name in enumerate(df_weekly_col9_unpivot['variable']
                                            .unique()):

                    if list_name == 'Asst Clin Hrs':
                        section_title = 'Assessment'
                    elif list_name == 'Diag Reject Hrs':
                        section_title = 'Diagnosis'

                    if list_name == 'Asst Clin Hrs':
                        chart_colour = 'green'
                    elif list_name == 'Diag Reject Hrs':
                        chart_colour = 'red'

                    st.subheader(section_title)

                    df_weekly_col9_filtered = df_weekly_col9_unpivot[
                                        df_weekly_col9_unpivot["variable"]==list_name]
                    
                    weekly_avg_hrs_col9 = df_weekly_col9_filtered.groupby(['variable',
                                                    'Week Number'])['value'
                                                    ].mean().reset_index()
                    
                    fig = px.histogram(weekly_avg_hrs_col9, 
                                       x="Week Number",
                                       y='value',
                                       nbins=sim_duration_input,
                                       labels={"value": "Hours"},
                                       color_discrete_sequence=[chart_colour],
                                       title=f'{list_name} by Week')
                   
                    # get rid of 'variable' prefix resulting from df.melt
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split
                                                            ("=")[1]))
                    fig.update_layout(title_x=0.4,font=dict(size=10),bargap=0.2)
                    fig.update_traces(marker_line_color='black', marker_line_width=1)
                    #fig.

                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

            with col10:
            
                for i, list_name in enumerate(df_weekly_col10_unpivot['variable']
                                            .unique()):

                    if list_name == 'Asst Admin Hrs':
                        chart_colour = 'blue'
                    elif list_name == 'Diag Accept Hrs':
                        chart_colour = 'green'
                    
                    st.subheader('')
                    
                    df_weekly_col10_filtered = df_weekly_col10_unpivot[
                                        df_weekly_col10_unpivot["variable"]==list_name]
                    
                    weekly_avg_hrs_col10 = df_weekly_col10_filtered.groupby(['variable',
                                                    'Week Number'])['value'
                                                    ].mean().reset_index()
                    
                    fig = px.histogram(weekly_avg_hrs_col10, 
                                       x="Week Number",
                                       y='value',
                                       nbins=sim_duration_input,
                                       labels={"value": "Hours"},
                                       color_discrete_sequence=[chart_colour],
                                       title=f'{list_name} by Week')
                   
                    # get rid of 'variable' prefix resulting from df.melt
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split
                                                            ("=")[1]))
                    fig.update_layout(title_x=0.4,font=dict(size=10),bargap=0.2)
                    fig.update_traces(marker_line_color='black', marker_line_width=1)
                    #fig.

                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

        ########## Job Plan Tab ##########

        with tab3:

            st.subheader('Job Plans')

            ##### Band 6 Practitioner #####

            fig = px.histogram(df_weekly_b6_unpivot, 
                                x='Week Number',
                                y='value',
                                nbins=sim_duration_input,
                                labels={'value': 'Hours'
                                        ,'variable':'Time Alloc'},
                                color='variable',
                                color_discrete_sequence=px.colors.qualitative.Dark24,
                                title=f'Band 6 Practitioner Hours by Week')
            
            fig.update_layout(title_x=0.4,font=dict(size=10),bargap=0.2,legend_traceorder="reversed")
            
            fig.update_traces(marker_line_color='black', marker_line_width=1)

            # add line for available B4 hours
            fig.add_trace(
                                go.Scatter(x=weekly_avg_wt["Week Number"],
                                        y=np.repeat(total_b6_prac_hours,g.sim_duration),
                                        name='Avail Hrs',line=dict(width=3,
                                        color='green')))

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            ##### Band 4 Practitioner #####
            
            fig = px.histogram(df_weekly_b4_unpivot, 
                                x='Week Number',
                                y='value',
                                nbins=sim_duration_input,
                                labels={'value': 'Hours'
                                        ,'variable':'Time Alloc'},
                                color='variable',
                                color_discrete_sequence=px.colors.qualitative.Light24,
                                title=f'Band 4 Practitioner Hours by Week')
            
            fig.update_layout(title_x=0.4,font=dict(size=10),bargap=0.2,legend_traceorder="reversed")
            
            fig.update_traces(marker_line_color='black', marker_line_width=1)

            # add line for available B4 hours
            fig.add_trace(
                                go.Scatter(x=weekly_avg_wt["Week Number"],
                                        y=np.repeat(total_b4_prac_hours,g.sim_duration),
                                        name='Avail Hrs',line=dict(width=3,
                                        color='green')))
            
            st.plotly_chart(fig, use_container_width=True)

            st.divider()