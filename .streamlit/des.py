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

    # Referral Inputs
    st.markdown("#### Referrals")
    referral_input = st.slider("Number of Referrals Per Week", 1, 100, 50)
    referral_reject_input = st.slider("Referral Rejection Rate (%)",
                                      0.0, 10.0, 5.0)
    
    # Triage Inputs
    st.divider()
    st.markdown("#### Triage")
    triage_rejection_input = st.slider("Triage Rejection Rate (%)",
                                       0.0, 10.0, 5.0)
    triage_target_input = st.slider("Number of Weeks to Triage", 1, 10, 4)
    triage_resource_input =  st.slider("Number of Triage Slots p/w", 20, 60, 48)
    triage_clin_time_input =  st.slider("Avg Clinical Time per Triage (mins)", 20, 60, 48)
    triage_admin_time_input =  st.slider("Avg Admin Time per Triage (mins)", 20, 60, 48)

    
    # School/Home Assessment Packs
    st.divider()
    st.markdown("#### School/Home Assessment Packs")
    target_pack_input = st.slider("Number of Weeks to Return Information Pack"
                                                                    ,2, 6, 3)
    pack_rejection_input = st.slider("Assessment Pack Rejection Rate (%)"
                                                            , 0.0, 10.0, 3.0)
    # Observations
    st.divider()
    st.markdown("#### QB and Observations")
    target_obs_input = st.slider("Number of Weeks to Return Observations"
                                                                    ,2, 6, 4)
    obs_rejection_input = st.slider("Observations Rejection Rate (%)"
                                                            , 0.0, 10.0, 1.0)
    # MDT Inputs
    st.divider()
    st.markdown("#### MDT")
    mdt_rejection_input = st.slider("MDT Rejection Rate (%)", 0.0, 10.0, 5.0)
    mdt_target_input = st.slider("Number of Weeks to MDT", 0, 5, 1)
    mdt_resource_input =  st.slider("Number of MDT Slots p/w", 20, 60, 25)

    # Assessment Inputs
    st.divider()
    st.markdown("#### Assessment")
    asst_rejection_input = st.slider("Assessment Rejection Rate (%)",
                                     0.0, 10.0, 1.0)
    asst_target_input = st.slider("Number of Weeks to Assess", 0, 5, 4)
    asst_resource_input =  st.slider("Number of Assessment Slots p/w",
                                     40, 80, 62)
    asst_clin_time_input =  st.slider("Avg Clinical Time per Asst (mins)", 20, 60, 48)
    asst_admin_time_input =  st.slider("Avg Admin Time per Asst (mins)", 20, 60, 48)

    st.divider()
    st.markdown("#### Simulation Parameters")
    sim_duration_input =  st.slider("Simulation Duration (weeks)", 1, 260, 52)
    st.write(f"The service is running for {sim_duration_input} weeks")
    number_of_runs_input = st.slider("Number of Simulation Runs", 1, 100, 10)

g.mean_referrals_pw = referral_input
g.base_waiting_list = 2741
g.referral_rejection_rate = referral_reject_input/100
g.triage_rejection_rate = triage_rejection_input/100
g.target_triage_wait = triage_target_input
g.triage_resource = triage_resource_input
g.triage_time_clin = triage_clin_time_input
g.triage_time_admin = triage_admin_time_input
g.target_pack_wait = target_pack_input
g.pack_rejection_rate = pack_rejection_input/100
g.target_obs_wait = target_obs_input
g.obs_rejection_rate = obs_rejection_input/100
g.mdt_rejection_rate = mdt_rejection_input/100
g.target_mdt_wait = mdt_target_input
g.mdt_resource = mdt_resource_input

g.asst_rejection_rate = asst_rejection_input/100
g.target_asst_wait = asst_target_input
g.asst_resource = asst_resource_input
g.asst_time_clin = asst_clin_time_input
g.asst_time_admin = asst_admin_time_input

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

        # df_trial_results = pd.DataFrame(df_trial_results)
        st.subheader("Summary of Simulation Runs")
        #st.write(df_trial_results)
        #df_trial_results.to_csv('adhd_trial_results.csv')

        # turn mins values from running total to weekly total
        df_weekly_stats['Triage Clin Hrs'] = (df_weekly_stats['Triage Clin Mins']-df_weekly_stats['Triage Clin Mins'].shift(1))/60
        df_weekly_stats['Triage Admin Hrs'] = (df_weekly_stats['Triage Admin Mins']-df_weekly_stats['Triage Admin Mins'].shift(1))/60
        df_weekly_stats['Asst Clin Hrs'] = (df_weekly_stats['Asst Clin Mins']-df_weekly_stats['Asst Clin Mins'].shift(1))/60
        df_weekly_stats['Asst Admin Hrs'] = (df_weekly_stats['Asst Admin Mins']-df_weekly_stats['Asst Admin Mins'].shift(1))/60

        # get rid of negative values
        num = df_weekly_stats._get_numeric_data()

        num[num < 0] = 0

        #st.write(df_weekly_stats)

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
        
        df_weekly_clin = df_weekly_stats[['Run','Week Number','Triage Clin Hrs',
                                        'Asst Clin Hrs']]
        
        df_weekly_clin_unpivot = pd.melt(df_weekly_clin, value_vars=['Triage Clin Hrs',
                                        'Asst Clin Hrs'], id_vars=['Run',
                                        'Week Number'])
        
        df_weekly_admin = df_weekly_stats[['Run','Week Number','Triage Admin Hrs',
                                        'Asst Admin Hrs']]
        
        df_weekly_admin_unpivot = pd.melt(df_weekly_admin, value_vars=['Triage Admin Hrs',
                                        'Asst Admin Hrs'], id_vars=['Run',
                                        'Week Number'])


        tab1, tab2 = st.tabs(["Waiting Lists", "Clinical & Admin"])

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
                    fig.update_layout(title_x=0.2,font=dict(size=10))
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
                    fig2.update_layout(title_x=0.2,font=dict(size=10))
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
                                        color='red')))
        
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
                    fig3.update_layout(title_x=0.2,font=dict(size=10))

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

        with tab2:

            col1, col2 = st.columns(2)

            with col1:
            
                for i, list_name in enumerate(df_weekly_clin_unpivot['variable']
                                            .unique()):

                    if list_name == 'Triage Clin Hrs':
                        section_title = 'Triage'
                    # elif list_name == 'MDT WL':
                    #     section_title = 'MDT'
                    elif list_name == 'Asst Clin Hrs':
                        section_title = 'Assessment'

                    st.subheader(section_title)

                    df_weekly_clin_filtered = df_weekly_clin_unpivot[
                                        df_weekly_clin_unpivot["variable"]==list_name]
                    
                    weekly_avg_mins_clin = df_weekly_clin_filtered.groupby(['Week Number',
                                                    'variable'])['value'
                                                    ].mean().reset_index()
                    
                    fig = px.histogram(weekly_avg_mins_clin, 
                                       x="Week Number",
                                       y='value',
                                       color="green"
                                       labels={"value": "Hours",},
                                       title=f'{list_name} by Week')
                   
                    # get rid of 'variable' prefix resulting from df.melt
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split
                                                            ("=")[1]))
                    #fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))

                    # fig.update_layout(
                    #     title=dict(text=f'ADHD {'variable'} Waiting Lists by Week, 
                    #               font=dict(size=20), automargin=True, yref='paper')
                    #     ))
                    fig.update_layout(title_x=0.2,font=dict(size=10))
                    #fig.

                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()

            with col2:
            
                for i, list_name in enumerate(df_weekly_admin_unpivot['variable']
                                            .unique()):

                    st.subheader('')                   
                    
                    df_weekly_admin_filtered = df_weekly_admin_unpivot[
                                        df_weekly_admin_unpivot["variable"]==list_name]
                    
                    weekly_avg_mins_admin = df_weekly_admin_filtered.groupby(['Week Number',
                                                    'variable'])['value'
                                                    ].mean().reset_index()
                    
                    fig = px.histogram(weekly_avg_mins_admin, 
                                       x="Week Number",
                                       y='value',
                                       color="blue"
                                       labels={"value": "Hours",},
                                       title=f'{list_name} by Week')
                   
                    # get rid of 'variable' prefix resulting from df.melt
                    fig.for_each_annotation(lambda a: a.update(text=a.text.split
                                                            ("=")[1]))
                    #fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))

                    # fig.update_layout(
                    #     title=dict(text=f'ADHD {'variable'} Waiting Lists by Week, 
                    #               font=dict(size=20), automargin=True, yref='paper')
                    #     ))
                    fig.update_layout(title_x=0.2,font=dict(size=10))
                    #fig.

                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()
