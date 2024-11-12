import simpy
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# This model aims to simulate the flow of CYP through the ADHD clinical pathway
# Assumptions - CYP stay on caseload until they are 18
# only accepted referralS flow through the pathway
# Same clinician will take CYP through each of the steps
# Overall caseload = clinician_caseload * number_adhd_clinicians
# repeat titration taken out due to complexities of modelling for this
# assume all patients get a triage and it doesn't stop patients flowing through
# the rest of the pathway
# assessment doesn't start until there is space on a clinicians caseload
 
########## Streamlit App ##########
st.set_page_config(layout="wide")

st.title("ADHD Pathway Simulation")

with st.sidebar:
    st.subheader("Model Inputs")

    st.markdown("#### Referrals")
    referral_input = st.slider("Number of Referrals Per Week", 1, 20, 8)
    
    st.divider()
    st.markdown("#### Activity Durations")
    mean_asst_time_input = st.slider("Number of Weeks for Assessment",
                                    1, 20, 13)
    mean_school_time_input = st.slider("Number of Weeks for School Reports and\
                                    Obs", 1,20,12)
    mean_diag_time_input = st.slider("Number of Weeks for Diagnosis", 1, 20, 14)
    mean_titr_time_input = st.slider("Number of Weeks for Titration", 1, 20, 15)

    st.divider()
    st.markdown("#### Resources")
    number_of_clinicians_input = st.slider("Number of Clinicians", 1, 10, 5)
    max_caseload_input = st.slider("Maximum Caseload", 1, 40, 20)

    # st.divider()
    # st.markdown("#### Branch Probabilities")
    # prob_book_test_input = st.number_input("Probability of booking a test", 0.0, 1.0, 0.25)

    st.divider()
    st.markdown("#### Simulation Parameters")
    sim_duration_input =  st.slider("Simulation Duration (weeks)", 52, 260, 52)
    st.write(f"The service is running for {sim_duration_input} weeks")
    number_of_runs_input = st.slider("Number of Simulation Runs", 1, 100, 10)

########## Classes ##########

# Class to store global parameter values.  We don't create an instance of this
# class - we just refer to the class blueprint itself to access the numbers
# inside.
class g:
    mean_referrals = referral_input # average number of referrals per week = 8
    base_waiting_list = 2741 # current number of patients on waiting list
    #mean_triage_time = 14 # average wait for triage
    mean_asst_time = mean_asst_time_input # average wait for assessment = 13
    mean_school_time = mean_school_time_input # average time taken for observations and info gather = 12
    mean_diag_time = mean_diag_time_input # average time taken to diagnose = 14
    mean_titrate_time = mean_titr_time_input # average time taken to titrate meds = 15
    #prob_req_repeat_titr = 0.4 # probability that they will need re-titrating
    number_of_clinicians = number_of_clinicians_input # = 10
    max_caseload =  max_caseload_input # maximum number of CYP a clinician can carry = 20
    caseload_slots = number_of_clinicians * max_caseload 
    #prob_req_group_t = 0.4 # probability that they will require group therapy
    #mean_oto_sessions = 10 # average number of 1:1 sessions required
    #mean_group_sessions = 10 # average number of group sessions required
    sim_duration = sim_duration_input # number of weeks to run the simulation for = 52
    number_of_runs = number_of_runs_input # = 10
    weekly_wl_tracker = [] # container to hold weekly waiting list position
    

# Class representing patients coming in to the pathway
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        #self.wait_time_triage = 0 # waiting time for Triage (weeks)
        self.q_time_assessment = 0 # waiting time for Assessment (weeks)
        self.school_time = 0 # time taken doing obs
        self.place_on_waiting_list = 0 # position they are on waiting list
        #self.ig_time = 0 # time taken infornation gathering
        self.diag_time = 0 # time taken to diagnose
        self.titrate_time = 0 # time taken to titrate
        #self.oto_sessions = 0 # number of 1:1 sessions received
        #self.group_sessions = 0 # number of group sessions received
        
# Class representing our model of the ADHD clinical pathway
class Model:
    # Constructor to set up the model for a run. We pass in a run number when
    # we create a new model
    def __init__(self, run_number):
        # Create a SimPy environment in which everything will live
        self.env = simpy.Environment()

        # Create a patient counter (which we'll use as a patient ID)
        self.patient_counter = 0

        self.week_number = 0

        self.place_on_waiting_list = 0
        
        # Create our resources
        self.caseload = simpy.Resource(
            self.env, capacity=g.caseload_slots)
        
        # Store the passed in run number
        self.run_number = run_number

        # Create a new DataFrame that will store some results against
        # the patient ID (which we'll use as the index).
        self.results_df = pd.DataFrame()
        self.results_df["Patient ID"] = [1]
        self.results_df["Week Number"] = [0]
        self.results_df["Waiting List"] = [0]
        self.results_df["Q Time Enter Pathway"] = [0.0]
        self.results_df["Wait Time Asst"] = [0.0]
        self.results_df["Wait Time School"] = [0.0]
        self.results_df["Wait Time Diag"] = [0.0]
        self.results_df["Wait Time Titr"] = [0.0]
        self.results_df.set_index("Patient ID", inplace=True)

        # Create an attribute to store the mean queuing times across this run of
        # the model
        self.mean_q_enter_pathway = 0
        #self.mean_q_time_nurse = 0
        #self.mean_q_time_doctor = 0 ##NEW

    # A generator function that represents the DES generator for patient
    # referrals
    def generator_patient_referrals(self):
        # We use an infinite loop here to keep doing this indefinitely whilst
        # the simulation runs
        
        while True:
                   
            # Randomly sample the number of referrals coming in. Here, we
            # sample from a Poisson distribution as recommended by Michael Allen 
            # The mean referral rate is stored in the g class.
            sampled_referrals_poisson = np.random.poisson(lam=g.mean_referrals,
                                                  size=g.sim_duration)
            
            sampled_referrals = int(random.choice(sampled_referrals_poisson))
            
            self.referral_counter = 0
            #increment week number by 1
            self.week_number += 1
          
            while self.referral_counter < sampled_referrals:

                # Increment the patient counter by 1 (this means our first 
                # patient will have an ID of 1)
                self.patient_counter += 1

                self.place_on_waiting_list += 1

                # Create a new patient - an instance of the Patient Class we
                # defined above.  Remember, we pass in the ID when creating a
                # patient - so here we pass the patient counter to use as the ID
                p = Patient(self.patient_counter)

                self.referral_counter += 1

                # Tell SimPy to start up the pathway generator function with
                # this patient (the generator function that will model the
                # patient's journey through the system)
                self.env.process(self.pathway(p))

            # Freeze this instance of this function in place for one
            # unit of time
            # reset referral counter ready for next batch
            self.referral_counter = 0
            # store weekly waiting list position
            g.weekly_wl_tracker.append({'Run Number':self.run_number,
                                        'Week Number':self.week_number,
                                        'Waiting List':self.place_on_waiting_list})

            yield self.env.timeout(1)

    # A generator function is required that goes through each step of the 
    # pathway.
    # The patient object is passed into the generator function so we can 
    # extract information from / record information to it
    
    def pathway(self, patient):
        start_q_assessment = self.env.now

        with self.caseload.request() as req: # request a caseload resource
            yield req
        
         # take a patient off the waiting list once they enter the pathway
            
            self.place_on_waiting_list -= 1

            end_q_assessment = self.env.now

            # sample the waiting list
            self.results_df.at[patient.id, "Waiting List"] = \
                                            self.place_on_waiting_list

            self.results_df.at[patient.id, 'Week Number'] = self.week_number

            patient.q_time_assessment = end_q_assessment - \
                                            start_q_assessment

            sampled_asst_wait_time = random.expovariate(
                1.0 / g.mean_asst_time
            )
            
            self.results_df.at[patient.id, "Q Time Enter Pathway"] = (
                    patient.q_time_assessment
            )
            self.results_df.at[patient.id, "Wait Time Asst"] = (
                    sampled_asst_wait_time
            )

            #yield self.env.timeout(sampled_asst_wait_time)

            ## this part is for the waiting time for school asst and obs
        
            sampled_school_wait_time = random.expovariate(
                1.0 / g.mean_school_time
            )
            # there's no queue time for this as it immediately follows asst
            #self.results_df.at[patient.id, "Q Time to Enter Pathway"] = (
                    #patient.q_time_assessment
            #)
            self.results_df.at[patient.id, "Wait Time School"] = (
                    sampled_school_wait_time
            )

            #yield self.env.timeout(sampled_school_wait_time)

            ## this part is for the waiting time for diagnosis
        
            sampled_diag_wait_time = random.expovariate(
                1.0 / g.mean_diag_time
            )
            # there's no queue time for this as it immediately follows school
            #self.results_df.at[patient.id, "Q Time to Enter Pathway"] = (
                    #patient.q_time_assessment
            #)
            self.results_df.at[patient.id, "Wait Time Diag"] = (
                    sampled_diag_wait_time
            )

            #yield self.env.timeout(sampled_diag_wait_time)

            ## this part is for the waiting time for titration
        
            sampled_titrate_wait_time = random.expovariate(
                1.0 / g.mean_titrate_time
            )
            # there's no queue time for this as it immediately follows diagnosis
            #self.results_df.at[patient.id, "Q Time to Enter Pathway"] = (
                    #patient.q_time_assessment
            #)
            self.results_df.at[patient.id, "Wait Time Titr"] = (
                    sampled_titrate_wait_time
            )

            # release the resource once all steps of the pathway are completed
            yield self.env.timeout(sampled_asst_wait_time+
                                   sampled_school_wait_time+
                                   sampled_diag_wait_time+
                                   sampled_titrate_wait_time)
        
        # This method calculates results over a single run.  Here we just calculate
        # a mean, but in real world models you'd probably want to calculate more.
   
    def calculate_run_results(self):
        # Take the mean of the queuing times across patients in this run of the 
        # model.
        self.mean_q_time_asst = self.results_df["Q Time Enter Pathway"].mean()
        self.max_waiting_list = self.results_df["Waiting List"].max()
        self.mean_wait_time_asst = self.results_df["Wait Time Asst"].mean()
        self.mean_wait_time_school = self.results_df["Wait Time School"].mean()
        self.mean_wait_time_diag = self.results_df["Wait Time Diag"].mean()
        self.mean_wait_time_titr = self.results_df["Wait Time Titr"].mean()

    # The run method starts up the DES entity generators, runs the simulation,
    # and in turns calls anything we need to generate results for the run
    def run(self):
        # Start up our DES entity generators that create new patients.  We've
        # only got one in this model, but we'd need to do this for each one if
        # we had multiple generators.
        self.env.process(self.generator_patient_referrals())

        # Run the model for the duration specified in g class
        self.env.run(until=g.sim_duration)

        # Now the simulation run has finished, call the method that calculates
        # run results
        self.calculate_run_results()

        # Print the run number with the patient-level results from this run of 
        # the model
        print (f"Run Number {self.run_number}")
        print (self.results_df)

# Class representing a Trial for our simulation - a batch of simulation runs.
class Trial:
    # The constructor sets up a pandas dataframe that will store the key
    # results from each run against run number, with run number as the index.
    def  __init__(self):
        self.df_trial_results = pd.DataFrame()
        self.df_trial_results["Run Number"] = [0]
        self.df_trial_results["Max Waiting List"] = [0]
        self.df_trial_results["Mean Q Time Enter Pathway"] = [0.0]
        self.df_trial_results["Mean Wait Time Asst"] = [0.0]
        self.df_trial_results["Mean Wait Time School"] = [0.0]
        self.df_trial_results["Mean Wait Time Diag"] = [0.0]
        self.df_trial_results["Mean Wait Time Titr"] = [0.0]
        self.df_trial_results.set_index("Run Number", inplace=True)

    # Method to print out the results from the trial.  In real world models,
    # you'd likely save them as well as (or instead of) printing them
    def print_trial_results(self):
        print ("Trial Results")
        print (self.df_trial_results)

    # Method to run a trial
    def run_trial(self):
        # Run the simulation for the number of runs specified in g class.
        # For each run, we create a new instance of the Model class and call its
        # run method, which sets everything else in motion.  Once the run has
        # completed, we grab out the stored run results (just mean queuing time
        # here) and store it against the run number in the trial results
        # dataframe.
        for run in range(g.number_of_runs):
            my_model = Model(run)
            my_model.run()
            
            self.df_trial_results.loc[run] =    [my_model.max_waiting_list,
                                                my_model.mean_q_time_asst,
                                                my_model.mean_wait_time_asst,
                                                my_model.mean_wait_time_school,
                                                my_model.mean_wait_time_diag,
                                                my_model.mean_wait_time_titr]
            
        # Once the trial (ie all runs) has completed, print the final results
        return self.df_trial_results

# Create an instance of the Trial class
my_trial = Trial()

# Call the run_trial method of our Trial object
df_trial_results = my_trial.run_trial()

weekly_wl_position = pd.DataFrame(g.weekly_wl_tracker)

weekly_wl_position['Run Number'] = 'Run Number ' + weekly_wl_position['Run Number'].astype(str)

# get the average waiting list across all the runs
weekly_avg_wl = weekly_wl_position.groupby(['Week Number'
                                            ])['Waiting List'].mean().reset_index()

weekly_wl_position.to_csv('adhd_weekly_wl.csv')

weekly_avg_wl.to_csv('adhd_avg_wl.csv')

df_trial_results = pd.DataFrame(df_trial_results)

df_trial_results.to_csv('adhd_trial_results.csv')

st.write(df_trial_results)

fig = px.line(weekly_wl_position,x="Week Number" ,y="Waiting List",color="Run Number", title='ADHD Diagnosis Waiting List by Week')

fig.update_traces(line=dict(color="Blue", width=0.5))

fig.update_layout(title_x=0.4)

fig.add_trace(
    go.Scatter(x=weekly_avg_wl["Week Number"],y=weekly_avg_wl["Waiting List"], name='Average'))

fig.update_layout(xaxis_title='Week Number',
                  yaxis_title='Patients Waiting')

st.plotly_chart(fig, use_container_width=True)