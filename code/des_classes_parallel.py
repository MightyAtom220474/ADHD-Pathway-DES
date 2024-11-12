import simpy
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# This model aims to simulate the flow of CYP through the ADHD clinical pathway
# Assumptions - CYP stay on caseload until they are 18
# only accepted referralS flow through the pathway
# Same clinician will take CYP through each of the steps
# Overall caseload = clinician_caseload * number_adhd_clinicians
# repeat titration taken out due to complexities of modelling for this
# assume all patients get a triage and it doesn't stop patients flowing through
# the rest of the pathway
# assessment doesn't start until there is space on a clinicians caseload
 
########## Classes ##########

# Class to store global parameter values.  We don't create an instance of this
# class - we just refer to the class blueprint itself to access the numbers
# inside.
class g:
    
    # Referrals
    mean_referrals_pw = 5
    referral_rejection_rate = 0.05 # % of referrals rejected, assume 5%
    base_waiting_list = 2741 # current number of patients on waiting list
    
    # Triage
    target_triage_wait = 4 # triage within 4 weeks
    triage_waiting_list = 0 # number waiting for triage
    weekly_triage_wl_tracker = [] # container to hold triage w/l at end of week
    triage_rejection_rate = 0.05 # % rejected at triage, assume 5%
    triage_resource = 48 # number of triage slots p/w @ 10 mins
    #triage_admin_resource = 48 # number of triage slots p/w @ 30 mins
    total_triage_clinical_time = 0 # clinical time used on triages p/w
    total_triage_admin_time = 0 # # admin time used on triages p/w

    # School/Home Assesment Pack
    target_pack_wait = 3 # pack to be returned within 3 weeks
    pack_rejection_rate = 0.03 # % rejected based on pack assume 3%
    
    # QB and Observations
    target_obs_wait = 4 # QB and School obs to be completed within 4 weeks
    obs_rejection_rate = 0.02 # % rejected due to obs not taking place assume 1%
    
    # MDT
    target_mdt_wait = 1 # how long did it take to be reviewed at MDT, assume 1 week
    mdt_rejection_rate = 0.05 # % rejected at MDT, assume 5%
    weekly_mdt_wl_tracker = [] # container to hold mdt w/l at end of each week
    mdt_resource = 25 # no. of MDT slots p/w, assume 1 per day & review 5 cases
    
    # Assessment
    target_asst_wait = 4 # assess within 4 weeks
    weekly_asst_wl_tracker = [] # container to hold asst w/l at end of each week
    asst_resource = 62 # number of assessment slots p/w @ 60 mins
    #asst_admin_resource = 62 # number of assessment slots p/w @ 90 mins
    asst_rejection_rate = 0.01 # % found not to have ADHD, assume 1%
    
    # Diagnosis
    accepted_into_service = 0 # number accepted into service
    rejected_from_service = 0 # number rejected from service

    # Simulation
    sim_duration = 10
    number_of_runs = 2
    cores = -1 # number of cpu cores to use, -1 = all available
   
# Class representing patients coming in to the pathway
class Patient:
    def __init__(self, p_id):
        # Patient
        self.id = p_id
        
        # Referral
        self.referral_rejected = 0 # were they rejected at referral
        
        #Triage
        self.q_time_triage = 0 # how long they waited for triage
        self.place_on_triage_wl = 0 # position they are on Triage waiting list
        self.triage_rejected = 0 # were they rejected following triage
        
        # School/Home Assesment Pack
        self.pack_rejected = 0 # rejected as school pack not returned
        self.pack_time = 0 # actual time taken doing school pack
        
        # Observations
        self.obs_rejected = 0 # rejected as observations not completed
        self.obs_time = 0 # actual time taken doing observations

        # MDT
        self.q_time_mdt = 0 # how long they waited for MDT
        self.place_on_mdt_wl = 0 # position they are on MDT waiting list
        self.mdt_rejected = 0 # were they rejected following MDT

        # Assessment
        self.q_time_asst = 0 # how long they waited for assessment
        self.place_on_asst_wl = 0 # position they are on assessment waiting list
        self.asst_rejected = 0 # were they rejected following assessment

        # Diagnosis
        self.diagnosis_status = 0 # were they accepted or rejected
        
# Class representing our model of the ADHD clinical pathway
class Model:
    # Constructor to set up the model for a run. We pass in a run number when
    # we create a new model
    def __init__(self, run_number):
        # Create a SimPy environment in which everything will live
        self.env = simpy.Environment()

        # Create a counters for various metrics we want to record
        self.patient_counter = 0
        self.week_number = 0
        self.place_on_triage_wl = 0
        self.place_on_mdt_wl = 0
        self.place_on_asst_wl = 0

        # decide whether patient gets rejected at triage
        self.reject_triage = random.uniform(0,1)
        
        # Create our resources
        self.triage_res = simpy.Resource(
            self.env,capacity=g.triage_resource)
        
        self.mdt_res = simpy.Resource(
            self.env,capacity=g.mdt_resource)
        
        self.asst_res = simpy.Resource(
            self.env,capacity=g.asst_resource)

        # Store the passed in run number
        self.run_number = run_number

        # Create a new DataFrame that will store results against the patient ID
        self.results_df = pd.DataFrame()
        # Patient
        self.results_df['Patient ID'] = [1]
        # Referral
        self.results_df['Week Number'] = [0]
        self.results_df['Run Number'] = [0]
        self.results_df['Referral Rejected'] = [0]
        # Triage
        self.results_df['Q Time Triage'] = [0.0]
        self.results_df['Time to Triage'] = [0.0]
        self.results_df['Total Triage Time'] = [0.0]
        self.results_df['Triage WL Posn'] = [0]
        self.results_df['Triage Rejected'] = [0]
        # School Pack
        self.results_df['Return Time Pack'] = [0.0]
        self.results_df['Pack Rejected'] = [0]
        #School Obs
        self.results_df['Return Time Obs'] = [0.0]
        self.results_df['Obs Rejected'] = [0]
        # Triage
        self.results_df['Q Time MDT'] = [0.0]
        self.results_df['Time to MDT'] = [0.0]
        self.results_df['Total MDT Time'] = [0.0]
        self.results_df['MDT WL Posn'] = [0]
        self.results_df['MDT Rejected'] = [0]
        # Triage
        self.results_df['Q Time Asst'] = [0.0]
        self.results_df['Time to Asst'] = [0.0]
        self.results_df['Total Asst Time'] = [0.0]
        self.results_df['Asst WL Posn'] = [0]
        self.results_df['Asst Rejected'] = [0]
        # Diagnosis
        self.results_df['Diagnosis Rejected'] = [0]
        # Indexing
        self.results_df.set_index("Patient ID", inplace=True)

        # Create an attribute to store the mean queuing times across this run of
        # the model
        self.mean_q_time_triage = 0
        self.mean_q_time_mdt = 0
        self.mean_q_time_asst = 0
        
    # A generator function that represents the DES generator for patient
    # referrals
    def generator_patient_referrals(self):
        # We use an infinite loop here to keep generating referrals indefinitely
        # whilst the simulation runs
        
            while True:
                    
                # Randomly sample the number of referrals coming in. Here we
                # sample from Poisson distribution - recommended by Mike Allen 
                sampled_referrals_poisson = np.random.poisson(
                                    lam=g.mean_referrals_pw,size=g.sim_duration)
                # pick a value at random from the Poisson distribution
                sampled_referrals = \
                                int(random.choice(sampled_referrals_poisson))
                
                # referral counter to record how many referrals have been made
                self.referral_counter = 0
                
                # increment week number by 1
                self.week_number += 1

                # generate referrals while number of referrals < sampled value
                while self.referral_counter <= sampled_referrals:

                    # Increment the patient counter by 1 (this means our first 
                    # patient will have an ID of 1)
                    self.patient_counter += 1

                    # Randomly reject referrals
                    self.reject_referral = random.uniform(0,1)
                    # if this referral is deemed to be accepted do the following
                    if self.reject_referral > g.referral_rejection_rate:

                        # Mark referral as accepted
                        self.results_df['Rejected Referral'] = 0

                        # Create a new patient from Patient Class we
                        p = Patient(self.patient_counter)

                        self.referral_counter += 1

                        # Tell SimPy to start up the triage generator function 
                        self.env.process(self.triage(p))
                                          
                    # otherwise simply mark the referral as rejected
                    else: 
                        self.results_df['Rejected Referral'] = 1

                    # record the Triage WL position at end of that week
                    self.weekly_triage_wl_tracker = ({'Run Number':self.run_number,
                                                'Week Number':self.week_number,
                                                'Triage Waiting List':self.
                                                place_on_triage_wl})
                    
                    # Tell SimPy to start up the pack generator function
                    self.env.process(self.pack(p))

                    # Tell SimPy to start up the obs generator function
                    self.env.process(self.obs(p))

                    # Tell SimPy to start up the MDT generator function
                    self.env.process(self.mdt(p))

                    self.weekly_mdt_wl_tracker = ({'Run Number':self.run_number,
                                                'Week Number':self.week_number,
                                                'MDT Waiting List':self.
                                                place_on_mdt_wl})
                    
                    # Tell SimPy to start up the assessment generator function
                    self.env.process(self.assessment(p))

                    self.weekly_asst_wl_tracker = ({'Run Number':self.run_number,
                                                'Week Number':self.week_number,
                                                'Asst Waiting List':self.
                                                place_on_asst_wl})
                               
                # reset referral counter ready for next batch                
                self.referral_counter = 0
                # Freeze this instance of this function in place for one
                # unit of time i.e. 1 week
                yield self.env.timeout(1)

                return self.weekly_triage_wl_tracker

    # Generator function for the Triage
    def triage(self, patient):
        
        # only run if the referral wasn't rejected
        if self.reject_referral > g.referral_rejection_rate:

            start_q_triage = self.env.now

            self.place_on_triage_wl += 1

            # Record where they patient is on the Triage WL
            self.results_df.at[patient.id, "Triage WL Posn"] = \
                                                self.place_on_triage_wl
            # Wait until a Triage resource becomes available
            with self.triage_res.request() as req:
                yield req
            
                # take patient off Triage waiting list once they are Triaged
                self.place_on_triage_wl -= 1

                end_q_triage = self.env.now
                # pick a random time from 1-4 for how long it took to Triage
                sampled_triage_time = round(random.uniform(0,4),1)
                    
                self.results_df.at[patient.id, 'Week Number'] = self.week_number
            
                # Calculate how long it took the patient to be Triaged
                self.q_time_triage = end_q_triage - start_q_triage

                p = Patient(self.patient_counter)
            
                # Record how long the patient waited to be Triaged
                self.results_df.at[patient.id, 'Q Time Triage'] = \
                                                        (self.q_time_triage)
                # Record how long the patient took to be Triage
                self.results_df.at[patient.id, 'Time to Triage'] = \
                                                        sampled_triage_time
                # Record total time it took to triage patient 
                self.results_df.at[patient.id, 'Total Triage Time'] = \
                                                            (sampled_triage_time  
                                                            +(end_q_triage - 
                                                            start_q_triage))
                
                # Determine whether or not patient was rejected following Triage
                #
                                
                if self.reject_triage > g.triage_rejection_rate:
                    self.results_df.at[patient.id, 'Triage Rejected'] = 0
                else:
                    self.results_df.at[patient.id, 'Triage Rejected'] = 1
            
            # release the resource once the Triage is completed
            yield self.env.timeout(sampled_triage_time)
    
    # Generator function for the Pack - there are no waits for this part
    def pack(self, patient):

        p = Patient(self.patient_counter)

        # if the triage wasn't rejected start up the pack generator function
        if self.reject_triage > g.triage_rejection_rate:
           
            self.env.process(self.pack(p))
            # decide whether the pack was returned on time or not
            self.reject_pack = random.uniform(0,1)
                           
            # pick a random time for how long it took for Pack to be returned
            if self.reject_pack > g.pack_rejection_rate:
                self.sampled_pack_time = round(random.uniform(0,3),1)
            else:
                self.sampled_pack_time = round(random.uniform(3,5),1)
            self.results_df.at[Patient.id, 'Week Number'] = self.week_number
            
            # Record how long the pack took to be returned
            self.results_df.at[Patient.id, 'Return Time Pack'] = \
                                                    self.sampled_pack_time
            
            # Mark whether or not the patient was rejected depending on
            # pack returned time
            if self.reject_pack > g.pack_rejection_rate:
                self.results_df.at[Patient.id, 'Pack Rejected'] = 0
            else:
                self.results_df.at[Patient.id, 'Pack Rejected'] = 1
        
            # release the resource once the Pack is returned
            yield self.env.timeout(self.sampled_pack_time)

    # Generator function for the Obs - there are no waits for this part
    def obs(self, patient):

        # if the Pack wasn't rejected start up the Obs generator function
        if self.reject_pack > g.pack_rejection_rate:

            p = Patient(self.patient_counter)
            
            self.env.process(self.obs(p))
            
            # decide whether the Obs were returned on time or not
            self.reject_obs = random.uniform(0,1)
                           
            # pick a random time for how long it took for Obs to be returned
            if self.reject_obs > g.obs_rejection_rate:
                self.sampled_obs_time = round(random.uniform(0,4),1)
            else:
                self.sampled_obs_time = round(random.uniform(4,6),1)
            self.results_df.at[Patient.id, 'Week Number'] = self.week_number
            
            # Record how long the patient took for Obs
            self.results_df.at[Patient.id, 'Return Time Obs'] = \
                                                        self.sampled_obs_time
                              
            # Mark whether or not the patient was rejected depending on
            # pack returned time
            if self.reject_obs > g.obs_rejection_rate:
                self.results_df.at[Patient.id, 'Obs Rejected'] = 0
            else:
                self.results_df.at[Patient.id, 'Obs Rejected'] = 1
        
            # release the resource once the Obs are completed
            yield self.env.timeout(self.sampled_obs_time)
        
    # Generator function for the MDT
    def mdt(self, patient):

        # if the patient wasn't rejected start up the MDT generator function
        if self.reject_obs > g.obs_rejection_rate:
            self.env.process(self.mdt(p))

            start_q_mdt = self.env.now

            self.place_on_mdt_wl += 1

            # Record where they patient is on the MDT WL
            self.results_df.at[Patient.id, "MDT WL Posn"] = \
                                                self.place_on_mdt_wl
            # Wait until an MDT resource becomes available
            with self.mdt_res.request() as req: # request an MDT resource
                yield req
            
                # take patient off the MDT waiting list once MDT has taken place
                self.place_on_mdt_wl -= 1

                end_q_mdt = self.env.now
                # pick a random time from 0-1 weeks for how long it took for MDT
                sampled_mdt_time = round(random.uniform(0,1),1)
                    
                self.results_df.at[Patient.id, 'Week Number'] = self.week_number
            
                # Calculate how long the patient waited to have MDT
                self.q_time_mdt = end_q_mdt - start_q_mdt

                p = Patient(self.patient_counter)
            
                # Record how long the patient waited to be MDT'd
                self.results_df.at[Patient.id, 'Q Time MDT'] = (self.q_time_mdt)
                # Record how long the patient took to be MDT'd
                self.results_df.at[Patient.id, 'Time to MDT'] = sampled_mdt_time
                # Record total time it took to MDT patient 
                self.results_df.at[Patient.id, 'Total MDT Time'] = \
                                                            (sampled_mdt_time  
                                                            +(end_q_mdt - 
                                                            start_q_mdt))
                self.reject_mdt = random.uniform(0,1)
                
                # Determine whether or not the patient was rejected following MDT
                if self.reject_mdt > g.mdt_rejection_rate:
                    self.results_df.at[Patient.id, 'MDT Rejected'] = 0
                else:
                    self.results_df.at[Patient.id, 'MDT Rejected'] = 1
            
                # release the resource once the Triage is completed
                yield self.env.timeout(sampled_mdt_time)

    # Generator function for the Assessment
    def assessment(self, patient):

        # if patient wasn't rejected at MDT start up the Asst generator function
        if self.reject_mdt > g.mdt_rejection_rate:
            self.env.process(self.assessment(p))
    
            start_q_asst = self.env.now

            self.place_on_asst_wl += 1
            
            # Record where they patient is on the MDT WL
            self.results_df.at[Patient.id, "Asst WL Posn"] = \
                                                        self.place_on_asst_wl
            # Wait until an Assessment resource becomes available
            with self.asst_res.request() as req:
                yield req
            
                # take patient off the Asst waiting list once Asst starts
                self.place_on_asst_wl -= 1

                end_q_asst = self.env.now
                
                # pick a random time from 1-4 for how long it took to Assess
                sampled_asst_time = round(random.uniform(0,4),1)
                    
                self.results_df.at[Patient.id, 'Week Number'] = self.week_number
            
                # Calculate how long it took the patient to be Assessed
                self.q_time_asst = end_q_asst - start_q_asst

                p = Patient(self.patient_counter)
            
                # Record how long the patient waited to be Assessed
                self.results_df.at[Patient.id, 'Q Time Asst'] = \
                                                            (self.q_time_asst)
                # Record how long the patient took to be Triage
                self.results_df.at[Patient.id, 'Time to Asst'] = \
                                                            sampled_asst_time
                # Record total time it took to triage patient 
                self.results_df.at[Patient.id, 'Total Asst Time'] = \
                                                            (sampled_asst_time  
                                                            +(end_q_asst - 
                                                            start_q_asst))
                # Randomly determine whether the patient was rejected at Asst
                self.reject_asst = random.uniform(0,1)
                
                # Determine whether patient was rejected following assessment
                if self.reject_asst > g.asst_rejection_rate:
                    self.results_df.at[Patient.id, 'Asst Rejected'] = 0
                else:
                    self.results_df.at[Patient.id, 'Asst Rejected'] = 1
            
                # release the resource once the Assessment is completed
                yield self.env.timeout(sampled_asst_time)

    # This method calculates results over each single run
    def calculate_run_results(self):
        # Take the mean of the queuing times and the maximum waiting list 
        # across patients in this run of the model
        self.mean_q_time_triage = self.results_df["Mean Q Time Triage"].mean()
        self.max_triage_wl = self.results_df["Max Triage WL"].max()
        self.mean_q_time_mdt = self.results_df["Mean Q Time MDT"].mean()
        self.max_mdt_wl = self.results_df["Max MDT WL"].max()
        self.mean_q_time_asst = self.results_df["Mean Q Time Asst"].mean()
        self.max_asst_wl = self.results_df["Max Asst WL"].max()

    # The run method starts up the DES entity generators, runs the simulation,
    # and in turns calls anything we need to generate results for the run
    def run(self):

        # Start up our DES entity generators to create new referrals 
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

        self.df_trial_results =[]
        # self.df_trial_results = pd.DataFrame()
        # self.df_trial_results["Run Number"] = [0]
        # self.df_trial_results["Mean Q Time Triage"] = [0.0]
        # self.df_trial_results["Max Triage WL"] = [0]
        # self.df_trial_results["Mean Q Time MDT"] = [0.0]
        # self.df_trial_results["Max MDT WL"] = [0]
        # self.df_trial_results["Mean Q Time Asst"] = [0.0]
        # self.df_trial_results["Max Asst WL"] = [0]
        # self.df_trial_results.set_index("Run Number", inplace=True)

    # Method to print out the results from the trial.  In real world models,
    # you'd likely save them as well as (or instead of) printing them
    def process_trial_results(self):
        self.df_trial_results = pd.DataFrame(self.df_trial_results)
        self.df_trial_results.set_index("Run Number", inplace=True)
    
    def run_single(self, run):
        # For each run, we create a new instance of the Model class and call its
        # run method, which sets everything else in motion.  Once the run has
        # completed, we grab out the stored run results (just mean queuing time
        # here) and store it against the run number in the trial results
        # dataframe.
        random.seed(run)

        my_model = Model(run)
        patient_level_results = my_model.run()

        results = {"Run Number":run,
            "Arrivals": len(patient_level_results),
            "Mean Q Time Triage": my_model.mean_q_time_triage,
            "Max Triage WL": my_model.max_triage_wl,
            "Mean Q Time MDT": my_model.mean_q_time_mdt,
            "Max MDT WL": my_model.max_mdt_wl,
            "Mean Q Time Asst": my_model.mean_q_time_asst,
            "Max Asst WL": my_model.max_asst_wl
            }

        return results

    def print_trial_results(self):
        print ("Trial Results")
        print (self.df_trial_results)

    # Method to run a trial
    def run_trial(self):
        # Run the simulation for the number of runs specified in g class.
        # For each run, we create a new instance of the Model class and call its
        # run method, which sets everything else in motion.  Once the run has
        # completed, we grab out the stored run results and store it against 
        # the run number in the trial results dataframe
        self.df_trial_results = Parallel(n_jobs=g.cores)(delayed(self.run_single)(run) for run in range(g.number_of_runs))
            # my_model = Model(run)
            # my_model.run()
            
            # self.df_trial_results.loc[run] =    [my_model.mean_q_time_triage,
            #                                     my_model.max_triage_wl,
            #                                     my_model.mean_q_time_mdt,
            #                                     my_model.max_mdt_wl,
            #                                     my_model.mean_q_time_asst,
            #                                     my_model.max_asst_wl,
            #                                     ]
            
        # Once the trial (i.e. all runs) has completed, print the final results
        self.process_trial_results()
        self.print_trial_results()
        
        return self.df_trial_results
    
my_trial = Trial()

#my_trial.run_trial(cores=4)

# Call the run_trial method of our Trial class object

df_trial_results = my_trial.run_trial()

df_trial_results

########## maybe turn the next bit into a function as do it 3 times
weekly_triage_wl_position = pd.DataFrame(g.weekly_triage_wl_tracker)

weekly_triage_wl_position['Run Number'] = 'Run Number ' + weekly_triage_wl_position['Run Number'].astype(str)

# get the average waiting list across all the runs
weekly_triage_avg_wl = weekly_triage_wl_position.groupby(['Week Number'
                                            ])['Waiting List'].mean().reset_index()

weekly_triage_wl_position.to_csv('adhd_triage_weekly_wl.csv')

weekly_triage_avg_wl.to_csv('adhd_triage_avg_wl.csv')

weekly_mdt_wl_position = pd.DataFrame(g.weekly_mdt_wl_tracker)

weekly_mdt_wl_position['Run Number'] = 'Run Number ' + weekly_mdt_wl_position['Run Number'].astype(str)

# get the average waiting list across all the runs
weekly_mdt_avg_wl = weekly_mdt_wl_position.groupby(['Week Number'
                                            ])['Waiting List'].mean().reset_index()

weekly_mdt_wl_position.to_csv('adhd_mdt_weekly_wl.csv')

weekly_mdt_avg_wl.to_csv('adhd_mdt_avg_wl.csv')

weekly_asst_wl_position = pd.DataFrame(g.weekly_asst_wl_tracker)

weekly_asst_wl_position['Run Number'] = 'Run Number ' + weekly_asst_wl_position['Run Number'].astype(str)

# get the average waiting list across all the runs
weekly_asst_avg_wl = weekly_asst_wl_position.groupby(['Week Number'
                                            ])['Waiting List'].mean().reset_index()

weekly_asst_wl_position.to_csv('adhd_asst_weekly_wl.csv')

weekly_asst_avg_wl.to_csv('adhd_asst_avg_wl.csv')

df_trial_results = pd.DataFrame(df_trial_results)

df_trial_results.to_csv('adhd_trial_results.csv')
