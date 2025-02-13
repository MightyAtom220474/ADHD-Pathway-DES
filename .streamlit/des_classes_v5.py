import simpy
import random
import numpy as np
import pandas as pd

# This model aims to simulate the flow of CYP through the ADHD clinical pathway
# Assumptions - CYP stay on caseload until they are 18
# only accepted referralS flow through the pathway
# Same clinician will take CYP through each of the steps
# assume all patients get a triage and it doesn't stop patients flowing through
# the rest of the pathway
# assessment doesn't start until there is space on a clinicians caseload

########## Classes ##########

# Class to store global parameter values.  We don't create an instance of this
# class - we just refer to the class blueprint itself to access the numbers
# inside.
class g:

    debug_level = 2

    # Referrals
    mean_referrals_pw = 60
    referral_rejection_rate = 0.05 # % of referrals rejected, assume 5%
    referral_screen_time = 15

    # Triage
    target_triage_wait = 4 # triage within 4 weeks
    triage_average_wait = 4 # current avg wait for triage for patients on the waiting list
    triage_waiting_list = 0 # current number of patients on waiting list
    triage_rejection_rate = 0.05 # % rejected at triage, assume 5%
    triage_clin_time = 60 # number of mins for clinician to do triage
    triage_admin_time = 15 # number of mins of admin to do triage
    triage_discharge_time = 45 # time taken if discharged at triage

    # School/Home Assesment Pack
    target_pack_wait = 3 # pack to be returned within 3 weeks
    pack_rejection_rate = 0.03 # % rejected based on pack assume 3%
    pack_admin_time = 30 # B4 admin sending out pack
    pack_reject_time = 45 # time taken if patient rejected at this stage
    
    # QB and Observations
    target_obs_wait = 4 # QB and School obs to be completed within 4 weeks
    obs_rejection_rate = 0.02 # % rejected due to obs not taking place assume 1%
    qb_test_time = 90 # tike taken by B4 to do QB and write up
    school_obs_time = 180 # time taken for B4 to do obs incl travel
    obs_reject_time = 45 # time taken if patient rejected at this stage

    # MDT
    target_mdt_wait = 1 # how long did it take to be reviewed at MDT, assume 1 week
    mdt_rejection_rate = 0.05 # % rejected at MDT, assume 5%
    mdt_resource = 30 # no. of MDT slots p/w, assume 1 mdt/wk @1hr & review 6 cases
    mdt_meet_time = 60 # number of mins to do MDT
    mdt_prep_time = 90 # time take for B4 to prep case for MDT
    mdt_reject_time = 45 # time taken if patient rejected at this stage

    # Assessment
    target_asst_wait = 4 # assess within 4 weeks
    asst_average_wait = 52 # current avg wait for triage for patients on the waiting list
    asst_resource = 62 # number of assessment slots p/w @ 60 mins
    asst_clin_time = 90 # number of mins for clinician to do asst
    asst_admin_time = 90 # number of mins of admin following asst
    asst_rejection_rate = 0.01 # % found not to have ADHD, assume 1%
    asst_waiting_list = 0 # current number of patients on waiting list

    # Diagnosis
    diag_time_disch = 90 # time taken after asst if discharged
    diag_time_accept = 150 # time taken after asst if accepted

    # Job Plans
    number_staff_b6_prac = 9.0
    number_staff_b4_prac = 10.0
    hours_avail_b6_prac = 20.0
    hours_avail_b4_prac = 22.0
    staff_weeks_lost = 10
    weeks_lost_pc = (52-staff_weeks_lost)/52
    triage_resource = int(48*weeks_lost_pc) # number of triage slots p/w @ 10 mins
    asst_resource = int(62*weeks_lost_pc) # number of assessment slots p/w @ 60 mins
    
    # Simulation
    sim_duration = 52
    number_of_runs = 1
    std_dev = 3 # used for randomising activity times
    active_patients = 0
    
    # checks whether the prefill step need to be run based on whether a waiting
    # list has been input
    if triage_waiting_list + asst_waiting_list > 0:
        prefill = True
    else:
        prefill = False
    # Result storage
    all_results = []
    weekly_wl_posn = pd.DataFrame() # container to hold w/l position at end of week
    number_on_triage_wl = 0 # used to keep track of triage WL position
    number_on_mdt_wl = 0 # used to keep track of MDT WL position
    number_on_asst_wl = 0 # used to keep track of asst WL position

# Class representing patients coming in to the pathway

# SR comment
# make it clear what the possible states are for the rejection statuses
# e.g. is this 0 = rejected, 1 = not rejected?

class Patient:
    def __init__(self, p_id):
        # Patient
        self.id = p_id
        self.patient_source = [] # where the patient came from - referral gen or WL

        self.week_added = None # Week they were added to the waiting list (for debugging purposes)

        # Referral
        self.referral_rejected = 0 # were they rejected at referral
        self.referral_time_screen = 0 # B6 time taken to screen referral

        #Triage
        self.q_time_triage = 0 # how long they waited for triage
        self.triage_time_clin = 0 # how long the triage took clinician in minutes
        self.triage_time_admin = 0 # how long the triage took admin in minutes
        self.place_on_triage_wl = 0 # position they are on Triage waiting list
        self.triage_rejected = 0 # were they rejected following triage
        self.triage_time_disch = 0 # time take if rejected at this stage
        self.triage_already_seen = False # has the patient already been seen for triage
        self.triage_wl_added = False # was the patient added from the triage WL
        
        # School/Home Assesment Pack
        self.pack_rejected = 0 # rejected as school pack not returned
        self.pack_time = 0 # actual time taken doing school pack
        self.pack_time_admin = 0 # admin sending out pack in mins
        self.pack_time_reject = 0 # time taken notifying patient if rejected

        # Observations
        self.obs_rejected = 0 # rejected as observations not completed
        self.obs_time = 0 # actual time taken doing observations
        self.obs_time_clin = 0 # clicical time doing obs incl travel

        # MDT
        self.mdt_time_prep = 0 # time taken by B4 prepping for MDT
        self.q_time_mdt = 0 # how long they waited for MDT
        self.place_on_mdt_wl = 0 # position they are on MDT waiting list
        self.mdt_rejected = 0 # were they rejected following MDT
        self.mdt_time_reject = 0 # time taken notifying patient if rejected
        self.mdt_already_seen = False # has the patient already been seen for MDT
        self.mdt_wl_added = False # was the patient added from the MDT WL

        # Assessment
        self.q_time_asst = 0 # how long they waited for assessment
        self.asst_time_clin = 0 # how long the asst took clinician in minutes
        self.asst_time_admin = 0 # how long the asst took to write up in minutes
        self.place_on_asst_wl = 0 # position they are on assessment waiting list
        self.asst_rejected = 0 # were they rejected following assessment
        self.asst_already_seen = False # has the patient already been seen for asst
        self.asst_wl_added = False # was the patient added from the MDT WL
        
        # Diagnosis
        self.diagnosis_status = 0 # were they accepted or rejected
        self.diag_time_reject = 0 # time taken notifying if rejected
        self.diag_time_accept = 0 # time taken notifying if accepted

# Class representing our model of the ADHD clinical pathway
class Model:
    # Constructor to set up the model for a run. We pass in a run number when
    # we create a new model
    def __init__(self, run_number):
        # Create a SimPy environment in which everything will live
        self.env = simpy.Environment()

        # # Create counters for various metrics we want to record
        self.patient_counter = 0
        self.run_number = run_number

        # Store the passed in run number
        self.run_number = run_number

        # Create a new DataFrame that will store results against the patient ID
        self.results_df = pd.DataFrame()
        # Patient
        self.results_df['Patient ID'] = [1]
        # Referral
        self.results_df['Week Number'] = [0]
        self.results_df['Run Number'] = [0]
        self.results_df['Referral Time Screen'] = [0.0]
        self.results_df['Referral Rejected'] = [0]
        # Triage
        self.results_df['Q Time Triage'] = [0.0]
        self.results_df['Time to Triage'] = [0.0]
        self.results_df['Triage Mins Clin'] = [0.0]
        self.results_df['Triage Mins Admin'] = [0.0]
        self.results_df['Total Triage Time'] = [0.0]
        self.results_df['Triage WL Posn'] = [0]
        self.results_df['Triage Rejected'] = [0]
        self.results_df['Triage Time Reject'] = [0.0]
        # School Pack
        self.results_df['Time Pack Send'] = [0.0]
        self.results_df['Return Time Pack'] = [0.0]
        self.results_df['Pack Rejected'] = [0]
        self.results_df['Time Pack Reject'] = [0.0]
        # School Obs
        self.results_df['Time Obs Visit'] = [0.0]
        self.results_df['Return Time Obs'] = [0.0]
        self.results_df['Obs Rejected'] = [0]
        self.results_df['Time Obs Reject'] = [0.0]
        # MDT
        self.results_df['Q Time MDT'] = [0.0]
        self.results_df['Time to MDT'] = [0.0]
        self.results_df['Time Prep MDT'] = [0.0]
        self.results_df['Time Meet MDT'] = [0.0]
        self.results_df['Time to MDT'] = [0.0]
        self.results_df['Total MDT Time'] = [0.0]
        self.results_df['MDT WL Posn'] = [0]
        self.results_df['MDT Rejected'] = [0]
        self.results_df['MDT Time Reject'] = [0.0]
        # Asst
        self.results_df['Q Time Asst'] = [0.0]
        self.results_df['Time to Asst'] = [0.0]
        self.results_df['Asst Mins Clin'] = [0.0]
        self.results_df['Asst Mins Admin'] = [0.0]
        self.results_df['Total Asst Time'] = [0.0]
        self.results_df['Asst WL Posn'] = [0]
        self.results_df['Asst Rejected'] = [0]
        # Diagnosis
        self.results_df['Diag Rejected Time'] = [0.0]
        self.results_df['Diag Accepted Time'] = [0.0]
        # Indexing
        self.results_df.set_index("Patient ID", inplace=True)

        # Create an attribute to store the mean queuing times across this run of
        # the model
        self.mean_q_time_triage = 0
        self.mean_q_time_mdt = 0
        self.mean_q_time_asst = 0

    # random number generator for activity times
    def random_normal(self, mean, std_dev):
        while True:
            activity_time = random.gauss(mean, std_dev)
            if activity_time > 0:
                return activity_time

    # this function governs the whole process and kicks off the waiting list builder
    # and week runner, as and when required
    def governor_function(self):

        # create resources required for the sim
        # Create our resources which are appt slots for 1 week
        self.triage_res = simpy.Container(
            self.env,capacity=g.triage_resource,
            init=g.triage_resource
            )

        self.mdt_res = simpy.Container(
            self.env,
            capacity=g.mdt_resource,
            init=g.mdt_resource
            )

        self.asst_res = simpy.Container(
            self.env,
            capacity=g.asst_resource,
            init=g.asst_resource
            )

        if g.debug_level >= 2:
            print(f"Starting Triage Resource Level: {self.triage_res.level}")
            print(f"Starting MDT Resource Level: {self.mdt_res.level}")
            print(f"Starting Asst Resource Level: {self.asst_res.level}")
        
        # if the waiting lists need to be added run this first
        # wait until prefill has completed before setting the simulation running
        
        if g.triage_waiting_list+g.asst_waiting_list > 0: 
        
            # run prefilling of waiting lists first
            yield self.env.process(self.prefill_waiting_lists())
            # now run the week runner
            yield self.env.process(self.week_runner(g.sim_duration))

        # otherwise go straight to the week_runner to run the normal week_runner process
        else:
            yield self.env.process(self.week_runner(g.sim_duration))

    def week_runner(self,number_of_weeks):

        if g.debug_level >= 2:
            print(f"########## Week Runner Started ##########")
        
        # now we're in the proper sim start at week 0
        self.week_number = 0

        # list to hold weekly statistics
        self.df_weekly_stats = []

        if g.debug_level >= 2:
            print(f"Weekly Triage Resource Level: {self.triage_res.level}")
            print(f"Weekly MDT Resource Level: {self.mdt_res.level}")
            print(f"Weekly Asst Resource Level: {self.asst_res.level}")

        # top up the resources ready to start a full run and start recording data
        triage_amount_to_fill = g.triage_resource - self.triage_res.level
        mdt_amount_to_fill = g.mdt_resource - self.mdt_res.level
        asst_amount_to_fill = g.asst_resource - self.asst_res.level

        if triage_amount_to_fill > 0:
            if g.debug_level >= 2:
                print(f"Triage Level: {self.triage_res.level}")
                print(f"Putting in {triage_amount_to_fill}")

            self.triage_res.put(triage_amount_to_fill)

            if g.debug_level >= 2:
                print(f"New Triage Level: {self.triage_res.level}")

        if mdt_amount_to_fill > 0:
            if g.debug_level >= 2:
                print(f"MDT Level: {self.mdt_res.level}")
                print(f"Putting in {mdt_amount_to_fill}")

            self.mdt_res.put(mdt_amount_to_fill)

            if g.debug_level >= 2:
                print(f"New MDT Level: {self.mdt_res.level}")

        if asst_amount_to_fill > 0:
            if g.debug_level >= 2:
                print(f"Asst Level: {self.asst_res.level}")
                print(f"Putting in {asst_amount_to_fill}")

                self.asst_res.put(asst_amount_to_fill)

                if g.debug_level >= 2:
                    print(f"New asst Level: {self.asst_res.level}")

        while self.week_number <= number_of_weeks:
           
            if g.debug_level >= 1:
                print(
                    f"""
    ##################################
    # Week {self.week_number}
    ##################################
                    """
                    )
            
            if g.debug_level >= 1:
                print(f'Starting up the referral generator')
            # Start up the referral generator function
            self.env.process(self.generator_patient_referrals())
            self.referral_tot_screen = self.results_df['Referral Time Screen'
                                                                        ].sum()
            self.max_triage_wl = g.number_on_triage_wl # self.results_df["Triage WL Posn"].max()
            self.triage_rej = self.results_df["Triage Rejected"].sum()
            self.triage_avg_wait = self.results_df["Q Time Triage"].mean()
            self.triage_tot_clin = self.results_df['Triage Mins Clin'].sum()
            self.triage_tot_admin = self.results_df['Triage Mins Admin'].sum()
            self.triage_tot_reject = self.results_df['Triage Time Reject'].sum()
            self.pack_tot_send = self.results_df["Time Pack Send"].sum()
            self.pack_rej = self.results_df["Pack Rejected"].sum()
            self.pack_tot_rej = self.results_df["Time Pack Reject"].sum()
            self.obs_tot_visit = self.results_df["Time Obs Visit"].sum()
            self.obs_rej = self.results_df["Obs Rejected"].sum()
            self.obs_tot_rej = self.results_df["Time Obs Reject"].sum()
            self.mdt_tot_prep = self.results_df["Time Prep MDT"].sum()
            self.mdt_tot_meet = self.results_df["Time Meet MDT"].sum()
            self.max_mdt_wl = g.number_on_mdt_wl # self.results_df["MDT WL Posn"].max()
            self.mdt_tot_rej = self.results_df["MDT Time Reject"].sum()
            self.mdt_rej = self.results_df["MDT Rejected"].sum()
            self.mdt_avg_wait = self.results_df["Q Time MDT"].mean()
            self.max_asst_wl = g.number_on_asst_wl # self.results_df["Asst WL Posn"].max()
            self.asst_rej = self.results_df["Asst Rejected"].sum()
            self.asst_avg_wait = self.results_df["Q Time Asst"].mean()
            self.asst_tot_clin = self.results_df['Asst Mins Clin'].sum()
            self.asst_tot_admin = self.results_df['Asst Mins Admin'].sum()
            self.diag_tot_rej = self.results_df['Diag Rejected Time'].sum()
            self.diag_tot_acc = self.results_df['Diag Accepted Time'].sum()

            # weekly waiting list positions
            self.df_weekly_stats.append(
                {
                 'Week Number':self.week_number,
                 'Referral Screen Mins':self.referral_tot_screen,   
                 'Triage WL':self.max_triage_wl,
                 'Triage Rejects':self.triage_rej,
                 'Triage Wait':self.triage_avg_wait,
                 'Triage Clin Mins':self.triage_tot_clin,
                 'Triage Admin Mins':self.triage_tot_admin,
                 'Triage Reject Mins':self.triage_tot_reject,
                 'Pack Send Mins':self.pack_tot_send,
                 'Pack Rejects':self.pack_rej,
                 'Pack Reject Mins':self.pack_tot_rej,
                 'Obs Visit Mins':self.obs_tot_visit,
                 'Obs Rejects':self.obs_rej,
                 'Obs Reject Mins':self.obs_tot_rej,
                 'MDT Prep Mins':self.mdt_tot_prep,
                 'MDT Meet Mins':self.mdt_tot_meet,
                 'MDT WL':self.max_mdt_wl,
                 'MDT Rejects':self.mdt_rej,
                 'MDT Reject Mins':self.mdt_tot_rej,
                 'MDT Wait':self.mdt_avg_wait,
                 'Asst WL':self.max_asst_wl,
                 'Asst Rejects':self.asst_rej,
                 'Asst Wait':self.asst_avg_wait,
                 'Asst Clin Mins':self.asst_tot_clin,
                 'Asst Admin Mins':self.asst_tot_admin,
                 'Diag Reject Mins':self.diag_tot_rej,
                 'Diag Accept Mins':self.diag_tot_acc,
                }
                )
            
                     
            triage_amount_to_fill = g.triage_resource - self.triage_res.level
            mdt_amount_to_fill = g.mdt_resource - self.mdt_res.level
            asst_amount_to_fill = g.asst_resource - self.asst_res.level

            if triage_amount_to_fill > 0:
                if g.debug_level >= 2:
                    print(f"Triage Level: {self.triage_res.level}")
                    print(f"Putting in {triage_amount_to_fill}")

                self.triage_res.put(triage_amount_to_fill)

                if g.debug_level >= 2:
                    print(f"New Triage Level: {self.triage_res.level}")

            if mdt_amount_to_fill > 0:
                if g.debug_level >= 2:
                    print(f"MDT Level: {self.mdt_res.level}")
                    print(f"Putting in {mdt_amount_to_fill}")

                self.mdt_res.put(mdt_amount_to_fill)

                if g.debug_level >= 2:
                    print(f"New MDT Level: {self.mdt_res.level}")

            if asst_amount_to_fill > 0:
                if g.debug_level >= 2:
                    print(f"Asst Level: {self.asst_res.level}")
                    print(f"Putting in {asst_amount_to_fill}")

                self.asst_res.put(asst_amount_to_fill)

                if g.debug_level >= 2:
                    print(f"New asst Level: {self.asst_res.level}")

            # Wait one unit of simulation time (1 week)
            yield(self.env.timeout(1))

            # increment our week number tracker by 1 week
            self.week_number += 1

        # Do these steps after all weeks have been iterated through
        #self.combined_wl = pd.concat(self.df_weekly_stats,ignore_index=False)
        self.combined_wl = pd.DataFrame(self.df_weekly_stats)
       
    # generator function that represents the DES generator for referrals
    def generator_patient_referrals(self):

        # Randomly sample the number of referrals coming in. Here we
        # sample from Poisson distribution - recommended by Mike Allen
        # need to look at reproducibility
        # https://hsma-programme.github.io/hsma6_des_book/reproducibility.html#sec-robust
        
        self.sampled_referrals_poisson = np.random.poisson(
                            lam=g.mean_referrals_pw,
                            size=g.sim_duration*10
                            )
        # pick a value at random from the Poisson distribution
        self.sampled_referrals = \
                        int(random.choice(self.sampled_referrals_poisson))
        
        # # increment week number by 1
        # self.week_number += 1

        if g.debug_level >= 1:
            print(f'Week {self.week_number}: {self.sampled_referrals} referrals generated')
            print('')
            print(f'Still remaining on triage WL from last week: {g.number_on_triage_wl}')

            print('')
            print(f'Still remaining on mdt WL from last week: {g.number_on_mdt_wl}')

            print('')
            print(f'Still remaining on Assessment WL from last week: {g.number_on_asst_wl}')
            print("----------------")

        self.referral_counter = 0

        while self.referral_counter <= self.sampled_referrals:

            # increment the referral counter by 1
            self.referral_counter += 1

            # Increment patient counter by 1
            self.patient_counter += 1
            
            # create a patient from the patient class
            p = Patient(self.patient_counter)

            p.patient_source = 'Referral Gen'
            p.triage_wl_added = False
            p.triage_already_seen = False
            p.asst_wl_added = False
            p.asst_already_seen = False

            if g.debug_level >= 1:
                print(f'Week {self.week_number} Patient number {p.id} starting at {p.patient_source} created')

            # start up the patient pathway generator at the referral stage
            self.env.process(self.pathway_start_point(p,self.week_number))

        # reset the referral counter
        self.referral_counter = 0

        yield(self.env.timeout(0))

    def prefill_waiting_lists(self):
              
        if g.active_patients <= g.triage_waiting_list+g.asst_waiting_list:
            # first fill up the assessment waiting list
            if g.debug_level >= 1:
                print(f'Filling Assessment Waiting list with {g.asst_waiting_list+g.asst_resource} patients')

            # prefill triage waiting list and resources using values from g class
            for b in range(g.asst_waiting_list+g.asst_resource):

                # Increment patient counter by 1
                self.patient_counter += 1
                
                p = Patient(self.patient_counter)

                p.triage_wl_added = False
                p.triage_already_seen = False
                p.asst_wl_added = True
                p.asst_already_seen = False

                self.week_number = -1

                p.patient_source = 'Asst WL'

                if g.debug_level >= 1:
                    print(f'Week {self.week_number} Patient number {p.id} starting at {p.patient_source} created')

                self.env.process(self.pathway_start_point(p, self.week_number))
        
            if g.debug_level >= 1:
                print(f'Filling Triage Waiting list with {g.triage_waiting_list+g.triage_resource} patients')
            
            # prefill triage waiting list and resources using values from g class
            for a in range(g.triage_waiting_list+g.triage_resource):

                # Increment patient counter by 1
                self.patient_counter += 1
                
                p = Patient(self.patient_counter)

                p.triage_wl_added = True
                p.triage_already_seen = False
                p.asst_wl_added = False
                p.asst_already_seen = False

                self.week_number = -1

                p.patient_source = 'Triage WL'

                if g.debug_level >= 1:
                        print(f'Week {self.week_number} Patient number {p.id} starting at {p.patient_source} created')
                
                self.week_number = -1

                self.env.process(self.pathway_start_point(p, self.week_number))
        
        # move on without waiting
        yield self.env.timeout(0)
    
    # function to decide where patients start on the pathway depending on 
    # whether they come from waiting lists or the referral generator
    # this function is the merging point for both the prefill_waiting_lists and
    # week_runner functions
    def pathway_start_point(self,patient,week_number):
        
        p = patient

        self.week_number = week_number
        ##### decide how the patient is entering the system #####
        # if we are in the prefill stage do it this way
        # wait until prefill has completed before setting the simulation running
        
        if p.triage_already_seen == False and p.triage_wl_added == True:

            if g.debug_level >= 1:
                        print(f'Week {self.week_number} Patient number {p.id} coming from {p.patient_source} sent down Triage path {g.number_on_triage_wl} now waiting')
            # add them to the Triage Waiting List here
            g.number_on_triage_wl += 1

            # Increment number of patients in the system by 1
            g.active_patients += 1
            
            yield self.env.process(self.triage_pathway(p,self.week_number))  
        # otherwise start them at assessment as alreay triaged
        elif p.asst_already_seen == False and p.asst_wl_added == True:
            # add them to the Asst Waiting List here
            g.number_on_asst_wl += 1
            # Increment number of patients in the system by 1
            g.active_patients += 1
            if g.debug_level >= 1:
                        print(f'Week {self.week_number} Patient number {p.id} coming from {p.patient_source} sent down Asst path {g.number_on_asst_wl} now waiting')
            yield self.env.process(self.asst_pathway(p,self.week_number))  
                       
        # otherwise start the patient right at the beginning of the pathway
        else:
            if g.debug_level >= 1:
                        print(f'Week {self.week_number} Patient number {p.id} coming from {p.patient_source} sent down Referral path')
            # Increment number of patients in the system by 1
            g.active_patients += 1
            yield self.env.process(self.referral_pathway(p,self.week_number))  

    # generator function that represents the referral part of the pathway
    def referral_pathway(self, patient, week_number):

        p = patient

        self.week_number = week_number

        p.week_added = self.week_number

        if g.debug_level >= 1:
            print(f'Week {week_number}: Patient number {p.id} (added week {p.week_added}) going through Referral Screening')

        # decide whether the referral was rejected
        self.reject_referral = random.uniform(0,1)
        
        p.week_added = week_number

        self.results_df.at[p.id, 'Referral Time Screen'] = self.random_normal(g.referral_screen_time,g.std_dev)

        # check whether the referral was rejected or not
        if self.reject_referral <= g.referral_rejection_rate:

            # if this referral is rejected mark as rejected
            self.results_df.at[p.id, 'Run Number'] = self.run_number

            self.results_df.at[p.id, 'Week Number'] = self.week_number

            self.results_df.at[p.id, 'Referral Rejected'] = 1

            if g.debug_level >= 1:
                        print(f'Week {week_number} Patient number {p.id} REJECTED at Referral')

            yield self.env.timeout(0)

        else:
            # Mark referral as accepted and move on to Triage
            self.results_df.at[p.id, 'Referral Rejected'] = 0

            self.results_df.at[p.id, 'Run Number'] = self.run_number

            self.results_df.at[p.id, 'Week Number'] = self.week_number

            if g.debug_level >= 1:
                print(f'Week {week_number} Patient number {p.id} ACCEPTED at Referral')

            if g.debug_level >= 1:
                print(f'Patient {p.id} added in week {p.week_added} from {p.patient_source}, current triage wl:{g.number_on_triage_wl}')    

            ##### Now do the Triage #####
            # start up the triage patient pathway generator
            if g.debug_level >= 1:
                        print(f'Week {week_number} Patient number {p.id} sent for Triage')
            yield self.env.process(self.triage_pathway(p,self.week_number))

    def triage_pathway(self, patient,week_number):

        # wait until prefill has completed before setting the simulation running
        while g.active_patients <= g.triage_waiting_list+g.asst_waiting_list:

            pass
        
            # if g.debug_level >= 1:
            #         print(f'Week {week_number}: Patient number {patient.id} (added week {patient.week_added}) {g.number_on_triage_wl} on WL for Triage')

        p = patient

        self.week_number = week_number

        p.week_added = self.week_number
        # as soon as prefill has completed set the simulation running
        p = patient

        p.week_added = self.week_number

        self.week_number = week_number

        # if the patient comes from the referral generator and them to the WL here
        if p.patient_source == 'Referral Gen':
            g.number_on_triage_wl += 1

        if g.debug_level >= 1:
            print(f'Week {self.env.now}: Patient number {p.id} added to Triage WL from {p.patient_source}')

        # decide whether the triage was rejected
        self.reject_triage = random.uniform(0,1)
        
        ##### Now do the Triage #####
        start_q_triage = self.env.now

        # Record where the patient is on the Triage WL
        self.results_df.at[p.id, "Triage WL Posn"] = \
                                            g.number_on_triage_wl

        # Request a Triage resource from the container
        with self.triage_res.get(1) as triage_req:
            yield triage_req

            if g.debug_level >= 1:
                print(f'Week {self.week_number} Patient number {p.id} Triage Resource Allocated')
                print(f'Triage slots available: {self.triage_res.level} (of intended {g.triage_resource})')
                print(f'Week {self.week_number} Patient number {p.id} being put through Triage')
            
            # as each patient reaches this stage take them off Triage wl
            g.number_on_triage_wl -= 1

            if g.debug_level >= 1:
                print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) put through Triage')

            if g.debug_level >= 1:
                print(f'Patient {p.id} added in week {p.week_added} from {p.patient_source}, current Triage WL:{g.number_on_triage_wl}')
            
            end_q_triage = self.env.now
            # pick a random time from 1-4 for how long it took to Triage
            sampled_triage_time = round(random.uniform(0, 4), 1)

            # Calculate how long the patient waited to be Triaged
            if g.prefill == True:
                self.q_time_triage = (end_q_triage - start_q_triage) + g.triage_average_wait
            else:
                self.q_time_triage = end_q_triage - start_q_triage 

            # Record how long the patient waited to be Triaged
            self.results_df.at[p.id, 'Q Time Triage'] = \
                                                    (self.q_time_triage)
            # Record how long the patient took to be Triaged
            self.results_df.at[p.id, 'Time to Triage'] = \
                                            sampled_triage_time
            self.results_df.at[p.id,'Triage Mins Clin'] = \
                                            self.random_normal(
                                            g.triage_clin_time,g.std_dev)
            self.results_df.at[p.id,'Triage Mins Admin'] = \
                                        self.random_normal(
                                        g.triage_admin_time,g.std_dev)

            # Record total time it took to triage patient
            self.results_df.at[p.id, 'Total Triage Time'] = \
                                                    (sampled_triage_time
                                                    +(end_q_triage -
                                                    start_q_triage))

            # Determine whether patient was rejected following triage
            if self.reject_triage <= g.triage_rejection_rate:

                self.results_df.at[p.id, 'Triage Rejected'] = 1
                
                self.results_df.at[p.id, 'Triage Time Reject'] = \
                    self.random_normal(g.triage_discharge_time,g.std_dev)
                
                if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) REJECTED at Triage')

                yield self.env.timeout(0)
            
            else:
                # record that the Triage was accepted
                self.results_df.at[p.id, 'Triage Rejected'] = 0

                if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) ACCEPTED at Triage')

                ##### Now send out the Pack #####
                # start up the pack patient pathway generator
                yield self.env.process(self.pack_pathway(p))

    def pack_pathway(self,patient):
        
        p = patient

        self.results_df.at[p.id, 'Time Pack Send'] = self.random_normal(\
                                            g.pack_admin_time,g.std_dev)

        # decide whether the pack was returned on time or not
        self.reject_pack = random.uniform(0,1)
        
        # determine whether the pack was returned on time or not
        if self.reject_pack < g.pack_rejection_rate:
        #print(f'Patient {p} pack sent out')
            self.sampled_pack_time = round(random.uniform(3,5),1) # came back late
            self.results_df.at[p.id, 'Return Time Pack'] = \
                                                    self.sampled_pack_time
            # Mark that the pack was returned on time
            self.results_df.at[p.id, 'Pack Rejected'] = 1
            self.results_df.at[p.id, 'Time Pack Reject'] = self.random_normal(\
                                                g.pack_reject_time,g.std_dev)
            
            if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) REJECTED at Pack')
            
            yield self.env.timeout(0)
        
        else:
            
            # pick a random time for how long it took for Pack to be returned
            self.sampled_pack_time = round(random.uniform(0, 3), 1) # came back in time

            # Record how long the pack took to be returned
            self.results_df.at[p.id, 'Return Time Pack'] = \
                                                    self.sampled_pack_time
            # Mark that the pack was returned on time
            self.results_df.at[p.id, 'Pack Rejected'] = 0

            if g.debug_level >= 2:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) ACCEPTED at Pack')

            ##### Now do the Observations #####
            # start up the mdt patient pathway generator
            yield self.env.process(self.obs_pathway(p))

    def obs_pathway(self, patient):
        
        p = patient

        self.results_df.at[p.id, 'Time Obs Visit'] = self.random_normal(\
                                                    g.school_obs_time,g.std_dev)

        # decide whether the obs were completed on time or not
        self.reject_obs = random.uniform(0,1)
        
        # determine whether the obs were returned on time or not
        if self.reject_obs < g.obs_rejection_rate:
        #print(f'Patient {p} obs started')
            # mark that the pack was returned late
            self.results_df.at[p.id, 'Obs Rejected'] = 1
            # record a return time that is after the target
            self.sampled_obs_time = round(random.uniform(4, 6), 1)
            # Record how long the patient took for Obs
            self.results_df.at[p.id, 'Return Time Obs'] = \
                                                        self.sampled_obs_time
            self.results_df.at[p.id, 'Time Obs Reject'] = self.random_normal(\
                                                    g.obs_reject_time,g.std_dev)
            
            if g.debug_level >= 2:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) REJECTED at Obs')

            yield self.env.timeout(0)

        else:
            # pick a random time for how long it took for Obs to be returned
            self.sampled_obs_time = round(random.uniform(0, 4), 1)

            # Record how long the patient took for Obs
            self.results_df.at[p.id, 'Return Time Obs'] = \
                                                        self.sampled_obs_time

            # Mark that the pack was returned on time
            self.results_df.at[p.id, 'Obs Rejected'] = 0

            if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) ACCEPTED at Obs')
            #print(f'Patient {p} obs completed')

            ##### Now do the MDT #####
            # start up the mdt patient pathway generator
            yield self.env.process(self.mdt_pathway(p))

    def mdt_pathway(self, patient):
        
        p = patient
        #print(f'Patient {p} MDT started')
        start_q_mdt = self.env.now

        self.results_df.at[p.id, 'Time Prep MDT'] = self.random_normal(g.mdt_prep_time,g.std_dev)
        self.results_df.at[p.id, 'Time Meet MDT'] = self.random_normal(g.mdt_meet_time,g.std_dev)
        # add referral to MDT waiting list as has passed obs
        g.number_on_mdt_wl += 1

        if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) waiting for MDT')

        # Record where they patient is on the MDT WL
        self.results_df.at[p.id, "MDT WL Posn"] = \
                                            g.number_on_mdt_wl

        # decide whether the mdt was rejected
        self.reject_mdt = random.uniform(0,1)
        
        # Wait until an MDT resource becomes available
        with self.mdt_res.get(1) as mdt_req: # request an MDT resource
            yield mdt_req

            if g.debug_level >= 1:
                print(f'Week {self.week_number} Patient number {p.id} MDT Resource Allocated')
                print(f'MDT slots available: {self.mdt_res.level} (of intended {g.mdt_resource})')
                print(f'Week {self.week_number} Patient number {p.id} being put through MDT')

            #print(f'Resource in use: {mdt_req}')
            # take patient off the MDT waiting list once MDT has taken place
            g.number_on_mdt_wl -= 1

            if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) put through MDT')

            end_q_mdt = self.env.now
            # pick a random time from 0-1 weeks for how long it took for MDT
            self.sampled_mdt_time = round(random.uniform(0,1),1)

            # Calculate how long the patient waited to have MDT
            self.q_time_mdt = end_q_mdt - start_q_mdt

            # Record how long the patient waited for MDT
            self.results_df.at[p.id, 'Q Time MDT'] = (self.q_time_mdt)
            # Record how long the patient took to be MDT'd
            self.results_df.at[p.id, 'Time to MDT'] = self.sampled_mdt_time
            # Record total time it took to MDT patient
            self.results_df.at[p.id, 'Total MDT Time'] = (self.sampled_mdt_time
                                                        +(end_q_mdt -
                                                        start_q_mdt))
            if self.reject_mdt <= g.mdt_rejection_rate:
                self.results_df.at[p.id, 'MDT Rejected'] = 1

                self.results_df.at[p.id, 'MDT Time Reject'] = self.random_normal(g.mdt_reject_time,g.std_dev)

                if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) REJECTED at MDT')

                # release the MDT resource
                yield self.env.timeout(0)
            else:
                self.results_df.at[p.id, 'MDT Rejected'] = 0
                if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) ACCEPTED at MDT')
                # release the MDT resource
                yield self.env.timeout(0)

            #print(f'Patient {p} MDT completed')

                ##### Now do the Assessment #####
                # start up the asst patient pathway generator
                yield self.env.process(self.asst_pathway(p,self.week_number))
    
    def asst_pathway(self, patient, week_number):
        
        # wait until prefill has completed before setting the simulation running
        while g.active_patients <= g.triage_waiting_list+g.asst_waiting_list:

            pass
                           
        # decide whether the assessment was rejected
        self.reject_asst = random.uniform(0,1)

        p = patient

        self.week_number = week_number

        # if the patient comes from the referral generator or Triage WL add them to the asst WL here
        if p.patient_source == 'Referral Gen' or p.patient_source == 'Triage WL':
            g.number_on_asst_wl += 1

        if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} added to Asst WL from {p.patient_source}, current Asst WL:{g.number_on_asst_wl}')

        p.week_added = self.week_number

        #print(f'Patient {p} assessment started')
        start_q_asst = self.env.now

        if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) waiting for Asst')

        # Record where they patient is on the Asst WL
        self.results_df.at[p.id, "Asst WL Posn"] = \
                                                    g.number_on_asst_wl
        # Wait until an Assessment resource becomes available
        with self.asst_res.get(1) as asst_req:
            yield asst_req

            if g.debug_level >= 1:
                print(f'Week {self.week_number} Patient number {p.id} Asst Resource Allocated')
                print(f'Assessment slots available: {self.asst_res.level} (of intended {g.asst_resource})')
                print(f'Week {self.week_number} Patient number {p.id} being put through Asst')

            #print(f'Resource in use: {asst_req}')
            # take patient off the Asst waiting list once Asst starts
            g.number_on_asst_wl -= 1

            if g.debug_level >= 1:
                print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added})put through Asst')
                print(f'Patient {p.id} added in week {p.week_added} from {p.patient_source}, Asst WL now:{g.number_on_asst_wl}')

            end_q_asst = self.env.now

            # pick a random time from 1-4 for how long it took to Assess
            self.sampled_asst_time = round(random.uniform(0,4),1)

            # Calculate how long the patient waited to be Assessed
            if g.prefill == True:
                self.q_time_asst = (end_q_asst - start_q_asst) + g.triage_average_wait
            else:
                self.q_time_asst = end_q_asst - start_q_asst 

            # Record how long the patient waited to be Assessed
            self.results_df.at[p.id, 'Q Time Asst'] = \
                                                        (self.q_time_asst)
            # Record how long the patient took to be Triage
            self.results_df.at[p.id, 'Time to Asst'] = \
                    self.sampled_asst_time
            self.results_df.at[p.id,'Asst Mins Clin'] = \
                    self.random_normal(g.asst_clin_time,g.std_dev)
            self.results_df.at[p.id,'Asst Mins Admin'] = \
                    self.random_normal(g.asst_admin_time,g.std_dev)
            # Record total time it took to triage patient
            self.results_df.at[p.id, 'Total Asst Time'] = \
                                                        (self.sampled_asst_time
                                                        +(end_q_asst -
                                                        start_q_asst))

            # Determine whether patient was rejected following assessment
            if self.reject_asst <= g.asst_rejection_rate:

                self.results_df.at[p.id, 'Asst Rejected'] = 1
                self.results_df.at[p.id,'Diag Rejected Time'] = self.random_normal(g.diag_time_disch,g.std_dev)

                if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) REJECTED at Asst')

                # release the resource once the Assessment is completed
                yield self.env.timeout(0)

            else:
                self.results_df.at[p.id, 'Asst Rejected'] = 0
                self.results_df.at[p.id, 'Diag Accepted Time'] = self.random_normal(g.diag_time_accept,g.std_dev)

                if g.debug_level >= 1:
                    print(f'Week {self.env.now}: Patient number {p.id} (added week {p.week_added}) ACCEPTED at Asst at week {self.env.now}')

                # release the resource once the Assessment is completed
                yield self.env.timeout(0)

            #yield self.env.timeout(0)

            # reset referral counter ready for next batch
            self.referral_counter = 0

            #return self.results_df
    
    # This method calculates results over each single run
    def calculate_run_results(self):
        # Take the mean of the queuing times and the maximum waiting lists
        self.mean_q_time_triage = self.results_df["Q Time Triage"].mean()
        self.max_triage_wl = g.number_on_triage_wl#self.results_df["Triage WL Posn"].max()
        self.mean_q_time_mdt = self.results_df["Q Time MDT"].mean()
        self.max_mdt_wl = g.number_on_mdt_wl #self.results_df["MDT WL Posn"].max()
        self.mean_q_time_asst = self.results_df["Q Time Asst"].mean()
        self.max_asst_wl = g.number_on_asst_wl#self.results_df["Asst WL Posn"].max()
        
        # reset waiting lists and other counters ready for next run
        g.number_on_triage_wl = 0
        g.number_on_mdt_wl = 0
        g.number_on_asst_wl = 0
        g.active_patients = 0

    # The run method starts up the DES entity generators, runs the simulation,
    # and in turns calls anything we need to generate results for the run
    def run(self, print_run_results=True):

        # Start up the governor function
        self.env.process(self.governor_function())

        # Run the model for the duration specified in g class
        self.env.run() # until=g.sim_duration

        # Now the simulation run has finished, call the method that calculates
        # run results
        self.calculate_run_results()

        # Print the run number with the patient-level results from this run of
        # the model
        if print_run_results:
            print(g.weekly_wl_posn)
            print (f"Run Number {self.run_number}")
            print (self.results_df)

# Class representing a Trial for our simulation - a batch of simulation runs.
class Trial:
    # The constructor sets up a pandas dataframe that will store the key
    # results from each run against run number, with run number as the index.
    def  __init__(self):
        self.df_trial_results = pd.DataFrame()
        self.df_trial_results["Run Number"] = [0]
        self.df_trial_results["Mean Q Time Triage"] = [0.0]
        self.df_trial_results["Max Triage WL"] = [0]
        self.df_trial_results["Mean Q Time MDT"] = [0.0]
        self.df_trial_results["Max MDT WL"] = [0]
        self.df_trial_results["Mean Q Time Asst"] = [0.0]
        self.df_trial_results["Max Asst WL"] = [0]
        self.df_trial_results.set_index("Run Number", inplace=True)

        self.weekly_wl_dfs = []

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
        # completed, we grab out the stored run results and store it against
        # the run number in the trial results dataframe
        for run in range(g.number_of_runs):
            my_model = Model(run)
            my_model.run(print_run_results=False)

            self.df_trial_results.loc[run] =  [
                my_model.mean_q_time_triage,
                my_model.max_triage_wl,
                my_model.mean_q_time_mdt,
                my_model.max_mdt_wl,
                my_model.mean_q_time_asst,
                my_model.max_asst_wl,
                ]

            my_model.df_weekly_stats = pd.DataFrame(my_model.df_weekly_stats)

            my_model.df_weekly_stats['Run'] = run

            self.weekly_wl_dfs.append(my_model.df_weekly_stats)
                   
        # Once the trial (i.e. all runs) has completed, print the final results
        return self.df_trial_results, pd.concat(self.weekly_wl_dfs)
    
# my_trial = Trial()
# pd.set_option('display.max_rows', 1000)
# # Call the run_trial method of our Trial class object

# df_trial_results, df_weekly_stats = my_trial.run_trial()

# df_trial_results, df_weekly_stats
