import simpy
import random
import pandas as pd

# Class to store global parameter values
class g:
    # Inter-arrival times
    patient_inter = 3
    call_inter = 10

    # Activity times
    mean_reg_time = 2
    mean_gp_time = 8
    mean_book_test_time = 4
    mean_call_time = 4

    # Resource numbers
    number_of_receptionists = 1 # Base = 1
    number_of_gps = 2 # Base = 2

    # Branch probabilities
    prob_book_test = 0.25

    # Simulation meta parameters
    sim_duration = 480
    number_of_runs = 100

# Class representing patients coming in to the GP surgery
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_reg = 0
        self.q_time_gp = 0
        self.q_time_book_test = 0

# Class representing callers phoning the GP surgery
class Caller:
    def __init__(self, c_id):
        self.id = c_id
        self.q_time_call = 0

# Class representing our model of the GP surgery
class Model:
    # Constructor
    def __init__(self, run_number):
        # Set up SimPy environment
        self.env = simpy.Environment()

        # Set up counters to use as entity IDs
        self.patient_counter = 0
        self.caller_counter = 0

        # Set up resources
        self.receptionist = simpy.Resource(
            self.env, capacity=g.number_of_receptionists
        )
        self.gp = simpy.Resource(
            self.env, capacity=g.number_of_gps
        )

        # Set run number from value passed in
        self.run_number = run_number

        # Set up DataFrame to store patient-level results
        self.patient_results_df = pd.DataFrame()
        self.patient_results_df["Patient ID"] = [1]
        self.patient_results_df["Q Time Reg"] = [0.0]
        self.patient_results_df["Q Time GP"] = [0.0]
        self.patient_results_df["Q Time Book Test"] = [0.0]
        self.patient_results_df.set_index("Patient ID", inplace=True)

        # Set up DataFrame to store caller-level results
        self.caller_results_df = pd.DataFrame()
        self.caller_results_df["Caller ID"] = [1]
        self.caller_results_df["Q Time Call"] = [0.0]
        self.caller_results_df.set_index("Caller ID", inplace=True)

        # Set up attributes that will store mean queuing times across the run
        self.mean_q_time_reg = 0
        self.mean_q_time_gp = 0
        self.mean_q_time_book_test = 0
        self.mean_q_time_call = 0

    # Generator function that represents the DES generator for patient arrivals
    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1

            p = Patient(self.patient_counter)

            self.env.process(self.attend_gp_surgery(p))

            sampled_inter = random.expovariate(1.0 / g.patient_inter)

            yield self.env.timeout(sampled_inter)

    # Generator function that represents the DES generator for caller arrivals
    def generator_callers(self):
        while True:
            self.caller_counter += 1

            c = Caller(self.caller_counter)

            self.env.process(self.call_gp_surgery(c))

            sampled_inter = random.expovariate(1.0 / g.call_inter)

            yield self.env.timeout(sampled_inter)

    # Generator function representing pathway for patients attending the GP
    # surgery to see a GP
    def attend_gp_surgery(self, patient):
        # Registration activity
        start_q_reg = self.env.now

        with self.receptionist.request() as req:
            yield req

            end_q_reg = self.env.now

            patient.q_time_reg = end_q_reg - start_q_reg

            self.patient_results_df.at[patient.id, "Q Time Reg"] = (
                patient.q_time_reg
            )

            sampled_reg_time = random.expovariate(
                1.0 / g.mean_reg_time
            )

            yield self.env.timeout(sampled_reg_time)

        # GP Consultation activity
        start_q_gp = self.env.now

        with self.gp.request() as req:
            yield req

            end_q_gp = self.env.now

            patient.q_time_gp = end_q_gp - start_q_gp

            self.patient_results_df.at[patient.id, "Q Time GP"] = (
                patient.q_time_gp
            )

            sampled_gp_time = random.expovariate(
                1.0 / g.mean_gp_time
            )

            yield self.env.timeout(sampled_gp_time)

        # Branching path check to see if patient needs to book a test
        if random.uniform(0,1) < g.prob_book_test:
            # Book test activity
            start_q_book_test = self.env.now

            with self.receptionist.request() as req:
                yield req

                end_q_book_test = self.env.now

                patient.q_time_book_test = end_q_book_test - start_q_book_test

                self.patient_results_df.at[patient.id, "Q Time Book Test"] = (
                    patient.q_time_book_test
                )

                sampled_book_test_time = random.expovariate(
                    1.0 / g.mean_book_test_time
                )

                yield self.env.timeout(sampled_book_test_time)

    # Generator function representing callers phoning the GP surgery
    def call_gp_surgery(self, caller):
        # Answering call activity
        start_q_call = self.env.now

        with self.receptionist.request() as req:
            yield req

            end_q_call = self.env.now

            caller.q_time_call = end_q_call - start_q_call

            self.caller_results_df.at[caller.id, "Q Time Call"] = (
                caller.q_time_call
            )

            sampled_call_time = random.expovariate(
                1.0 / g.mean_call_time
            )

            yield self.env.timeout(sampled_call_time)

    # Method to calculate and store results over the run
    def calculate_run_results(self):
        self.mean_q_time_reg = self.patient_results_df["Q Time Reg"].mean()
        self.mean_q_time_gp = self.patient_results_df["Q Time GP"].mean()
        self.mean_q_time_book_test = (
            self.patient_results_df["Q Time Book Test"].mean()
        )

        self.mean_q_time_call = self.caller_results_df["Q Time Call"].mean()

    # Method to run a single run of the simulation
    def run(self):
        # Start up DES generators
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.generator_callers())

        # Run for the duration specified in g class
        self.env.run(until=g.sim_duration)

        # Calculate results over the run
        self.calculate_run_results()

        # Print patient and caller level results for this run
        print (f"Run Number {self.run_number}")
        print ("PATIENT RESULTS")
        print (self.patient_results_df)
        print ()
        print ("CALLER RESULTS")
        print (self.caller_results_df)
        print ()

# Class representing a trial for our simulation
class Trial:
    # Constructor
    def __init__(self):
        self.df_trial_results = pd.DataFrame()
        self.df_trial_results["Run Number"] = [0]
        self.df_trial_results["Mean Q Time Reg"] = [0.0]
        self.df_trial_results["Mean Q Time GP"] = [0.0]
        self.df_trial_results["Mean Q Time Book Test"] = [0.0]
        self.df_trial_results["Mean Q Time Call"] = [0.0]
        self.df_trial_results.set_index("Run Number", inplace=True)

    # Method to calculate and store means across runs in the trial
    def calculate_means_over_trial(self):
        self.mean_q_time_reg_trial = (
            self.df_trial_results["Mean Q Time Reg"].mean()
        )
        self.mean_q_time_gp_trial = (
            self.df_trial_results["Mean Q Time GP"].mean()
        )
        self.mean_q_time_book_test_trial = (
            self.df_trial_results["Mean Q Time Book Test"].mean()
        )
        self.mean_q_time_call_trial = (
            self.df_trial_results["Mean Q Time Call"].mean()
        )

    # Method to print trial results, including averages across runs
    def print_trial_results(self):
        print ("Trial Results")
        # The .to_string() below converts the dataframe to a string to print
        # all of it (rather than just the top and bottom 5 entries)
        print (self.df_trial_results.to_string())

        print (f"Mean Q Reg : {self.mean_q_time_reg_trial:.1f} minutes")
        print (f"Mean Q GP : {self.mean_q_time_gp_trial:.1f} minutes")
        print (f"Mean Q Book Test : {self.mean_q_time_book_test_trial:.1f}",
               "minutes")
        print (f"Mean Q Call : {self.mean_q_time_call_trial:.1f} minutes")

    # Method to run trial
    def run_trial(self):
        for run in range(g.number_of_runs):
            my_model = Model(run)
            my_model.run()

            self.df_trial_results.loc[run] = [my_model.mean_q_time_reg,
                                              my_model.mean_q_time_gp,
                                              my_model.mean_q_time_book_test,
                                              my_model.mean_q_time_call]
            
        self.calculate_means_over_trial()
        self.print_trial_results()

# Create new instance of Trial and run it
my_trial = Trial()
my_trial.run_trial()

