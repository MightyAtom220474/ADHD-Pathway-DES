import simpy
import random
import pandas as pd

######################################################
class g:
    number_of_receptionists = 1
    number_of_call_handler = 1
    number_of_doctors = 2
    patient_inter_phone = 10
    patient_inter_person = 3
    mean_n_phone_time = 4
    mean_n_registration_time = 2
    mean_n_booking_time = 4
    mean_n_doctor_time = 8
    prob_booking_test = 0.25
    sim_duration = 8 * 60
    number_of_runs = 100

######################################################
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_registration = 0
        self.q_time_phone = 0
        self.q_time_doctor = 0
        self.q_time_booking = 0
        self.q_time_total = 0

######################################################
class Model:
    def __init__(self, run_number):
        self.env = simpy.Environment()

        self.patient_counter = 0

######################################################
        self.receptionist = simpy.Resource(
            self.env, capacity=g.number_of_receptionists
        )
        self.doctor = simpy.Resource(
            self.env, capacity=g.number_of_doctors
        )
        self.call_handler = simpy.Resource(
            self.env, capacity=g.number_of_call_handler
        )

        self.run_number = run_number

######################################################
        self.results_df = pd.DataFrame()
        self.results_df["Patient ID"] = [1]
        self.results_df["Q Time Registration"] = [0.0]
        self.results_df["Time Registration"] = [0.0]
        self.results_df["Q Time Doctor"] = [0.0]
        self.results_df["Time with Doctor"] = [0.0]
        self.results_df["Q Time Booking Test"] = [0.0]
        self.results_df["Time Booking Test"] = [0.0]
        self.results_df["Q Time Phone"] = [0.0]
        self.results_df["Time Phone"] = [0.0]
        self.results_df["Time Total"] = [0.0]
        self.results_df.set_index("Patient ID", inplace=True)

######################################################
        self.mean_q_time_registration = 0
        self.mean_q_time_phone = 0
        self.mean_q_time_doctor = 0
        self.mean_q_time_booking = 0
        self.mean_q_time_total = 0
    
######################################################
    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            p = Patient(self.patient_counter)
            self.env.process(self.attend_clinic(p))
            sampled_inter = random.expovariate(1.0 / g.patient_inter_person)
            yield self.env.timeout(sampled_inter)
    
    def generator_patient_phone(self):
        while True:
            self.patient_counter += 1
            p = Patient(self.patient_counter)
            self.env.process(self.phone_clinic(p))
            sampled_inter = random.expovariate(1.0 / g.patient_inter_phone)
            yield self.env.timeout(sampled_inter)
######################################################
    def attend_clinic(self, patient):
        start_q_registration = self.env.now

        with self.receptionist.request() as req:
            yield req

            end_q_registration = self.env.now

            patient.q_time_registration = end_q_registration - start_q_registration

            sampled_registration_act_time = random.expovariate(
                1.0 / g.mean_n_registration_time
            )

            self.results_df.at[patient.id, "Q Time Registration"] = (
                 patient.q_time_registration
            )
            self.results_df.at[patient.id, "Time Registration"] = (
                 sampled_registration_act_time
            )

            yield self.env.timeout(sampled_registration_act_time)

        start_q_doctor = self.env.now

        with self.doctor.request() as req:
            yield req

            end_q_doctor = self.env.now

            patient.q_time_doctor = end_q_doctor - start_q_doctor

            sampled_doctor_act_time = random.expovariate(
                1.0 / g.mean_n_doctor_time
            )

            self.results_df.at[patient.id, "Q Time Doctor"] = (
                 patient.q_time_doctor
            )
            self.results_df.at[patient.id, "Time with Doctor"] = (
                 sampled_doctor_act_time
            )

            yield self.env.timeout(sampled_doctor_act_time)

        if random.uniform(0,1) < g.prob_booking_test:
            
            start_q_booking = self.env.now
            
            with self.receptionist.request() as req:
                yield req

                end_q_booking = self.env.now

                patient.q_time_booking = end_q_booking - start_q_booking

                sampled_booking_act_time = random.expovariate(
                    1.0 / g.mean_n_booking_time
                )

                self.results_df.at[patient.id, "Q Time Booking Test"] = (
                    patient.q_time_booking
                )
                self.results_df.at[patient.id, "Time Booking Test"] = (
                    sampled_booking_act_time
                )

                yield self.env.timeout(sampled_doctor_act_time)

        end_q_total = self.env.now
        patient.q_time_total = end_q_total - start_q_registration

        self.results_df.at[patient.id, "Time Total"] = (
                patient.q_time_total
        )        

######################################################
    def phone_clinic(self, patient):
        start_q_phone = self.env.now

        with self.call_handler.request() as req:
            yield req

            end_q_phone = self.env.now

            patient.q_time_phone = end_q_phone - start_q_phone

            sampled_phone_act_time = random.expovariate(
                1.0 / g.mean_n_phone_time
            )

            self.results_df.at[patient.id, "Q Time Phone"] = (
                 patient.q_time_phone
            )
            self.results_df.at[patient.id, "Time Phone"] = (
                 sampled_phone_act_time
            )

            yield self.env.timeout(sampled_phone_act_time)
  
        end_q_total = self.env.now
        patient.q_time_total = end_q_total - start_q_phone

        self.results_df.at[patient.id, "Time Total"] = (
                patient.q_time_total
        )    

############################################
    def calculate_run_results(self):
        self.mean_q_time_registration = self.results_df["Q Time Registration"].mean()
        self.mean_q_time_phone = self.results_df["Q Time Phone"].mean()
        self.mean_q_time_doctor = self.results_df["Q Time Doctor"].mean()
        self.mean_q_time_booking = self.results_df["Q Time Booking Test"].mean()
        self.mean_q_time_total = self.results_df["Time Total"].mean()

    def run(self):
        # Start up our DES entity generators that create new patients.  We've
        # only got one in this model, but we'd need to do this for each one if
        # we had multiple generators.
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.generator_patient_phone())

        # Run the model for the duration specified in g class
        self.env.run(until=g.sim_duration)

        # Now the simulation run has finished, call the method that calculates
        # run results
        self.calculate_run_results()

        # Print the run number with the patient-level results from this run of 
        # the model
        print (f"Run Number {self.run_number}")
        print (self.results_df)

class Trial:
    # The constructor sets up a pandas dataframe that will store the key
    # results from each run against run number, with run number as the index.
    def  __init__(self):
        self.df_trial_results = pd.DataFrame()
        self.df_trial_results["Run Number"] = [0]
        self.df_trial_results["Mean Q Time Registration"] = [0.0]
        self.df_trial_results["Mean Q Time Phone"] = [0.0]
        self.df_trial_results["Mean Q Time Doctor"] = [0.0]
        self.df_trial_results["Mean Q Time Booking"] = [0.0]
        self.df_trial_results["Mean Time Total"] = [0.0]
        self.df_trial_results.set_index("Run Number", inplace=True)

    def print_trial_results(self):
        print ("Trial Results")
        print (self.df_trial_results)

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
            
            ##NEW - added mean queue time for doctor to end of list
            self.df_trial_results.loc[run] = [my_model.mean_q_time_registration,
                                              my_model.mean_q_time_phone,
                                              my_model.mean_q_time_doctor,
                                              my_model.mean_q_time_booking,
                                              my_model.mean_q_time_total
                                              ]

        # Once the trial (ie all runs) has completed, print the final results
        self.print_trial_results()

        self.df_trial_results.to_csv('out_call_handler.csv')

# Create an instance of the Trial class
my_trial = Trial()

# Call the run_trial method of our Trial object
my_trial.run_trial()