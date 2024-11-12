import simpy
import random
import pandas as pd
import plotly.express as px # For plotting
import numpy as np # For adding some conditional columns to our dataframes

class g:
    patient_inter = 5
    mean_n_consult_time = 25
    number_of_nurses = 10
    sim_duration = 720
    number_of_runs = 5

# Class representing patients coming in to the clinic.
# Here, patients have two attributes that they carry with them -
# their ID, and the amount of time they spent queuing for the nurse.
# The ID is passed in when a new patient is created.
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_nurse = 0
        self.arrival_time = 0  ## NEW

# Class representing our model of the clinic.
class Model:
    # Constructor to set up the model for a run.  We pass in a run number when
    # we create a new model.
    def __init__(self, run_number):
        # Create a SimPy environment in which everything will live
        self.env = simpy.Environment()

        # Create a patient counter (which we'll use as a patient ID)
        self.patient_counter = 0

        # Create a SimPy resource to represent a nurse, that will live in the
        # environment created above.  The number of this resource we have is
        # specified by the capacity, and we grab this value from our g class.
        self.nurse = simpy.Resource(self.env, capacity=g.number_of_nurses)

        # Store the passed in run number
        self.run_number = run_number

        # Create a new Pandas DataFrame that will store some results against
        # the patient ID (which we'll use as the index).
        self.results_df = pd.DataFrame()
        self.results_df["Patient ID"] = [1]
        self.results_df["Arrival Time"] = [0.0] ## NEW
        self.results_df["Time Seen"] = [0.0] ## NEW
        self.results_df["Q Time Nurse"] = [0.0]
        self.results_df["Time with Nurse"] = [0.0]
        self.results_df["Time Departed"] = [0.0] ## NEW
        self.results_df["Arrived Within 60 Minutes of Closing Time"] = [False] ## NEW
        self.results_df.set_index("Patient ID", inplace=True)

        # Create an attribute to store the mean queuing time for the nurse
        # across this run of the model
        self.mean_q_time_nurse = 0

    # A generator function that represents the DES generator for patient
    # arrivals
    def generator_patient_arrivals(self):
        # We use an infinite loop here to keep doing this indefinitely whilst
        # the simulation runs
        while True:
            # Increment the patient counter by 1 (this means our first patient
            # will have an ID of 1)
            self.patient_counter += 1

            # Create a new patient - an instance of the Patient Class we
            # defined above.  Remember, we pass in the ID when creating a
            # patient - so here we pass the patient counter to use as the ID.
            p = Patient(self.patient_counter)

            # Tell SimPy to start up the attend_clinic generator function with
            # this patient (the generator function that will model the
            # patient's journey through the system)
            self.env.process(self.attend_clinic(p))

            # Randomly sample the time to the next patient arriving.  Here, we
            # sample from an exponential distribution (common for inter-arrival
            # times), and pass in a lambda value of 1 / mean.  The mean
            # inter-arrival time is stored in the g class.
            sampled_inter = random.expovariate(1.0 / g.patient_inter)

            # Freeze this instance of this function in place until the
            # inter-arrival time we sampled above has elapsed.
            yield self.env.timeout(sampled_inter)

    # A generator function that represents the pathway for a patient going
    # through the clinic.
    def attend_clinic(self, patient):
        # Record the time the patient started queuing for a nurse
        start_q_nurse = self.env.now
        patient.arrival_time = start_q_nurse
        ################
        ## NEW         #
        ################
        # By recording the arrival time in the dataframe before
        # the patient starts waiting for the nurse, we ensure we
        # have a row for every patient in the simulation, not just
        # those who do go on to be seen by the nurse
        self.results_df.at[patient.id, "Arrival Time"] = (
                patient.arrival_time)

        self.results_df.at[patient.id, "Arrived Within 60 Minutes of Closing Time"] = np.where(
                g.sim_duration - patient.arrival_time <60, True, False
                )
        ################
        # END NEW      #
        ################

        # Request a nurse resource, and do all of the following block of code with
        # that nurse resource held in place (and therefore not usable by another patient)
        with self.nurse.request() as req:
            # Freeze the function until the request for a nurse can be met.
            # The patient is currently queuing.
            yield req

            # When we get to this bit of code, control has been passed back to the generator
            # function, and therefore the request for a nurse has been met.
            # We now have the nurse, and have stopped queuing, so we can record the current time
            # as the time we finished queuing.
            end_q_nurse = self.env.now

            # Calculate the time this patient was queuing for the nurse, and
            # record it in the patient's attribute for this.
            patient.q_time_nurse = end_q_nurse - start_q_nurse

            # Now we'll randomly sample the time this patient with the nurse.
            # Here, we use an Exponential distribution for simplicity
            # As with sampling the inter-arrival times, we grab the mean from the g class,
            # and pass in 1 / mean as the lambda value.
            sampled_nurse_act_time = random.expovariate(1.0 /
                                                        g.mean_n_consult_time)

            # Here we'll store the queuing time for the nurse and the sampled
            # time to spend with the nurse in the results DataFrame against the
            # ID for this patient.
            self.results_df.at[patient.id, "Q Time Nurse"] = (
                patient.q_time_nurse)
            self.results_df.at[patient.id, "Time with Nurse"] = (
                sampled_nurse_act_time)

            #########################
            ## NEW                  #
            #########################
            self.results_df.at[patient.id, "Time Seen"] = (
                patient.q_time_nurse + patient.arrival_time)
            self.results_df.at[patient.id, "Time Departed"] = (
                patient.q_time_nurse + patient.arrival_time + sampled_nurse_act_time)
            #########################
            # END NEW               #
            #########################

            # Freeze this function in place for the activity time we sampled
            # above.  This is the patient spending time with the nurse.
            yield self.env.timeout(sampled_nurse_act_time)

            # When the time above elapses, the generator function will return
            # here.  As there's nothing more that we've written, the function
            # will simply end.  This is a sink.  We could choose to add
            # something here if we wanted to record something - e.g. a counter
            # for number of patients that left, recording something about the
            # patients that left at a particular sink etc.

    def calculate_run_results(self):
        # Take the mean of the queuing times for the nurse across patients in
        # this run of the model.
        self.mean_q_time_nurse = self.results_df["Q Time Nurse"].mean()

    # The run method starts up the DES entity generators, runs the simulation,
    # and in turns calls anything we need to generate results for the run
    def run(self):
        # Start up our DES entity generators that create new patients.
        # We've only got one in this model, but we'd need
        # to do this for each one if we had multiple generators.
        self.env.process(self.generator_patient_arrivals())

        # Run the model for the duration specified in g class
        self.env.run(until=g.sim_duration)

        # Now the simulation run has finished, call the method that calculates
        # run results
        self.calculate_run_results()

        return self.results_df

# Class representing a Trial for our simulation - a batch of simulation runs.
class Trial:
    # The constructor sets up a pandas dataframe that will store the key
    # results from each run (just the mean queuing time for the nurse here)
    # against run number, with run number as the index.
    def  __init__(self):
        self.df_trial_results = pd.DataFrame()
        self.df_trial_results["Run Number"] = [0]
        self.df_trial_results["Mean Q Time Nurse"] = [0.0]
        self.df_trial_results.set_index("Run Number", inplace=True)

    # Method to run a trial
    def run_trial(self):
        # Run the simulation for the number of runs specified in g class.
        # For each run, we create a new instance of the Model class and call its
        # run method.  Once the run has completed, we grab out the stored run results
        # (just mean queuing time here) and store it against the run number
        # in the trial results dataframe.

        run_result_dfs = []

        for run in range(g.number_of_runs):
            my_model = Model(run)

            model_results_df = my_model.run()
            # Let's add 1 to the run figure in our dataframes so we don't end up
            # with run 0, which doens't look very polished for stakeholders
            model_results_df["Run"] = run + 1
            run_result_dfs.append(model_results_df)

            self.df_trial_results.loc[run + 1] = [my_model.mean_q_time_nurse]

        # Once the trial (ie all runs) has completed, return the final results
        return self.df_trial_results, pd.concat(run_result_dfs)

mean_results_df, detailed_run_results_df = Trial().run_trial()

print("The median wait to see a nurse across all runs was " +
        f"{mean_results_df['Mean Q Time Nurse'].median():.1f} minutes")

print("Out of all runs, the run with the shortest average wait had an average wait of " +
        f"{mean_results_df['Mean Q Time Nurse'].min():.1f} minutes")

print("Out of all runs, the run with the longest average wait had an average wait of " +
        f"{mean_results_df['Mean Q Time Nurse'].max():.1f} minutes")

print(f"{len(mean_results_df[mean_results_df['Mean Q Time Nurse'] > 60])} runs of" +
        f" {g.number_of_runs} had an average wait that exceeded the target of 60 minutes")

print("Variation Across Runs in Mean Queue Time for Nurse")

fig1 = px.box(mean_results_df,
        x="Mean Q Time Nurse",
        points="all",
        range_x=[0, max(mean_results_df["Mean Q Time Nurse"])*1.1],
        title="Mean Queueing Time for Nurse Per Run"
        )

fig1.show()

print("Performance Against the 60 Minute Wait Target")

detailed_run_results_df["Wait Target Met"] = np.where(
    detailed_run_results_df["Q Time Nurse"] < 60, "Seen - Wait Target Met",
    np.where(detailed_run_results_df["Arrived Within 60 Minutes of Closing Time"] == True,
                "Arrived Within 60 Minutes of Closing Time - Not Seen",
    np.where(detailed_run_results_df["Q Time Nurse"].isna(), "Not Seen Before Clinic Close",
                "Seen - Wait Target Not Met")
        )
    )

print("#### Percentage of Arrivals meeting wait target")

in_time_arrivals_only = (
    detailed_run_results_df[detailed_run_results_df["Arrived Within 60 Minutes of Closing Time"] == False]
    )

target_met_summary_df = pd.DataFrame(
    in_time_arrivals_only
    .groupby("Run")
    .value_counts(["Wait Target Met"])
).reset_index(drop=False).pivot(index="Run", columns="Wait Target Met", values="count").fillna(0)

print(target_met_summary_df)

target_met_summary_df['Target Wait Met'] = (
    target_met_summary_df["Seen - Wait Target Met"] /
    target_met_summary_df.sum(axis=1)
    )*100

fig2 = px.bar(
    target_met_summary_df.reset_index(drop=False),
    x="Run",
    y="Target Wait Met",
    range_y=[0,100],
    title="Percentage of Patients Meeting Waiting Time Target"
)

fig2.update_layout(yaxis_title="Target Wait Met (% of Patients)")

fig2.show()

print("""
            In this plot, each row represents one run of the simulation.

            Each dot is a patient, with the horizontal axis representing the passage of time.

            The patients are placed at the time they arrived at the clinic.

            The colour indicates whether, depending on their arrival time, a patient was
            - seen within the target time
            - seen before the clinic closed, but not within the target wait time
            - not seen before the clinic closed for the day

            The orange line indicates 60 minutes before the closing time of the clinic, indicating patients who have arrived
            at a point where it may be reasonable to expect they will not be seen before closing time.

            The red line indicates the time of day the clinic closes at.

            This plot can be useful to give an indication of
            - how soon in the day the clinic becomes overloaded
            - how much variation there is in the clinic becoming overloaded on different days as patient arrival patterns naturally vary (though note all days will have broadly similar total numbers of simulated patients).
            """)

detailed_arrival_fig = px.scatter(
        detailed_run_results_df,
        x="Arrival Time",
        y="Run",
        color="Wait Target Met",
        title="Patient Arrivals per Run - Outcomes"
    )

detailed_arrival_fig.add_vline(g.sim_duration-60, line_color="orange")

detailed_arrival_fig.add_vline(g.sim_duration, line_color="red")

detailed_arrival_fig.show()

print("Run Averages")
print(mean_results_df)


print("Wait Targets Met - Per Run")
print(target_met_summary_df)

print("Patient-Level Simulation Outputs")
print(detailed_run_results_df)
