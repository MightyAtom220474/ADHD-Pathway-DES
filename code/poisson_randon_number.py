import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample


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
    mean_referrals = 100 # average number of referrals per week
    base_waiting_list = 2741 # current number of patients on waiting list
    mean_triage_time = 14 # average wait for triage
    mean_asst_time = 13 # average wait for assessment
    mean_school_time = 12 # average time taken for observations and info gather
    mean_diag_time = 14 # average time taken to diagnose
    mean_titrate_time = 15 # average time taken to titrate meds
    #prob_req_repeat_titr = 0.4 # probability that they will need re-titrating
    number_of_clinicians = 5
    max_caseload = 10 # maximum number of CYP a clinician can carry
    caseload_slots = number_of_clinicians * max_caseload 
    #prob_req_group_t = 0.4 # probability that they will require group therapy
    #mean_oto_sessions = 10 # average number of 1:1 sessions required
    #mean_group_sessions = 10 # average number of group sessions required
    sim_duration = 520 # number of weeks to run the simulation for
    number_of_runs = 10

sampled_referrals_poisson = (np.random.poisson(g.mean_referrals,
                                                  g.sim_duration))

sampled_referrals = random.choice(sampled_referrals_poisson)

sampled_referrals

