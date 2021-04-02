import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


def simulate_preferences(num_households, num_lots, num_choice, simulation_type):
    """
    Simulation function
    """
    household_preferences = np.zeros((num_households, num_lots))
    for i in range(num_households):
        if simulation_type == "easy":
            lots = np.random.choice(num_lots, size=num_choice*2, replace=False)
        # Note: assuming num_lots > 3*num_choice
        elif simulation_type == "medium":
            num_wanted_lots = 10
            p = np.ones(num_lots)
            p[num_wanted_lots:] = 1/(num_lots*2)
            p[0:num_wanted_lots] = (1 - np.sum(p[num_wanted_lots:]))/num_wanted_lots
            lots = np.random.choice(num_lots, size=num_choice * 2, replace=False, p=p)
            print(p)
        else:
            lots = np.random.choice(num_lots//3, size=num_choice * 2, replace=False)

        pref_lots = lots[:num_choice]
        no_go_lots = lots[num_choice:]
        household_preferences[i,pref_lots] = 1
        household_preferences[i, no_go_lots] = -1
    return household_preferences


def build_weight_matrix(num_choice, household_preferences, lots_properties=None,
                      household_profiles=None,
                      weight_for_pref_lot=10):
    """
    Build the weight matrix for lots assignment
    :param num_choice: number of lots a household can specify they prefer/reject
    :param household_preferences: matrix M, where M[i,j] = 1 if household i wants lot j,
                                  -1 if it's a no-go, 0 otherwise
    :param lots_properties: properties of rows (optional, can be None)
    :param household_profiles: profiles of households with respect to lots (optional can be None)
    :param weight_for_pref_lot: the weight to assign to desired lots
    :return: weight matrix
    """
    weight_matrix = np.zeros(household_preferences.shape)
    num_lots = weight_matrix.shape[1]
    # assign desired lots with a large weight
    weight_matrix[household_preferences==1] = weight_for_pref_lot
    # assign undesired lots with a small weight > 0
    weight_matrix[household_preferences == 0] = (num_lots - num_choice*weight_for_pref_lot)/(num_lots-num_choice*2)
    # Note: no-go lots are assigned with -1
    weight_matrix[household_preferences == -1] = -1

    if household_profiles is not None and lots_properties is not None:
        # modify weight matrix according to additional priors
        pass
    return weight_matrix

# Parameters
np.random.seed = 2
num_households = 37
num_lots = 37
num_choice = 6
simulate = True
simulation_type = "medium"
weight_for_pref_lot = 10

if simulate:
    print("Simulating household preferences, difficulty degree: {}".format(simulation_type))
    household_preferences = simulate_preferences(num_households, num_lots, num_choice, simulation_type)
    lots_properties = None
    household_profiles = None
else:
    print("Reading household preferences")
    # read household preferences
    # read lots properties
    # read household profiles
    household_preferences = None
    lots_properties = None
    household_profiles = None

# Build preferences matrix
print("Building weight matrix")
weight_matrix = build_weight_matrix(num_choice, household_preferences, lots_properties,
                                    household_profiles, weight_for_pref_lot)

# Assign using the Hungarian Algorithm
print("Starting assignment (Hungarian Maximum Matching Algorithm) with weight matrix")
household_ind, lot_ind = linear_sum_assignment(-1*weight_matrix)
print("Assignment completed")

# Evaluate assignment
assignments_scores = household_preferences[household_ind, lot_ind]
num_happy = np.sum(assignments_scores == 1)
num_sad = np.sum(assignments_scores == -1)
num_not_happy = np.sum(assignments_scores == 0)
print(household_preferences[0:2, :])
print(lot_ind[0:2])
print("Evaluating assignment")
print("Number of households which received a lot they wanted: {} ({}%)".format(num_happy, num_happy*1.0/num_households*100))
print("Number of households did not receive a lot they wanted, but the lot is not a NO-GO lot: {} ({}%)".format(num_not_happy, num_not_happy*1.0/num_households*100))
print("Number of households which received a NO-GO lot: {} ({}%)".format(num_sad, num_sad*1.0/num_households*100))

