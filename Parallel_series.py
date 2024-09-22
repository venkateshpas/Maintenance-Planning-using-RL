from heuristics import *
from functions import *


#Approach 1: Risk Heuristic

risk1 = np.arange(0.1,1,0.1) #Risk factor for inspecting a component
risk2 = np.arange(0.2,1,0.1) #Risk factor for repairing a component: Since it is a major cost start repairing if more than 0.5 only.
#Also inspect between risk1 and risk2 for major damage

results_risk = []

for r1 in risk1:
    for r2 in risk2:
            average_episodes, average_ep_cost, total_avg_cost = one_full_iteration(int(1e3),r1,r2,'risk')
            results_risk.append((r1,r2, average_episodes, average_ep_cost, total_avg_cost))

plot_risk_results(results_risk)
write_to_csv('csv_results_risk.csv',results_risk)

#Approach 2: Age Heuristic

age1 = np.arange(0,50,5)
age2 = np.arange(0,50,5)
#Similar logic as risk too here but for age.

results_age = []
# for a1 in age1:
#     for a2 in age2:
#             average_episodes, average_ep_cost, total_avg_cost = one_full_iteration(int(1e3),a1,a2,'age')
#             results_age.append((a1,a2, average_episodes, average_ep_cost, total_avg_cost))


# plot_age_results(results_age)
# write_to_csv('csv_results_age.csv',results_age)