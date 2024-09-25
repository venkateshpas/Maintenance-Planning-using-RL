import numpy as np
from heuristics import *
import csv
import random

ncomp = 5 # No. of components in the system
nstcomp = 4 # No. of states for each component.. 1: no damage, 2: minor-damage, 3: major-damage, 4: failure
nacomp = 3 # No. of actions that can be taken.. 1: do nothing, 2: repair, 3: inspect



class DeteriorationMatrix():
    def __init__(self,ncomp,nstcomp):
        self.system_matrix = np.zeros((ncomp,nstcomp,nstcomp))
    def component_deterioration_matrix(self, c):
        self.component_matrix = c
        return self.component_matrix
    def system_deterioration_matrix(self,d1,d2,d3,d4,d5):
        self.system_matrix[0] = d1
        self.system_matrix[1] = d2
        self.system_matrix[2] = d3
        self.system_matrix[3] = d4
        self.system_matrix[4] = d5
        return self.system_matrix
    
class ObservationMatrix():
    def __init__(self, ncomp):
        self.ncomp = ncomp   
        self.system_matrix = np.zeros((self.ncomp,4,4))
    def component_observation_matrix(self,p):
        self.component_matrix = np.array([[p,(1-p),0,0],
                                          [(1-p)/2,p,(1-p)/2,0],
                                          [0,1-p,p,0],
                                          [0,0,0,1]])
        return self.component_matrix
    def system_observation_matrix(self,c1,c2,c3,c4,c5):
        self.system_matrix[0] = c1
        self.system_matrix[1] = c2
        self.system_matrix[2] = c3
        self.system_matrix[3] = c4
        self.system_matrix[4] = c5
        return self.system_matrix

class CostMatrix():
    def __init__(self,ncomp):
        self.ncomp = ncomp
        self.system_matrix = np.zeros((ncomp,3))
    def component_cost_matrix(self,c1,c2,c3): # These are the costs for noaction, inspect and repair
        self.cost_matrix = np.array([c1,c2,c3])
        return self.cost_matrix
    def system_cost_matrix(self,c1,c2,c3,c4,c5):
        self.system_matrix[0] = c1
        self.system_matrix[1] = c2
        self.system_matrix[2] = c3
        self.system_matrix[3] = c4
        self.system_matrix[4] = c5
        return self.system_matrix


#Td matrix
deterioration = DeteriorationMatrix(ncomp,nstcomp)
c1_dm = deterioration.component_deterioration_matrix(np.array([[0.82,0.13,0.05,0],[0,0.87,0.09,0.04],[0,0,0.91,0.09],[0,0,0,1]])) # The Natural deterioration matrix for component 1
c2_dm = deterioration.component_deterioration_matrix(np.array([[0.72,0.19,0.09,0],[0,0.78,0.18,0.04],[0,0,0.85,0.15],[0,0,0,1]])) # The Natural deterioration matrix for component 2
c3_dm = deterioration.component_deterioration_matrix(np.array([[0.79,0.17,0.04,0],[0,0.85,0.09,0.06],[0,0,0.91,0.09],[0,0,0,1]])) # The Natural deterioration matrix for component 3
c4_dm = deterioration.component_deterioration_matrix(np.array([[0.8,0.12,0.08,0],[0,0.83,0.12,0.05],[0,0,0.89,0.11],[0,0,0,1]]))  # The Natural deterioration matrix for component 4
c5_dm = deterioration.component_deterioration_matrix(np.array([[0.88,0.12,0,0],[0,0.9,0.1,0],[0,0,0.93,0.07],[0,0,0,1]]))         # The Natural deterioration matrix for component 5

pcomp1 = deterioration.system_deterioration_matrix(c1_dm,c2_dm, c3_dm, c4_dm, c5_dm)


#Omega matrix: Imperfect observations of its true state
observation = ObservationMatrix(ncomp)
c1_om = observation.component_observation_matrix(0.8) #Omege matrix for component1
c2_om = observation.component_observation_matrix(0.85) #Omege matrix for component1
c3_om = observation.component_observation_matrix(0.9) #Omege matrix for component1
c4_om = observation.component_observation_matrix(0.95) #Omege matrix for component1
c5_om = observation.component_observation_matrix(0.8) #Omege matrix for component1


pobs_insp = observation.system_observation_matrix(c1_om, c2_om,c3_om,c4_om,c5_om)
pobs_no_insp = np.array([[1.,0],[1.,0],[1.,0],[0,1.]])

    
cost = CostMatrix(ncomp)
c1_cm = cost.component_cost_matrix(0,-20,-30)
c2_cm = cost.component_cost_matrix(0,-40,-90)
c3_cm = cost.component_cost_matrix(0,-25,-80)
c4_cm = cost.component_cost_matrix(0,-50,-250)
c5_cm = cost.component_cost_matrix(0,-100,-350)

cost_comp_action = cost.system_cost_matrix(c1_cm,c2_cm,c3_cm,c4_cm,c5_cm)

st_a = np.zeros((1,ncomp,nstcomp,1))

def state_action(st,a):
    st_a[:] = st
    for i in range(ncomp):
        if a[0,i] == 2: # So that means, action is repair
            st_a[0,i,:,0] = 0 # Set all elements to zero
            if i == 0: # If the component is 1
                st_a[0,i,0,0] = 1 # Set 1st element to 1
            elif i == 1: # If the component is 2
                st_a[0,i,0,0] = 0.9
                st_a[0,i,1,0] = 1 - 0.9
            elif i == 2: # If the component is 3
                st_a[0,i,0,0] = 0.95
                st_a[0,i,1,0] = 1 - 0.95
            elif i == 3: # If the component is 4
                st_a[0,i,0,0] = 0.85
                st_a[0,i,1,0] = 1 - 0.85
            elif i == 4: # If the component is 5
                st_a[0,i,0,0] = 0.8
                st_a[0,i,1,0] = 1 - 0.8
    return st_a

def system_state(state_a, pcomp1, action):
    ncomp = len(state_a[0])
    o = np.zeros(ncomp,dtype=int)
    state_next = np.zeros((1,ncomp,nstcomp,1))
    for j in range(ncomp):
        if action[0,j] == 2:
            p1 = np.zeros(nstcomp)
            if j == 0:
                p1[0] = 1
            elif j == 1:
                p1[0] = 0.9
                p1[1] = 1-0.9
            elif j == 2:
                p1[0] = 0.95
                p1[1] = 1-0.95
            elif j == 3:
                p1[0] = 0.85
                p1[1] = 1-0.85
            elif j == 4:
                p1[0] = 0.8
                p1[1] = 1-0.8
        else:
            p1 = (pcomp1[j].T).dot(state_a[0,j,:,0]) # What is this?

        if action[0,j] == 1:
            ob_dist = p1.dot(pobs_insp[j])
            o[j] = np.random.choice(range(0,4), size = None, replace=True, p=ob_dist)
            state_next[0,j,:,0] =  p1 * pobs_insp[j,:,o[j]]/(p1.dot(pobs_insp[j,:,o[j]]))
        else:
            ob_dist = p1.dot(pobs_no_insp)
            o[j] = np.random.choice(range(0,2), size=None, replace=True, p=ob_dist)
            state_next[0,j,:,0] = p1* pobs_no_insp[:,o[j]]/(p1.dot(pobs_no_insp[:,o[j]]))
    return state_next,o


def immediatecost(action,cost_comp_action):
    cost_action = 0
    for j in range(ncomp):
        cost_action += cost_comp_action[j][int(action[0,j])]
    return cost_action

def is_system_failed(state):
    '''
    Basically, this function checks if the system failed.
    There are three links .. link 1: connected by components 1 and 5
    link 2: connected by components 2 and 3
    link 3 : connected by component 4
    Each of these links are in parallel configuration but the components within the link are in series
    Therefore, for a link failure either of the component failing would result in the links failure
    Where as for the system failure, all three links must fail. Hence, using and
    '''
    link_1_failure = (state[0,0,-1,0] == 1) or (state[0,4,-1,0] == 1) 
    link_2_failure = (state[0,1,-1,0] == 1) or (state[0,2,-1,0] == 1)
    link_3_failure = (state[0,3,-1,0] == 1)
    return link_1_failure and link_2_failure and link_3_failure


def one_full_iteration(episodes,param1,param2,heuristic):
    buffer_batch = 50 #Possibly a batch size to process at once?
    buffer_memo_size = 100 #The maximum memory (no. of experiences) that the algorithm can remember
    ep_length = 50 # 50 years ?
    gamma = 0.975 # The discount factor ?

    buffer_state = np.zeros((buffer_memo_size, ncomp, nstcomp, 1)) #For each of the past 100 experiences, this is used to store the state
    buffer_time = np.zeros((buffer_memo_size)) #Maybe to store the time step?
    buffer_action = np.zeros((buffer_memo_size,ncomp),dtype=int) #Actions taken for each of the step for the past 100 experiences, also for each of the component
    buffer_cost = np.zeros(buffer_memo_size) #Stores the immediate cost after an action for each state.

    buffer_terminal_flag = np.zeros(buffer_memo_size) #Maybe to indicate if system failed during the past 100 experiences
    buffer_count = 0 

    total_ep_cost = np.zeros(episodes+1) #Total cost for each episode
    m_average = np.zeros(episodes+1)
    average_ep_cost = []
    average_episodes = [] 
    total_ep_risk = np.zeros(episodes+1) #Total risk for each episode?
    m_average_risk = np.zeros(episodes+1)
    for ep in range(0,episodes+1):
        oo = np.zeros(5) # What is this?
        print(f'Episode number: {ep}')
        state = np.zeros((1,ncomp,nstcomp,1))
        state[0,:,0,0] = 1 # Every component, initial state is no damage .. so the first row for every component is made 1
        
        total_ep_cost[ep] = 0

        age_components = np.zeros((1,ncomp),dtype=int)

        for i in range(0,ep_length):
            if buffer_count == buffer_memo_size:
                buffer_count = 0
            
            action = []
            action = np.zeros((1,ncomp),dtype=int)
            #Write a code that tries to find optimized action .. ? #Here contrary to state_action, we need to take an action based on current state.
            '''
            If current state not good enough (failure), then the action must be repair
            If current state is good enough, then no action
            When to inspect but? Maybe when current state is minor/major?
            '''
            if heuristic == 'risk':
                action = risk_heuristic(ncomp, state,action,param1,param2)
            elif heuristic == 'age':
                action = age_heuristic(ncomp,state,action,param1,param2,age_components)
            
            for j in range(0,ncomp):
                if action[0,j] == 2:
                    age_components[0,j] = 0
                else:
                    age_components[0,j]+=1

            cost_action = immediatecost(action, cost_comp_action)
            state_a = state_action(state,action) #This basically does something to the state based on our action .. either restore it or keep it as it is

            if ep%100 == 0:
                print(action)

            cost = cost_action
            if is_system_failed(state):
                cost -= 2400
            total_ep_cost[ep] += gamma**i*cost

            
            state_dot,oo = system_state(state_a,pcomp1,action)
            buffer_state[buffer_count] = state
            buffer_time[buffer_count] = i
            buffer_action[buffer_count] = action
            buffer_cost[buffer_count] = cost

            if i==ep_length-1:
                flag=1
                buffer_terminal_flag[buffer_count] = flag
            else:
                flag = 0
                buffer_terminal_flag[buffer_count] = flag

            state = state_dot
            buffer_count+=1

        '''
        More code here
        '''
        if ep >= 100:
            average_ep_cost.append(sum(total_ep_cost[ep-100:ep])/100)
            average_episodes.append(ep)
    return average_episodes, average_ep_cost, sum(total_ep_cost)/len(total_ep_cost)



def plot_risk_results(results):
    r1_values = np.array([res[0] for res in results])
    r2_values = np.array([res[1] for res in results])
    r1_vals = sorted(list(set([res[0] for res in results])))
    r2_vals = sorted(list(set([res[1] for res in results])))
    total_avg_costs = np.array([res[4] for res in results])


    r1_grid, r2_grid = np.meshgrid(r1_vals, r2_vals)
    cost_grid = np.zeros_like(r1_grid)
    for i, r1 in enumerate(r1_vals):
        for j, r2 in enumerate(r2_vals):
            cost_grid[j, i] = next((res[4] for res in results if res[0] == r1 and res[1] == r2), np.nan)

    max_cost = np.max(total_avg_costs)
    max_index = np.argmax(total_avg_costs)
    max_r1 = r1_values[max_index]
    max_r2 = r2_values[max_index]

    print(f"The maximum total average cost is {max_cost:.2f} for r1={max_r1:.2f}, r2={max_r2:.2f}")

    fig, ax = plt.subplots(figsize=(14, 14), dpi = 100)

    heatmap = ax.imshow(cost_grid, cmap='Reds_r', origin='lower', 
                        extent=[0, 1, 0, 1])
                        #extent=[min(r1_vals), max(r1_vals), min(r2_vals), max(r2_vals)])

    ax.scatter(max_r1, max_r2, color='green', s=100, label=f'Optimal Cost: {max_cost:.2f}')

    cbar = plt.colorbar(heatmap)
    cbar.set_label('Average cost for each value of risk', fontsize=14, fontfamily='Trebuchet MS')
    ax.set_xlabel('Risk parameter for inspecting minor damage', fontsize=16, fontfamily='Trebuchet MS')
    ax.set_ylabel('Risk parameter for repairing major damage', fontsize=16, fontfamily='Trebuchet MS')
    ax.set_title('Total Avg Cost with Maximum Point Highlighted', fontsize=16, fontfamily='Trebuchet MS')
    ax.legend()
    ax.legend(prop={'size': 12, 'family': 'Trebuchet MS'})
    plt.xticks(fontsize=14, fontfamily='Trebuchet MS')
    plt.yticks(fontsize=14, fontfamily='Trebuchet MS')
    plt.show()

def plot_age_results(results):
    a1_values = np.array([res[0] for res in results])
    a2_values = np.array([res[1] for res in results])
    a1_vals = sorted(list(set([res[0] for res in results])))
    a2_vals = sorted(list(set([res[1] for res in results])))
    total_avg_costs = np.array([res[4] for res in results])


    a1_grid, a2_grid = np.meshgrid(a1_vals, a2_vals)
    cost_grid = np.zeros_like(a1_grid)
    for i, a1 in enumerate(a1_vals):
        for j, a2 in enumerate(a2_vals):
            cost_grid[j, i] = next((res[4] for res in results if res[0] == a1 and res[1] == a2), np.nan)

    max_cost = np.max(total_avg_costs)
    max_index = np.argmax(total_avg_costs)
    max_a1 = a1_values[max_index]
    max_a2 = a2_values[max_index]

    print(f"The maximum total average cost is {max_cost:.2f} for age1={max_a1:.2f}, age2={max_a2:.2f}")

    fig, ax = plt.subplots(figsize=(12, 12),dpi=100)

    heatmap = ax.imshow(cost_grid, cmap='Reds_r', origin='lower', 
                        extent=[0, 50, 0, 50])
                        #extent=[min(a1_vals), max(a1_vals), min(a2_vals), max(a2_vals)])

    ax.scatter(max_a1, max_a2, color='green', s=100, label=f'Optimal Cost: {max_cost:.2f}')

    cbar = plt.colorbar(heatmap)
    cbar.set_label('Average cost for age based replacement', fontsize=16, fontfamily='Trebuchet MS')
    ax.set_xlabel('Age parameter for inspecting minor damage', fontsize=16, fontfamily='Trebuchet MS')
    ax.set_ylabel('Age parameter for repairing major damage', fontsize=16, fontfamily='Trebuchet MS')
    ax.set_title('Total Avg Cost with Maximum Point Highlighted', fontsize=16, fontfamily='Trebuchet MS')
    ax.legend()
    ax.legend(prop={'size': 14, 'family': 'Trebuchet MS'})
    plt.xticks(fontsize=14, fontfamily='Trebuchet MS')
    plt.yticks(fontsize=14, fontfamily='Trebuchet MS')
    plt.show()


def action_plot(episodes,heuristic,results):
    r1_values = np.array([res[0] for res in results])
    r2_values = np.array([res[1] for res in results])
    total_avg_costs = np.array([res[4] for res in results])
    max_index = np.argmax(total_avg_costs)
    param1 = r1_values[max_index]
    param2 = r2_values[max_index]

    buffer_batch = 50 #Possibly a batch size to process at once?
    buffer_memo_size = 100 #The maximum memory (no. of experiences) that the algorithm can remember
    ep_length = 50 # 50 years ?
    gamma = 0.975 # The discount factor ?

    buffer_state = np.zeros((buffer_memo_size, ncomp, nstcomp, 1)) #For each of the past 100 experiences, this is used to store the state
    buffer_time = np.zeros((buffer_memo_size)) #Maybe to store the time step?
    buffer_action = np.zeros((buffer_memo_size,ncomp),dtype=int) #Actions taken for each of the step for the past 100 experiences, also for each of the component
    buffer_cost = np.zeros(buffer_memo_size) #Stores the immediate cost after an action for each state.

    buffer_terminal_flag = np.zeros(buffer_memo_size) #Maybe to indicate if system failed during the past 100 experiences
    buffer_count = 0 

    total_ep_cost = np.zeros(episodes+1) #Total cost for each episode
    
    for ep in range(0,episodes+1):
        oo = np.zeros(5) # What is this?
        state = np.zeros((1,ncomp,nstcomp,1))
        state[0,:,0,0] = 1 # Every component, initial state is no damage .. so the first row for every component is made 1
        actions_in_final_episode = []
        state_in_final_episode = []
        total_ep_cost[ep] = 0

        age_components = np.zeros((1,ncomp),dtype=int)

        for i in range(0,ep_length):
            if buffer_count == buffer_memo_size:
                buffer_count = 0
            
            action = []
            action = np.zeros((1,ncomp),dtype=int)
            #Write a code that tries to find optimized action .. ? #Here contrary to state_action, we need to take an action based on current state.
            '''
            If current state not good enough (failure), then the action must be repair
            If current state is good enough, then no action
            When to inspect but? Maybe when current state is minor/major?
            '''
            if heuristic == 'risk':
                action = risk_heuristic(ncomp, state,action,param1,param2)
            elif heuristic == 'age':
                action = age_heuristic(ncomp,state,action,param1,param2,age_components)
            
            for j in range(0,ncomp):
                if action[0,j] == 2:
                    age_components[0,j] = 0
                else:
                    age_components[0,j]+=1

            cost_action = immediatecost(action, cost_comp_action)
            state_a = state_action(state,action) #This basically does something to the state based on our action .. either restore it or keep it as it is


            cost = cost_action
            if is_system_failed(state):
                cost -= 2400
            total_ep_cost[ep] += gamma**i*cost

            
            state_dot,oo = system_state(state_a,pcomp1,action)
            buffer_state[buffer_count] = state
            buffer_time[buffer_count] = i
            buffer_action[buffer_count] = action
            buffer_cost[buffer_count] = cost

            if i==ep_length-1:
                flag=1
                buffer_terminal_flag[buffer_count] = flag
            else:
                flag = 0
                buffer_terminal_flag[buffer_count] = flag

            state = state_dot
            buffer_count+=1

            if ep == episodes:
                actions_in_final_episode.append(action.copy())
                state_in_final_episode.append(state_dot.copy())


        '''
        More code here
        '''
    actions_ = actions_in_final_episode
    state_beliefs = state_in_final_episode
    if not isinstance(actions_, np.ndarray):
        actions_ = np.array(actions_)
    if not isinstance(state_beliefs,np.ndarray):
        state_beliefs = np.array(state_beliefs)           
    state_beliefs_squeezed = state_beliefs.squeeze(axis=(1, 4))
    fig, axes = plt.subplots(ncomp, 2, figsize=(18, 12), gridspec_kw={'width_ratios': [1, 1]})

    # Loop through each component and create heatmaps and action plots
    for i in range(ncomp):
        # Heatmap
        sns.heatmap(state_beliefs_squeezed[:, i, :].T, ax=axes[i, 0], cmap='viridis',  # Time steps (1-50)
                    yticklabels=['No Damage', 'Minor Damage', 'Major Damage', 'Failure'])
        
        # Set titles and labels for heatmap
        axes[i, 0].set_title(f'State Heatmap for Component {i + 1}')
        axes[i, 0].set_ylabel('States')
        axes[i, 0].invert_yaxis()  # Invert y-axis to have state 1 at the bottom
        y_tick_labels = ['Do Nothing', 'Inspect', 'Repair']
        # Action plot
        axes[i, 1].plot(range(1, 51), actions_[:, :, i])
        axes[i, 1].set_title(f'Actions for Component {i + 1}')
        axes[i, 1].set_ylabel('Action')
        axes[i, 1].set_yticks(range(len(y_tick_labels)))  # Set y-ticks to [0, 1, 2]
        axes[i, 1].set_yticklabels(y_tick_labels)  # Set custom y-tick labels
        axes[i, 1].grid(False)

    # Set common labels for the x-axis
    axes[-1, 0].set_xlabel('Time Steps')  # X-label for the first column
    axes[-1, 1].set_xlabel('Time Steps')  # X-label for the second column
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def write_to_csv(csv_file_name, results):
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header
        writer.writerow(['r1', 'r2', 'Average Episodes', 'Average Episode Cost', 'Total Average Cost'])

        # Write the data
        writer.writerows(results)


