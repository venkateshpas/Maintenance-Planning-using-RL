import matplotlib.pyplot as plt
import numpy as np

ncomp = 5 # No. of components in the system
nstcomp = 4 # No. of states for each component.. 1: no damage, 2: minor-damage, 3: major-damage, 4: failure
nacomp = 3 # No. of actions that can be taken.. 1: do nothing, 2: repair, 3: inspect

#Td matrix
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

deterioration = DeteriorationMatrix(ncomp,nstcomp)
c1_dm = np.array([[0.82,0.13,0.05,0],[0,0.87,0.09,0.04],[0,0,0.91,0.09],[0,0,0,1]]) # The Natural deterioration matrix for component 1
c2_dm = np.array([[0.72,0.19,0.09,0],[0,0.78,0.18,0.04],[0,0,0.85,0.15],[0,0,0,1]]) # The Natural deterioration matrix for component 2
c3_dm = np.array([[0.79,0.17,0.04,0],[0,0.85,0.09,0.06],[0,0,0.91,0.09],[0,0,0,1]]) # The Natural deterioration matrix for component 3
c4_dm = np.array([[0.8,0.12,0.08,0],[0,0.83,0.12,0.05],[0,0,0.89,0.11],[0,0,0,1]])  # The Natural deterioration matrix for component 4
c5_dm = np.array([[0.88,0.12,0,0],[0,0.9,0.1,0],[0,0,0.93,0.07],[0,0,0,1]])         # The Natural deterioration matrix for component 5

pcomp1 = deterioration.system_deterioration_matrix(c1_dm,c2_dm, c3_dm, c4_dm, c5_dm)

pp = 0.9 # What is this?

#Omega matrix: Imperfect observations of its true state
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

observation = ObservationMatrix(ncomp)
c1_om = observation.component_observation_matrix(0.8) #Omege matrix for component1
c2_om = observation.component_observation_matrix(0.85) #Omege matrix for component1
c3_om = observation.component_observation_matrix(0.9) #Omege matrix for component1
c4_om = observation.component_observation_matrix(0.95) #Omege matrix for component1
c5_om = observation.component_observation_matrix(0.8) #Omege matrix for component1


pobs_insp = observation.system_observation_matrix(c1_om, c2_om,c3_om,c4_om,c5_om)
pobs_no_insp = np.array([[1.,0],[1.,0],[1.,0],[0,1.]])

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
            o[j] = np.random.choic(range(0,4), size = None, replace=True, p=ob_dist)
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


buffer_batch = 50 #Possibly a batch size to process at once?
buffer_memo_size = 100 #The maximum memory (no. of experiences) that the algorithm can remember
episodes = int(1e6)  
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
total_ep_risk = np.zeros(episodes+1) #Total risk for each episode?
m_average_risk = np.zeros(episodes+1)

for ep in range(0,episodes+1):
    oo = np.zeros(5) # What is this?
    print(f'Episode number: {ep}')
    state = np.zeros((1,ncomp,nstcomp,1))
    state[0,:,0,0] = 1 # Every component, initial state is no damage .. so the first row for every component is made 1
    
    total_ep_cost[ep] = 0

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
        for j in range(0,ncomp):
            indice_damage = np.argmax(state[0,j,:,0]) #For the particular component get which is having the highest value
            if indice_damage == 0:
                action[0,j] = 0 #Do nothing because state is in 1
            if indice_damage == 1 or indice_damage == 2:
                action[0,j] == 2 #Only inspect for minor/major damage
            if indice_damage == 3:
                action[0,j] == 1 #Do repair because this particular component has highest likelihood of failure

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