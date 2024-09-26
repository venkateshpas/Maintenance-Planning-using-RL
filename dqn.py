from functions import *
import math
import random
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

plt.ion()





device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = nacomp

state = np.zeros((1,ncomp,nstcomp,1))
n_observations = np.prod(state.shape[1:])
#Two neural networks

policy_net = DQN(n_observations,ncomp*n_actions).to(device)
target_net = DQN(n_observations,ncomp*n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

    
def select_action(state):
    global steps_done
    sample = random.random()
    #state = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Getting actions for all components
            #return policy_net(state.view(1, -1)).max(1).indices.view(1, -1)  # Shape should be (1, n_components)
            output = policy_net(state.view(1, -1)).view(ncomp, -1)
            return output.max(1).indices.view(1, -1)
    else:
        # Sample random actions for each component
        return torch.randint(0, 3, (1, ncomp), device=device)  #
    
episode_durations = []

def plot_durations(show_result = False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    plt.show()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    state_batch = torch.cat([s.view(-1) for s in batch.state])
    action_batch = torch.cat(batch.action)
    #reward_batch = torch.cat(batch.reward)
    reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(torch.cat([s.view(-1) for s in batch.next_state if s is not None])).max(1).values
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    state = np.zeros((1, ncomp, nstcomp, 1))
    state[0, :, 0, 0] = 1
    state = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1)
    done = False
    total_ep_cost = np.zeros(num_episodes)
    for t in range(50):
        if t == 49:
            done = True
        action = select_action(state)
        # print(action)
        reward = immediatecost(action, cost_comp_action)
        
        if is_system_failed(state.view(1, ncomp, nstcomp, 1)):
            reward -= 2400

        if done:
            next_state = None
        else:
            # print(state)
            # print('Yo')
            # print(action)
            state_a = state_action(state.view(1, ncomp, nstcomp, 1), action)
            next_state, oo = system_state(state_a, pcomp1, action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0) if next_state is not None else None
        
        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    print('Reached here')
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

