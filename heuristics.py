import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def risk_heuristic(ncomp,state,action,r1,r2):
    for j in range(0,ncomp):
        indice_damage = np.argmax(state[0,j,:,0]) #For the particular component get which is having the highest value
        if indice_damage == 1:
            if state[0,j,indice_damage,0] >= r1:
                action[0,j] = 1 #Only inspect for minor damage
        elif indice_damage == 2:
            if state[0,j,indice_damage,0] >= r2:
                action[0,j] = 2 #Repair for major damage
            # elif r2-0.2<=state[0,j,indice_damage,0]<r2:
            #     action[0,j] = 1 #Only inspect for major damage
        elif indice_damage == 3:
            action[0,j] = 2 #Do repair because this particular component has highest likelihood of failure
    return action

def age_heuristic(ncomp,state,action,age1,age2,age_components):
    for j in range(0,ncomp):
        indice_damage = np.argmax(state[0,j,:,0])
        if indice_damage == 1:
            if age_components[0,j] >= age1:
                action[0,j] = 1
        elif indice_damage == 2:
            if age_components[0,j] >= age2:
                action[0,j] = 2
            # elif age_components[j] >= age1:
            #     action[0,j] = 1
        elif indice_damage == 3:
            action[0,j] = 2
    return action

def component_heuristic():
    return 0

