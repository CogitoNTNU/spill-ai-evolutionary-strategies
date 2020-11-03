import numpy as np
import math as m
def f(x):
    return np.maximum(0,x)

def actionloop(obs, actions, vec, vec2, vec3, vec4):

    #print(np.dot(obs,vec))

    o2 = f(np.array(np.dot(obs.T,vec)))

    o3 = f(np.array(np.dot(o2.T, vec2)))

    o4 = f(np.array(np.dot(o3.T, vec3)))

    sum = np.dot(o4.T,vec4)
    action = np.zeros(actions)
    for i in range(len(sum)):
        action[i] = 2/(np.exp(-sum[i])+1)-1
    return action

