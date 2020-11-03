import numpy as np
def f(x):
    return x
    #return np.maximum(0,x)
def actionloop(obs, vec, vec2, vec3, vec4):

    #print(np.dot(obs,vec))
    o2 = f(np.array(np.dot(obs.T,vec)))
    o3 = f(np.array(np.dot(o2.T, vec2)))
    o4 = f(np.array(np.dot(o3.T, vec3)))
    sum = np.dot(o4.T,vec4)
    action = 0
    highest = sum[0]
    for i in range(len(sum)):
        if sum[i]> highest:
            action = i
    return action

