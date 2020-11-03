import gym
import ALlunarlander
import numpy as np
import ALflerevekter
env = gym.make('LunarLander-v2')
env.reset()
#Parametre
npop = 50
ngen = 10000
observasjoner = 8
noder_lag1 = 16
noder_lag2 = 22
noder_lag3 = 13
actions = 4
def makeRandomWeights():
    w = np.random.randn(observasjoner,noder_lag1)
    w2 = np.random.randn(noder_lag1,noder_lag2)
    w3 = np.random.randn(noder_lag2, noder_lag3)
    w4 = np.random.randn(noder_lag3, actions)
    return w, w2, w3, w4
def run():
    #Counting losses for resetting of weights
    losscount = 0
    w,w2,w3,w4 = makeRandomWeights()

    sigma = 0.5
    alpha = 0.2
    for i_generation in range(ngen):
        generationhighscore = -100
        totalrewards = 0
        allDisturbance = np.zeros((npop, observasjoner, noder_lag1))
        allDisturbance2 = np.zeros((npop, noder_lag1, noder_lag2))
        allDisturbance3 = np.zeros((npop, noder_lag2, noder_lag3))
        allDisturbance4 = np.zeros((npop, noder_lag3, actions))
        observation = env.reset()
        # KUN FOR Å SJEKKE OM RESULTATET BLIR BEDRE!!
        if(i_generation%5==0):
            for t in range(300):

                env.render()

                action = ALlunarlander.actionloop(observation, w, w2,w3, w4)
                observation, reward, done, info = env.step(action)
                #print(env.step(action))
                if done:

                    break
            print("Generasjon ", i_generation, " Standardvektene fullførte på  {} ".format(reward))
        #print('Reward ', reward)
        R = np.zeros(npop)  # Rewardvektor
        for i_episode in range(npop):
            allDisturbance[i_episode] = np.random.randn(observasjoner, noder_lag1)
            allDisturbance2[i_episode] = np.random.randn(noder_lag1, noder_lag2)
            allDisturbance3[i_episode] = np.random.randn(noder_lag2, noder_lag3)
            allDisturbance4[i_episode] = np.random.randn(noder_lag3, actions)
            observation = env.reset()
            for t in range(300):
                wTry = w+allDisturbance[i_episode]
                w2Try = w2+allDisturbance2[i_episode]
                w3Try = w3 + allDisturbance3[i_episode]
                w4Try = w4 + allDisturbance4[i_episode]

                action = ALlunarlander.actionloop(observation, wTry, w2Try, w3Try, w4Try)
                observation, reward, done, info = env.step(action)
                if done:
                    #print("Episode finished after {} timesteps".format(t+1))
                    break
            if(reward==-50):
                reward+= t/30
            elif(reward==80):
                reward+=(300-t)
            R[i_episode] = reward
            if reward > generationhighscore:
                generationhighscore = reward
            totalrewards+=reward
            #print(reward)
        std = np.std(R)
        if(std==0):
            std=2
        A = (R - np.mean(R)) *2/ std

        # Prøver å lage en løkke som kan regne ut summen av alle vekter ganget med sine respektive rewards
        dotproduktw = np.zeros((observasjoner, noder_lag1))
        dotproduktw2 = np.zeros((noder_lag1, noder_lag2))
        dotproduktw3 = np.zeros((noder_lag2, noder_lag3))
        dotproduktw4 = np.zeros((noder_lag3, actions))
        for i in range(npop):
            dotproduktw += A[i] * allDisturbance[i]
            dotproduktw2 += A[i] * allDisturbance2[i]
            dotproduktw3 += A[i] * allDisturbance3[i]
            dotproduktw4 += A[i] * allDisturbance4[i]
        # Nå skal dotprodukt være den riktige oppdateringen til vektene

        w = w + alpha / (npop * sigma) * dotproduktw
        w2 = w2 + alpha / (npop * sigma) * dotproduktw2
        w3 = w3 + alpha / (npop * sigma) * dotproduktw3
        w4 = w4 + alpha / (npop * sigma) * dotproduktw4
        print('[gjennomsnitt, Highscore] ', totalrewards/npop, generationhighscore)
        if(np.std(R)==0):
            losscount += 1
            if losscount>8:
                w,w2,w3,w4 = makeRandomWeights()
                losscount=0
                print('Reset Weights')
        else:
            losscount=0
    env.close()