# -*- coding: utf-8 -*-
"""
Created on Tue Aug 3 11:13:25 2019

@author: PascPeli

This is a the main script used for training the agents for the "RL Policy Tuning" experiments

"""

import os
import json
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from CatchClass import Catch
from ExperienceReplayClass import ExperienceReplay

#%%

def createModel(hidden_size, input_shape, output_size, learning_rate=0.2):
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=input_shape, activation='relu'))
    model.add(Dense(hidden_size, activation='elu'))
    model.add(Dense(output_size))
    model.compile(SGD(lr=learning_rate), "mse")
    return model


def train(reward_mode, game_mode, grid_Y, grid_X, weights_path, model, num_actions,
          batch_size, epochs=1000, max_memory=500, observe=False):
    #Exploration parameters
    epsilon = 1.        # Exploration rate
    max_epsilon = 1.    #
    min_epsilon = 0.1
    decay_rate = 0.01   # Exponential decay rate for epsilon

    print ("Training in Mode:", reward_mode, game_mode)

    # Define environment/game
    env = Catch(grid_X, grid_Y, game_mode,reward_mode)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    results = np.zeros((epochs,9))
    all_actions = []
    all_rewards = []

    win_cnt = 0
    start = time.time()
    for e in range(epochs):
        steps_per_game = 0
        loss = 0.
        actions = []
        rewards = []

        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            steps_per_game += 1
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over, info = env.act(action)
            actions.append(action.item())
            rewards.append(reward)
            if info == 1:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)

            if observe:
                canv = env.observe()
                #plt.figure()
                plt.imshow(canv.reshape((grid_Y, grid_X)))
                plt.show()

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*e)
        moves, wall = env.get_extramoves()

        results[e] = e+1, win_cnt, win_cnt/(e+1),  moves, wall, steps_per_game, info, epsilon, loss
        all_actions.append(actions)
        all_rewards.append(rewards)

        if (e+1)%100 == 0 :
            end = time.time()
            time_passed = end - start
            start = time.time()
            print("Epoch {}/{} | Loss {:.4f} | Win count {} | Winrate {:.2f} |Moves {} | Wall {} | steps {} | time {:.2f}s".format(e+1, epochs, loss, win_cnt, win_cnt/(e+1), moves, wall, steps_per_game, time_passed))


    # Save trained model weights and architecture, this will be used by the visualization code
    filename = "Weights_"+reward_mode+'_'+game_mode+".h5"
    model.save_weights(os.path.join(weights_path,filename), overwrite=True)

    return [results, all_actions, all_rewards]




def main ():

    reward_modes = ['default', 'penalty_once', 'penalty']
    game_modes = ['straight','diagonal', 'diagonal_slow', 'random']

    data_path = os.path.join(os.getcwd(),'data')
    model_path = os.path.join(data_path,'model')
    weights_path = os.path.join(model_path, 'weights')
    if not os.path.isdir(weights_path):
        os.makedirs(weights_path)

    # parameters
    num_actions = 3  # [move_left, stay, move_right]
    epochs = 2000
    max_memory = 500
    hidden_size = 100
    batch_size = 64
    #grid_size = 10
    grid_X=10
    grid_Y=10

    observe = False

    model = createModel(hidden_size, (grid_X*grid_Y,), num_actions, learning_rate=0.2)
    model.save_weights(os.path.join(weights_path,"initial_weights.h5"), overwrite=True)
    with open(os.path.join(model_path,"model.json"), "w") as outfile:
        json.dump(model.to_json(), outfile)

    all_results = {}

    for reward_mode in reward_modes:
        for game_mode in game_modes:
            model.load_weights(os.path.join(weights_path,"initial_weights.h5"))
            train_results = train(reward_mode, game_mode, grid_Y, grid_X, weights_path,
                                  model, num_actions, batch_size, epochs, max_memory, observe)
            all_results[reward_mode+'_'+game_mode] = train_results

    folder_name = os.path.join(data_path, 'train')
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    filename = os.path.join(folder_name,'all_results_train.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(all_results, handle)

    return all_results

#%%

if __name__ == "__main__":
    # Run main function
    results = main()
