# -*- coding: utf-8 -*-
"""
Created on Fri Aug 5 20:57:43 2019

@author: PascPeli

This is a the main script used for training the agents for the "RL Policy Tuning" experiments
"""

import os
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import model_from_json

from CatchClass import Catch


def display_statistics(nof_img, grid_X, grid_Y, results, c, folder_name):
    '''
    Create .png images of the statistics of the episode or the whole testing session.
    input:    
        nof_img (int) - Number of Images to create in sequence.
        grid_X, grid_Y (int) - dimentions of the images (same as env dims)
        results - 
        c (int) - counter used for the naming of the images
        folder_name (str) - path to store the images
    '''
    nof_img = nof_img
    input_t = np.zeros((1,grid_X*grid_Y))
    plt.imshow(input_t.reshape((grid_Y, grid_X)), interpolation='none', cmap='gray')
    # if results is an nd array. This is true at the end of the testing session for each game_mode
    if len(results.shape)==2: 
        e, _, _, moves, wall, steps, _ = np.sum(results,axis=0)
        wins = results[-1,1]
        win_rate = results[-1,2]
        plt.text(grid_X/2-0.3,grid_Y/2, "Tot_Wins: {:d}\nTot_Steps: {:d}\nTot_Moves: {:d}\nTot_WallHits: {:d}\n".format(int(wins),int(steps),int(moves),int(wall)),
                 horizontalalignment='center',verticalalignment='center',fontsize=20 ,weight='bold',color='w' )
        for i in range(nof_img*2):
            plt.savefig(os.path.join(folder_name,"%04d.png" % c))
            c +=1
        plt.cla()
        plt.imshow(input_t.reshape((grid_Y, grid_X)), interpolation='none', cmap='gray')
        plt.text(grid_X/2-0.3,grid_Y/2, "Win_rate: {:.2f}\nMoves/Steps:{:.2f}\nWallHits/Steps:{:.2f}\n".format(win_rate,moves/steps,wall/steps),
                 horizontalalignment='center',verticalalignment='center',fontsize=18 ,weight='bold',color='w' )
        for i in range(nof_img*2):
            plt.savefig(os.path.join(folder_name,"%04d.png" % c))
            c +=1
        plt.cla()
    # if results is an 1d array. this is true at each time-step of the same session
    else:   
        e, _, _, moves, wall, steps, info = results
        win_str = 'WIN' if info else 'DEFEAT'
        plt.text(grid_X/2-0.3,grid_Y/2, "Epoch: {:d}\nSteps: {:d}\nMoves: {:d}\nWallHits: {:d}\n\n{}".format(int(e),int(steps),int(moves),int(wall),win_str),
                 horizontalalignment='center',verticalalignment='center',fontsize=20 ,weight='bold',color='w' )
        for i in range(nof_img):
            plt.savefig(os.path.join(folder_name,"%04d.png" % c))
            c +=1
        plt.cla()


def main(epochs=1000, grid_X=10, grid_Y=10, nof_img=5, nof_vid=20):

    reward_modes = ['default', 'penalty_once', 'penalty']
    game_modes = ['straight','diagonal', 'diagonal_slow', 'random']

    all_results = {}
    # create a linspace of int. The images of the epochs in this linspace will be stored
    vid_e = np.linspace(0,epochs-1, num=nof_vid, dtype=int)

    model_path = os.path.join(os.getcwd(),'data','model')
    with open(os.path.join(model_path,"model.json"), "r") as jfile:
        model = model_from_json(json.load(jfile))


    for reward_mode in reward_modes:

        for game_mode in game_modes:
            weights_path = os.path.join(model_path, 'weights', "Weights_"+reward_mode+'_'+game_mode+".h5")
            model.load_weights(weights_path)
            model.compile("sgd", "mse")

            # Define environment, game
            env = Catch(grid_X, grid_Y, game_mode,reward_mode)

            results = np.zeros((epochs,7))
            all_actions = []
            all_rewards = []

            c = 0
            win_cnt = 0
            for e in range(epochs):
                steps_per_game = 0
                actions = []
                rewards = []

                env.reset()
                game_over = False
                # get initial input
                input_t = env.observe()
                
                if e in vid_e:
                    folder_name = os.path.join(os.getcwd(), 'data', 'test', 'images', reward_mode+'_'+game_mode)
                    if not os.path.isdir(folder_name):
                        os.makedirs(folder_name)
    
                    plt.imshow(input_t.reshape((grid_Y, grid_X)),
                               interpolation='none', cmap='gray')
                    plt.savefig(os.path.join(folder_name,"%04d.png" % c))
                    plt.cla()
                    c += 1

                while not game_over:
                    steps_per_game += 1
                    input_tm1 = input_t

                    # get next action
                    q = model.predict(input_tm1)
                    action = np.argmax(q[0])

                    # apply action, get rewards and new state
                    input_t, reward, game_over, info = env.act(action)
                    actions.append(action.item())
                    rewards.append(reward)
                    if info == 1:
                        win_cnt += 1    
                    if e in vid_e:
                        plt.imshow(input_t.reshape((grid_Y, grid_X)),
                                   interpolation='none', cmap='gray')
                        plt.savefig(os.path.join(folder_name,"%04d.png" % c))
                        plt.cla()
                        c += 1

                moves, wall = env.get_extramoves()
                results[e] = e+1, win_cnt, win_cnt/(e+1), moves, wall, steps_per_game, info
                all_actions.append(actions)
                all_rewards.append(rewards)
                
                if e in vid_e:
                    display_statistics(nof_img, grid_X, grid_Y, results[e], c, folder_name)
                    c += nof_img

            display_statistics(nof_img, grid_X, grid_Y, results, c, folder_name)
            all_results[reward_mode+'_'+game_mode] = [results, all_actions, all_rewards]

    filename = os.path.join(os.getcwd(),'data','test','all_results_test.pickle')
    with open(filename,'wb') as handle:
        pickle.dump(all_results, handle)


if __name__ == "__main__":

    main()

