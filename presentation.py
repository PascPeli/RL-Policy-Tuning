# -*- coding: utf-8 -*-
"""
Created on Tue Aug 6 19:57:43 2019

@author: PascPeli

This is a Set of utility functions used for the presentation and visualisation
of the results collected during the experiments of "RL Policy Tuning"

"""

import os
import pickle
import imageio
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.rcParams['animation.embed_limit'] = 50 # MB

path = os.path.join(os.getcwd(),'data','test')

reward_modes = ['default', 'penalty_once', 'penalty']
game_modes = ['straight','diagonal', 'diagonal_slow', 'random']


def load_results(folder='test'):
    '''
    Loads and returns a dict that was previously saved during training or testing
    '''
    results = {}
    if folder=='test':
        file_path = os.path.join(os.getcwd(),'data','test','all_results_test.pickle')
    elif folder=='train':
        file_path = os.path.join(os.getcwd(),'data','train','all_results_train.pickle')
    with open(file_path, 'rb') as handle:
        results = pickle.load(handle)
    return results


def plot_graphs(results, game_mode, extensive=False, statistics=True):
    '''

    Input :
        results - dict
        game_mode - str, one of 'straight','diagonal', 'diagonal_slow', 'random'
        extensive - bool
        statistics - bool
    Output:
        Returns a figure with 2 or 6 subplots (7 in case game_mode == 'random')
        extensive=False:
            1. Win Rate - The success rate of each agent
            2. Extra Moves per Steps - The number of undesirable behaviors performed by the agent per number of steps.
        extensive=True:
            1. Wins - The number of successful episodes of each agent
            2. Win Rate - The success rate of each agent
            3. Moves - The number of Left or Right actions of each agent
            4. Wall Hits - The number of Against-the-Wall actions of each agent
            5. Moves per Steps - The number of Left or Right actions performed by the agent per number of steps.
            6. Wall Hits pes Steps - The number of Against-the-Wall actions performed by the agent per number of steps.
            7. Steps per Epoch - The number of environment steps from begining to end.Meaningful only if game_mode == 'random', otherwise a straight line.
    '''
    title = game_mode.title()

    if extensive:
        nof_graphs = 7 if game_mode == 'random' else 6
        fig, axs = plt.subplots(nof_graphs, figsize=(20, 7*nof_graphs))
        axs[0].set_title('Wins', color='C0'); axs[0].set_ylabel('Wins')
        axs[1].set_title('Win rate ({})'.format(title), color='C0'); axs[1].set_ylabel('Win Rate')
        axs[2].set_title('Moves ({})'.format(title), color='C0'); axs[2].set_ylabel('Moves')
        axs[3].set_title('Wall Hits ({})'.format(title), color='C0'); axs[3].set_ylabel('Wall Hits')
        axs[4].set_title('Moves per Step ({})'.format(title), color='C0'); axs[4].set_ylabel('Moves / Step')
        axs[5].set_title('Wall Hits per Step ({})'.format(title), color='C0'); axs[5].set_ylabel('Wall Hits/Step')
        if game_mode == 'random':
            axs[6].set_title('Steps per Epoch ({})'.format(title), color='C0'); axs[6].set_ylabel('Steps')
        for reward_mode in reward_modes:
            axs[0].plot(results[reward_mode+'_'+game_mode][0][:,1], label=reward_mode)
            axs[1].plot(results[reward_mode+'_'+game_mode][0][:,2], label=reward_mode)
            axs[2].plot(results[reward_mode+'_'+game_mode][0][:,3].cumsum(), label=reward_mode)
            axs[3].plot(results[reward_mode+'_'+game_mode][0][:,4].cumsum(), label=reward_mode)
            axs[4].plot(results[reward_mode+'_'+game_mode][0][:,3].cumsum()/results[reward_mode+'_'+game_mode][0][:,5].cumsum(), label=reward_mode)
            axs[5].plot(results[reward_mode+'_'+game_mode][0][:,4].cumsum()/results[reward_mode+'_'+game_mode][0][:,5].cumsum(), label=reward_mode)
            if game_mode == 'random':
                axs[6].plot(results[reward_mode+'_'+game_mode][0][:,5].cumsum()/results[reward_mode+'_'+game_mode][0][:,0], label=reward_mode)
    else:
        nof_graphs = 2
        fig, axs = plt.subplots(nof_graphs, figsize=(20, 7*nof_graphs))
        axs[0].set_title('Win rate ({})'.format(title), color='C0'); axs[0].set_ylabel('Win Rate')
        axs[1].set_title('Extra Moves per Step ({})'.format(title), color='C0'); axs[1].set_ylabel('Extra Moves / Steps')
        for reward_mode in reward_modes:
            axs[0].plot(results[reward_mode+'_'+game_mode][0][:,2], label=reward_mode)
            axs[1].plot((results[reward_mode+'_'+game_mode][0][:,3]+results[reward_mode+'_'+game_mode][0][:,4]).cumsum() / results[reward_mode+'_'+game_mode][0][:,5].cumsum(), label=reward_mode)

    for i in range(nof_graphs):
        axs[i].grid()
        axs[i].legend()
        axs[i].set_xlabel('Epochs')

    if statistics:
        if extensive:
            fig.subplots_adjust(top=0.88)
        else:
            fig.subplots_adjust(top=0.70)
        text=''
        for reward_mode in reward_modes:
            if results[reward_mode+'_'+game_mode][0].shape[1] == 7:
                _, _, _, moves, wall, steps, _ = np.sum(results[reward_mode+'_'+game_mode][0], axis=0)
            else:
                _, _, _, moves, wall, steps, _, _, _ = np.sum(results[reward_mode+'_'+game_mode][0], axis=0)
            wins = results[reward_mode+'_'+game_mode][0][-1,1]
            win_rate = results[reward_mode+'_'+game_mode][0][-1,2]
            mode = 'Penal_O' if reward_mode =='penalty_once' else reward_mode.title()
            text += "\n%s: Wins: %d | WinRate: %.2f | Moves: %d | WallHits: %d | Steps: %d" %(mode,wins,win_rate, moves, wall, steps)
        fig.suptitle('{}\n{}'.format(title,text), fontsize=30,color='b')
    else:
        if extensive:
            fig.subplots_adjust(top=0.95)
        else:
            fig.subplots_adjust(top=0.90)
        fig.suptitle(title, fontsize=30,color='b')


def count_uniques(data, uniques=[0,1,2]):
    '''
    Returns a dict with the count of unique elements of iterable object
    It is particulary useful when the rows of the array are not of the same leght
    '''
    counts = dict( zip(uniques, np.zeros(len(uniques),int) ))
    for item in data:
        for i in item:
            counts[i] += 1
    return counts

def get_uniques(data,digits=5):
    '''

    '''
    uniques = set()
    for item in data:
        for i in item:
            uniques.add(round(i,digits))
    return uniques

def noD_mode(data, uniques=[0,1,2]):
    '''
    no_Dimension_mode returns an np.array of the mode of an iterable object.
    It returns the same result as scipy.stats.mode(array, axis=0)
    It is particulary useful when the rows of the array are not of the same leght
    '''
    max_keys = []
    max_values = []
    i=0
    while True:
        cnt_err = 0
        counts = dict( zip(uniques, np.zeros(len(uniques),int) ))
        for item in data:
            try:
                counts[item[i]] += 1
            except IndexError:
                cnt_err += 1
                pass
        if cnt_err == len(data):
            break
        key = max(counts, key=counts.get)
        value = counts[key]
        max_keys.append(key)
        max_values.append(value)
        i += 1
    return (np.array([max_keys]),np.array([max_values]))


def plot_actions_bar (results):
    '''
    Returns a figure with 4 bar subplots of the frequency of the actions
    that the agent perfomed during training or testing
    '''
    fig, axs = plt.subplots(2,2,figsize=(20,10))
    fig.suptitle('Actions Taken Frequency', fontsize=30,color='b')

    for e, game_mode in enumerate(game_modes):
        i = e // 2    # np.array([0,1,2,3])//2 -> array([0, 0, 1, 1])
        j = e %  2    # np.array([0,1,2,3]) %2 -> array([0, 1, 0, 1])
        for r, reward_mode in enumerate(reward_modes):
            if game_mode == 'random':
                act_cnt = count_uniques(np.array(results[reward_mode+'_'+game_mode][1]))
            else:
                unique, counts = np.unique(np.array(results[reward_mode+'_'+game_mode][1]), return_counts=True)
                act_cnt = dict(zip(unique, counts))
            x = np.fromiter(act_cnt.keys(), dtype=float) + 0.2 * (r-1)
            axs[i, j].bar(x, act_cnt.values(), width=0.3,label=reward_mode)

        axs[i, j].set_xticks([0,1,2])
        axs[i, j].set_xticklabels(('left','stay','right'))
        axs[i, j].title.set_text('Actions Frequency ({})'.format(game_mode.title()));
        axs[i, j].grid()
        axs[i, j].legend()

def plot_actions_mod (results):
    '''
    Returns a figure with 4 subplots of the most frequent (mode) action that the
    agent perfomed at each time step
    '''
    fig, axs = plt.subplots(2,2,figsize=(20,10))
    fig.suptitle('Actions Taken at Each Time-Step', fontsize=30,color='b')

    for e, game_mode in enumerate(game_modes):
        i = e // 2    # np.array([0,1,2,3])//2 -> array([0, 0, 1, 1])
        j = e %  2    # np.array([0,1,2,3]) %2 -> array([0, 1, 0, 1])
        for r, reward_mode in enumerate(reward_modes):
            if game_mode == 'random':
                data = noD_mode(results[reward_mode+'_'+game_mode][1])[0][0] - 1
            else:
                data = scipy.stats.mode(np.array(results[reward_mode+'_'+game_mode][1]),axis=0)[0][0] - 1
            axs[i, j].plot(data, label=reward_mode)

        axs[i, j].set_yticks([-1, 0, 1])
        axs[i, j].set_xlabel('Steps')
        axs[i, j].set_yticklabels(('left','stay','right'))
        axs[i, j].title.set_text('Actions ({})'.format(game_mode.title()));
        axs[i, j].grid()
        axs[i, j].legend()


def gif_maker(path):
    '''
    Creates and stores on disk .gif files using the images created on testing
    '''
    path = os.path.join(os.getcwd(),'data','tests')
    foldernames = []
    for i in os.listdir(os.path.join(path,'images')):
        foldernames.append(i)

    for folder in foldernames:
        filenames = []
        for i in os.listdir(os.path.join(path,'images',folder)):
            filenames.append(os.path.join(path,'images',folder,i))
        gif_path = os.path.join(path,'gifs',folder+'.gif')
        with imageio.get_writer(gif_path, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)




def load_images(path):
    '''
    Loads and returns the images that where saved during testing
    '''
    images = {}
    for folder in sorted(os.listdir(os.path.join(path,'images'))):
        img = []
        for im in sorted(os.listdir(os.path.join(path,'images',folder))):
            img.append(os.path.join(path,'images',folder,im))
        images[folder] = np.array(img)
    return images

def animation_maker(path, save=None):
    '''
    Creates an matplotlib.animation object that can be saved
    either in .mp4 or as a jshtml str
    '''
    if save:
        vid_path = os.path.join(path,'videos')
        if not os.path.isdir(vid_path):
            os.makedirs(vid_path)

    fig, axs = plt.subplots(1,3,figsize=(15,5))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    images = load_images(path)

    def update(frame, *fargs):
        game_mode = fargs[0]
        for i, reward_mode in enumerate(reward_modes):
            axs[i].cla()
            axs[i].set_title(reward_mode, color='b', fontsize=20)
            try:
                img = images[reward_mode+'_'+game_mode][frame]
            except IndexError:
                # in case of IndeError use the last available.
                #done to keep the videos with different number of total frames
                img = images[reward_mode+'_'+game_mode][len(images[reward_mode+'_'+game_mode])-1]
            axs[i].imshow(plt.imread(img))

    for game_mode in game_modes:
        fig.suptitle(game_mode.title(), fontsize=30)
        max_len = max(len(images[reward_modes[0]+'_'+game_mode]),len(images[reward_modes[1]+'_'+game_mode]),len(images[reward_modes[2]+'_'+game_mode]))
        anim =FuncAnimation(fig, update, frames=range(max_len),fargs=(game_mode,), interval=100)

        if save == 'jshtml':
            with open(os.path.join(vid_path, game_mode+'.vidstr'), 'w') as file:
                file.write(anim.to_jshtml())
        elif save == 'video':
            anim.save(os.path.join(vid_path, game_mode+'.mp4'))

def load_jshtml(path):
    try:
        vid_dict = {}
        vid_path = os.path.join(path,'videos')
        for game_mode in game_modes:
            with open(os.path.join(vid_path, game_mode+'.vidstr')) as file:
                vid_dict[game_mode] = file.read()
        return vid_dict
    except:
        print('File not found')
        return None