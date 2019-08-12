# -*- coding: utf-8 -*-
"""
Created on Tue Aug 3 11:02:42 2019

@author: PascPeli

The Catch game environment used during the experiments of "RL Policy Tuning"
The code is based on Eder Santana Catch evn (https://edersantana.github.io/articles/keras_rl/)
The code has been modified to include extra game_modes that change the way the fruit is falling
Two new reward functions have been implemented  "Penalized Once Reward Function" and "Penalized Reward Function"
"""
import numpy as np



class Catch():
    def __init__(self, grid_X=10, grid_Y=10, game_mode='straight', reward_mode='default'):
        self.grid_X = grid_X
        self.grid_Y = grid_Y
        self.game_mode = game_mode
        self.reward_mode = reward_mode
        self.p = -0.04
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new, states, reward and info
        """
        state = self.state
        if action == 0:  # left
            action = -1
            self.moves_cnt += 1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
            self.moves_cnt += 1

        fruit_row, fruit_col, basket = state[0]
        fruit_row, fruit_col = self._fruit_next_state (fruit_row, fruit_col)

        if (basket+action<=0) or (basket+action>=self.grid_X):
            self.against_wall_cnt += 1

        new_basket = min(max(1, basket + action), self.grid_X-1) # grid = X

        out = np.asarray([fruit_row, fruit_col, new_basket])[np.newaxis]
        #out = out

        assert len(out.shape) == 2
        self.state = out

    def _fruit_next_state (self, fruit_row, fruit_col):
        '''
        Returns the next state of the fruit based on the environments game_mode
        Straight Free Fall (default) - The fruit is falling down in a straight line.
        Diagonal Fall - The fruit makes two moves per step, one down and one left or right.  
        Diagonal Slow Fall - The fruit makes one move per step, once down and once to the side, left or right. 
        Random Fall - The fruit next move is randomly picked from the sets [0,1] (stay, move down) for the Y-axis
            and [-1,0,1] (left, stay, right) for the X-axis. This game mode makes the environment Non-Deterministic.
        
        Input: fruit_row, fruit_col
        Ouput: new_fruit_row, new_fruit_col
            
        '''
        if self.game_mode == 'straight':
            return fruit_row+1, fruit_col
        elif self.game_mode == 'diagonal':
            # if the fruit encounters a "wall" it bounces of it changing its direction on the X-axis
            if (fruit_col + self.diagonal<0) or (fruit_col + self.diagonal > self.grid_X-1):
                self.diagonal = self.diagonal * (-1)
            fruit_col = fruit_col + self.diagonal
            return fruit_row+1, fruit_col
        elif self.game_mode == 'diagonal_slow':
            # this ensures that in one step the fruit will move on the X-axis and the next on the Y-axis
            if self.row_or_col: 
                # if the fruit encounters a "wall" it bounces of it changing its direction on the X-axis
                if (fruit_col + self.diagonal<0) or (fruit_col + self.diagonal > self.grid_X-1): 
                    self.diagonal = self.diagonal * (-1)
                fruit_col = fruit_col + self.diagonal
            else:
                fruit_row += 1
            self.RorC = not(self.RorC)
            return fruit_row, fruit_col
        elif self.game_mode == 'random':
            row = np.random.choice([0,1], 1)
            col = np.random.choice([-1,0,1], 1)
            while (fruit_row + row < 0) or (fruit_row + row > self.grid_Y-1):
                row, _ = np.random.choice([0,1], 2)
            while (fruit_col + col < 0) or (fruit_col + col > self.grid_X-1):
                _, col = np.random.choice([-1,0,1], 2)
            return fruit_row+row, fruit_col+col

    def _draw_state(self):
        im_size = (self.grid_Y, self.grid_X)#(self.grid_size,)*2   #  or the oposite
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit

        canvas[-1, state[2]-1:state[2]+2] = 1  # draw basket
        return canvas

    def _reward (self):
        '''
                                       |  0  t < last_step
            Default :    reward(t,w) = |  1  t = last_step, w = 1
                                       | -1  t = last_step, w = 0
                            
                                       |  0                  t < last_step
        Penalized Once : reward(t,w) = |  1 + ExtraMoves*p   t = last_step, w = 1
                                       | -1 + ExtraMoves*p   t = last_step, w = 0
        
                                       |      ExtraMoves*p  t < last_step
          Penalized :    reward(t,w) = |  1 + ExtraMoves*p  t = last_step, w = 1
                                       | -1 + ExtraMoves*p  t = last_step, w = 0
        
        '''
        fruit_row, fruit_col, basket = self.state[0]
        if self.reward_mode == "default":
            # if the fruit is at the last row or the env
            if fruit_row == self.grid_Y-1:
                # if the fruit is in the basket
                if abs(fruit_col - basket) <= 1: # <=1 here takes into acount all the 3 blocks that "basket" ocupies
                    self.win = 1
                    return 1
                else:
                    return -1
            else:
                return 0
        elif self.reward_mode == "penalty_once":
            if fruit_row == self.grid_Y-1:
                penalty = (self.moves_cnt * self.p) + (self.against_wall_cnt * self.p)
                if abs(fruit_col - basket) <= 1: 
                    self.win = 1
                    return 1 + penalty
                else:
                    return -1 + penalty
            else:
                return 0
        elif self.reward_mode == "penalty":
            penalty = (self.moves_cnt * self.p) + (self.against_wall_cnt * self.p)
            if fruit_row == self.grid_Y-1:
                if abs(fruit_col - basket) <= 1:
                    self.win = 1
                    return 1 + penalty
                else:
                    return -1 + penalty
            else:
                return penalty


    def _get_reward(self):
        '''
        Default Reward Function
                      |  0  t < last_step
        reward(t,w) = |  1  t = last_step, w = 1
                      | -1  t = last_step, w = 0
        '''
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_Y-1:
            if abs(fruit_col - basket) <= 1: # <=1 here takes into acount all the 3 blocks that "basket" ocupies
                self.win = 1
                return 1
            else:
                return -1
        else:
            return 0

    def _get_reward_penalised_once(self):
        '''
        Penalized Once Reward Function
                      |  0                  t < last_step
        reward(t,w) = |  1 + ExtraMoves*p   t = last_step, w = 1
                      | -1 + ExtraMoves*p   t = last_step, w = 0
        '''
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_Y-1:
            penalty = (self.moves_cnt * self.p) + (self.against_wall_cnt * self.p)
            if abs(fruit_col - basket) <= 1: # <=1 here takes into acount all the 3 blocks that "basket" ocupies
                self.win = 1
                return 1 + penalty
            else:
                return -1 + penalty
        else:
            return 0

    def _get_reward_penalised(self):
        '''
        Penalized Reward Function
                      |      ExtraMoves*p  t < last_step
        reward(t,w) = |  1 + ExtraMoves*p  t = last_step, w = 1
                      | -1 + ExtraMoves*p  t = last_step, w = 0
        '''
        fruit_row, fruit_col, basket = self.state[0]
        penalty = (self.moves_cnt * self.p) + (self.against_wall_cnt * self.p)
        if fruit_row == self.grid_Y-1:
            if abs(fruit_col - basket) <= 1: # <=1 here takes into acount all the 3 blocks that "basket" ocupies
                self.win = 1
                return 1 + penalty
            else:
                return -1 + penalty
        else:
            return penalty

    def _is_over(self):
        if self.state[0, 0] == self.grid_Y-1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
#        reward = self._reward()
        if self.reward_mode == 'default':
            reward = self._get_reward()
        elif self.reward_mode == 'penalty_once':
            reward = self._get_reward_penalised_once()
        elif self.reward_mode == 'penalty':
            reward = self._get_reward_penalised()
        game_over = self._is_over()
        return self.observe(), reward, game_over, self.win

    def reset(self):
        self
        self.win = 0
        self.moves_cnt = 0
        self.moves_old = 0
        self.against_wall_cnt = 0
        self.wall_old = 0
        if self.game_mode != 'straight':
            self.diagonal = np.random.choice([-1,1], 1).item()
            self.row_or_col = 0

        fruit_row = 0
        fruit_col = np.random.randint(0, self.grid_X-1, size=1) 
        basket_col = self.grid_X // 2 #np.random.randint(1, self.grid_X-2, size=1).item() 
        self.state = np.asarray([fruit_row, fruit_col, basket_col])[np.newaxis]

    def get_extramoves(self):
        '''
        Returns the number of extra moves performed by the agent during this episode
        '''
        return self.moves_cnt, self.against_wall_cnt
