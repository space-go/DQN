# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:09:52 2019

@author: heyon
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from random import choice

import config as c
import envidesc as ed
import rsltdisp as rd
import schefunc as sf

class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory. 
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """
    def __init__(self, max_memory=100, discount=.9):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience
        
        In the memory the information whether the game ended at the state is stored seperately in a nested array
        [...
        [experience, game_over]
        [experience, game_over]
        ...]
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        #Save a state to memory
        self.memory.append([states, game_over])
        #We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        
        #How many experiences do we have?
        len_memory = len(self.memory)
        
        #Calculate the number of actions that can possibly be taken in the game
        num_actions = model.output_shape[-1]
        
        #Dimensions of the game field
        env_dim = self.memory[0][0][0].shape[1]
        
        #We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        
        #...and the target r + gamma * max Q(s’,a’)
        #Note that our target is a matrix, with possible fields not only for the action taken but also
        #for the other possible actions. The actions not take the same value as the prediction to not affect them
        targets = np.zeros((inputs.shape[0], num_actions))
        
        #We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            
            #We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            #add the state s to the input
            inputs[i:i+1] = state_t
            
            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            targets[i] = model.predict(state_t)[0]
            
            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1)[0])
            
            #if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets
    
    
def baseline_model(size, num_actions, hidden_size):#size是statesize*维数+1（这个1就是cur_time）
    #seting up the model with keras
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(size,), activation='relu'))#statesize*4，记录经度、纬度、收益以及剩余的可见窗口数量。
    #model.add(Dense(hidden_size, input_shape=(hidden_size,), activation='relu'))  # statesize*4，记录经度、纬度、收益以及剩余的可见窗口数量。
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(SGD(learning_rate=.1), "mse")#指定损失函数和优化器。
    return model


def train(model, epochs):#, verbose = 0
    for ss in range(c.scenes):
        # Define environment/game
        env = ed.SatScheduling()
        env.reset()
        # Initialize experience replay object
        exp_replay = ExperienceReplay(max_memory=c.max_memory, discount=c.discount)
        # Train
        #Reseting the win counter
        #win_cnt = 0
        # We want to keep track of the progress of the AI over time, so we save its win count history
        #win_hist = []
        #Epochs is the number of games we play
        train_scenes_start = time.perf_counter()
        for e in range(epochs):
            loss = 0.
            #Resetting the game
            env.epoch_init()
            # get initial input
            input_t = env.observe()
            game_over = False
            while env.cur_orbit < env.total_orbit:
                #The learner is acting on the last observed game screen
                #input_t is a vector containing representing the game screen
                input_tm1 = input_t
                sf.ft_by_orbit(env)
                """
                We want to avoid that the learner settles on a local minimum.
                Imagine you are eating eating in an exotic restaurant. After some experimentation you find 
                that Penang Curry with fried Tempeh tastes well. From this day on, you are settled, and the only Asian 
                food you are eating is Penang Curry. How can your friends convince you that there is better Asian food?
                It's simple: Sometimes, they just don't let you choose but order something random from the menu.
                Maybe you'll like it.
                The chance that your friends order for you is epsilon
                """
                if env.ot_count > 0:
                    orbit_over = False
                else:
                    orbit_over = True
                while not orbit_over:
                    #env.orbit_task_number.append(c.num_actions - 1)
                    if np.random.rand() <= c.epsilon:
                        #这个random要改写成一个循环，不能random到已经做过的任务。
                        randinlist = choice(env.orbit_task_number)
                        action = randinlist
                    else:
                        #Choose yourself
                        #q contains the expected rewards for the actions
                        #print(input_tm1),说明这个是一行的向量了啊……什么鬼= =#
                        #plt.switch_backend('agg')
                        q = model.predict(input_tm1)
                        #We pick the action with the highest expected reward
                        #done
                        qinlist = []
                        for i in range(len(env.orbit_task_number)):
                            qinlist.append(q[0][env.orbit_task_number[i]])
                        #qinlist.append(q[0][c.num_actions - 1])
                        action_index = np.argmax(qinlist)
                        #if action_index < len(qinlist) - 1:
                        action = env.orbit_task_number[action_index]
                        #else:
                            #action = c.num_actions - 1
                    # apply action, get rewards and new state
                    if len(env.selected_tasks) > len(set(env.selected_tasks)): #modified in 190405
                        orbit_over = True
                    else:
                        env.selected_tasks.append(action)
                        input_t, reward, orbit_over = env.act(action)
                    # modified by ym 190403
                    #if len(env.selected_tasks) > c.state_size + 1:
                    #If we managed to catch the fruit we add 1 to our win counter
                    #if reward == 1:
                    #    win_cnt += 1        #这个东西还要么？

                    #Uncomment this to render the game here
                    #display_screen(action,3000,inputs[0])

                    """
                    The experiences < s, a, r, s’ > we make during gameplay are our training data.
                    Here we first save the last experience, and then load a batch of experiences to train our model
                    """
                    if orbit_over and env.cur_orbit == env.total_orbit:
                        game_over = True
                    # store experience
                    exp_replay.remember([input_tm1, action, reward, input_t], game_over)

                    # Load batch of experiences
                    inputs, targets = exp_replay.get_batch(model, batch_size=c.batch_size)

                    # train model on experiences
                    batch_loss = model.train_on_batch(inputs, targets)

                    #print(loss)
                    loss += batch_loss
                env.result += env.orbit_result
                env.total_reward += env.orbit_reward
                rd.display_orbit_result(env)
                env.cur_orbit += 1
            print('The', e, "epochs' total reward and number: ", env.total_reward, " ", len(env.result), '\n')
            tw = open("C:\\Users\\Administrator\\Desktop\\yingyong\\res.txt", 'a+')
            tw.write("I " + str(e) + " " + str(env.total_reward) + " " + str(
                len(env.result)) + " " + str(loss) + '\n')
            tw.close()
        train_scenes_end = (time.perf_counter() - train_scenes_start)
        print("Training time in scenes", ss, ": ", train_scenes_end, '\n')
        tw = open("C:\\Users\\Administrator\\Desktop\\yingyong\\res.txt", 'a+')
        tw.write("Training time each scenes " + str(train_scenes_end) + '\n')
        tw.close()
            #if verbose > 0:
            #    print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {}".format(e,epochs, loss, win_cnt))
            #win_hist.append(win_cnt)
    return model


def test(model, tn):
    #This function lets a pretrained model play the game to evaluate how well it is doing
    #global last_frame_time
    #plt.ion()
    # Define environment, game
    #c is a simple counter variable keeping track of how much we train
    #count = 0
    #Reset the last frame time (we are starting from 0)
    last_frame_time = 0
    #Reset score
    #points = 0
    #For training we are playing the game 10 times

    for ts in range(c.test_size):
        env = ed.SatScheduling()
        env.prepdata(ts + 1, 1, tn)
        input_t = env.observe()
        test_epoch_start = time.perf_counter()
        while env.cur_orbit < env.total_orbit:
            # The learner is acting on the last observed game screen
            # input_t is a vector containing representing the game screen
            input_tm1 = input_t
            sf.ft_by_orbit(env)
            """
            We want to avoid that the learner settles on a local minimum.
            Imagine you are eating eating in an exotic restaurant. After some experimentation you find 
            that Penang Curry with fried Tempeh tastes well. From this day on, you are settled, and the only Asian 
            food you are eating is Penang Curry. How can your friends convince you that there is better Asian food?
            It's simple: Sometimes, they just don't let you choose but order something random from the menu.
            Maybe you'll like it.
            The chance that your friends order for you is epsilon
            """
            if env.ot_count > 0 or len(env.selected_tasks) <= len(env.orbit_tasks_origin):
                orbit_over = False
            else:
                orbit_over = True
            while not orbit_over:
                # Choose yourself
                # q contains the expected rewards for the actions
                # print(input_tm1),说明这个是一行的向量了啊……什么鬼= =#
                plt.switch_backend('agg')
                q = model.predict(input_tm1)
                # We pick the action with the highest expected reward
                # done
                qinlist = []
                for i in range(len(env.orbit_task_number)):
                    qinlist.append(q[0][env.orbit_task_number[i]])
                # qinlist.append(q[0][c.num_actions - 1])
                action_index = np.argmax(qinlist)
                # if action_index < len(qinlist) - 1:
                action = env.orbit_task_number[action_index]
                # else:
                # action = c.num_actions - 1
                # apply action, get rewards and new state
                if len(env.selected_tasks) > len(set(env.selected_tasks)):#modified in 190405
                    orbit_over = True
                else:
                    env.selected_tasks.append(action)
                    input_t, reward, orbit_over = env.act(action)
                # modified by ym 190402
                #if len(env.selected_tasks) > len(env.orbit_tasks_origin):
            env.result += env.orbit_result
            env.total_reward += env.orbit_reward
            rd.display_orbit_result(env)
            env.cur_orbit += 1
        print("The total reward and number are: ", env.total_reward, " ", len(env.result))
        test_epoch_end = (time.perf_counter() - test_epoch_start)
        print("The time used are: ", test_epoch_end)
        tw = open("C:\\Users\\Administrator\\Desktop\\yingyong\\res.txt", 'a+')
        tw.write("O " + str(ts) + " " + str(env.total_reward) + " " + str(
            len(env.result)) + '\n')
        tw.close()
    print("That's all, thank you!")
