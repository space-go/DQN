# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:34:49 2019

@author: heyon
"""
#import os
import random
import numpy as np
import config as c
import schefunc as sf
from copy import deepcopy


class SatScheduling(object):
    """
    Class catch is the actual game.
    In the game, fruits, represented by white tiles, fall from the top.
    The goal is to catch the fruits with a basked (represented by white tiles, this is deep learning, not game design).
    """
    def __init__(self):
        self.state_size = c.state_size
        self.state = []#用来记录state
        self.treat = []#第一个值表示第i个任务有几个时间窗，第二个值就i是第二个值存tuple或者数对。
        self.result = []#用来存调度结果。
        self.orbit_result = []#用来存当前轨的结果
        self.orbits = []#用来存轨道信息。
        self.orbit_tasks = [[]]#当前处理的轨道中的任务，会随着调度的进行而减少。
        self.orbit_tasks_origin = [[]]#当前轨道中可以执行的所有任务，不会随着调度的进行而变化。
        self.orbit_task_number = []#当前轨任务id
        self.orbit_reward = 0
        self.selected_tasks = []#已经选择的这一轨的任务的id
        self.ot_count = 0#当前轨道任务数量
        self.cur_orbit = 0#当前轨道编号
        self.total_orbit = 0#总共的轨道数
        self.total_reward = 0#目前获得的总收益
        #self.isdone = [0 for i in range(c.state_size)] #记录这个任务有没有被做过
        self.cur_time = 0

    def _update_state(self, action):#主要更新state[4]，也就是任务做了没做！
        """
        Input: action and states
        Ouput: new states and reward
        """

        if action != c.num_actions - 1:
            self.orbit_tasks = deepcopy(self.orbit_tasks_origin)
            self.orbit_task_number = [self.orbit_tasks_origin[k][0] for k in range(len(self.orbit_tasks_origin))]
            for i in range(len(self.orbit_tasks_origin)):
                self.state[4][self.orbit_tasks_origin[i][0]] = 0
            for i in range(len(self.orbit_result)):
                self.state[4][self.orbit_result[i][0]] = 1
                for j in range(len(self.orbit_tasks)):
                    if self.orbit_result[i][0] == self.orbit_tasks[j][0]:
                        self.orbit_tasks[j] = []
                        self.orbit_task_number.remove(self.orbit_result[i][0])
                while [] in self.orbit_tasks:
                    self.orbit_tasks.remove([])
            self.orbit_task_number.append(c.num_actions - 1)

    def _get_reward(self, action):
        if action == c.num_actions - 1:
            reward = 0
        else:
            for i in range(len(self.orbit_result)):
                self.state[4][self.orbit_result[i][0]] = 0
            cur_reward = self.orbit_reward
            last_reward = sf.single_orbit_schedule(self)
            reward = last_reward - cur_reward
        return reward

    def _orbit_is_over(self, action):
        if action == c.num_actions - 1:
            return True
        else:
            return False
        
    def observe(self):
        cans = [[] for i in range(c.state_dim)]
        state = self.state
        cans[0] = deepcopy(state[0])
        cans[1] = deepcopy(state[1])
        cans[2] = deepcopy(state[2])
        cans[3] = sf.cal_remain_vtw(self)
        cans[4] = deepcopy(state[4])
        for i in range(c.state_size):
            cans[0][i] = (state[0][i] - c.low_lon)/(c.high_lon - c.low_lon)
            cans[1][i] = (state[1][i] - c.low_lat)/(c.high_lat - c.low_lat)
            cans[2][i] = state[2][i] / 10.0
        return np.reshape(cans, (1, -1))

    def act(self, action):
        reward = self._get_reward(action)
        self._update_state(action)
        orbit_over = self._orbit_is_over(action)
        return self.observe(), reward, orbit_over

    def reset(self):
        n = [0 for i in range(c.state_size)]
        m = [0 for i in range(c.state_size)]
        b = [0 for i in range(c.state_size)]
        t = [5 for i in range(c.state_size)]
        y = [0 for i in range(c.state_size)]
        for i in range(c.state_size):
            m[i] = random.uniform(c.low_lon, c.high_lon)
            n[i] = random.uniform(c.low_lat, c.high_lat)
            n[i] = round(n[i], 4)
            m[i] = round(m[i], 4)
            b[i] = random.randint(c.lowest_profit, c.highest_profit)
        tb = np.sum(b)
        print(tb)
        tw = open("D:\\result.txt", 'a+')
        tw.write("Max profits: " + str(tb) + '\n')
        tw.close()
        self.state = [n, m, b, t, y]
        sf.calc_vtw(self)

    def epoch_init(self):
        self.result = []#用来存调度结果。
        self.orbit_result = []#用来存当前轨的结果
        self.orbit_tasks = [[]]#当前处理的轨道中的任务，会随着调度的进行而减少。
        self.orbit_tasks_origin = [[]]#当前轨道中可以执行的所有任务，不会随着调度的进行而变化。
        self.orbit_task_number = []#当前轨任务id
        self.orbit_reward = 0
        self.selected_tasks = []#已经选择的这一轨的任务的id
        self.ot_count = 0#当前轨道任务数量
        self.cur_orbit = 0#当前轨道编号
        self.total_reward = 0#目前获得的总收益
        self.cur_time = 0
        y = [0 for i in range(c.state_size)]
        self.state[4] = y

    def prepdata(self, t, d, n):
        sf.loadtxt(self, t, d, n)
