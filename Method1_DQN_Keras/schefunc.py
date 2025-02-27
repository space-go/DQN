# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:28:15 2019

@author: heyon
"""
import config as c
import os
import datetime
import win32process
import win32event
import numpy as np
import copy
        
#import numpy as np


def runexe(exe_file):
    exe_path = c.path_workingdirectory
    running = False
    handle = win32process.CreateProcess(os.path.join(exe_path, exe_file),
                                        '', None, None, 0, 
                                        win32process.CREATE_NO_WINDOW,
                                        None,
                                        exe_path,
                                        win32process.STARTUPINFO())
    running = True
    while running:
        rc = win32event.WaitForSingleObject(handle[0], 1000)
        if rc == win32event.WAIT_OBJECT_0:
            running = False#end while


def calc_vtw(self):#将任务写到txt中去，然后调用exe来计算时间窗，然后将vtw存在self中（在处理的同时就可以搞）。
    #晚上要检查一下这个文件中的open和close！！！不然导致exe在运行的时候某些文件打不开而运行失败！！
    ob = open(c.path_workingdirectory + c.path_orbit, 'r')#sw4
    #下面这一段是为了计算相对时间方便。
    base_t = ob.readlines()
    base_t = base_t[0].split()
    base_t = datetime.datetime(int(base_t[0]),
                               int(base_t[1]),
                               int(base_t[2]),
                               int(base_t[3]),
                               int(base_t[4]),
                               int(base_t[5]))
    ob.close()
    
    tl = open(c.path_workingdirectory + c.path_task, 'w')
    for i in range(c.state_size):
        line1 = str(i)+' '+str(self.state[0][i])+' '+str(self.state[1][i])+' '+str(self.state[2][i])+str(self.state[3][i])+' 5 1'#先是纬度，后是经度！！！
        line2 = "2  45.0 45.0 0.00"
        tl.write(line1 + '\n' + line2 + '\n')
    tl.close()
    
    oi = open(c.path_workingdirectory + c.path_shadow , 'r')#sw2
    count = 0
    for in_out_time in oi.readlines():
        #orbit_num = in_out_time[16:17]
        if len(in_out_time) < 10:
            continue
        count_t = count % 2
        """             #考虑地影的做法。
        if count_t == 0:
            in_time = in_out_time[22:-1]
        elif count_t == 1:
            out_time = in_out_time[22:-1]
            in_split = in_time.split()
            in_datetime = datetime.datetime(int(in_split[0]),
                                            int(in_split[1]),
                                            int(in_split[2]),
                                            int(in_split[3]),
                                            int(in_split[4]),
                                            round(float(in_split[5])))
            out_split = out_time.split()
            out_datetime = datetime.datetime(int(out_split[0]),
                                             int(out_split[1]),
                                             int(out_split[2]),
                                             int(out_split[3]),
                                             int(out_split[4]),
                                             round(float(out_split[5])))
            in2int = (in_datetime - base_t).seconds
            out2int = (out_datetime - base_t).seconds            
            self.orbits.append([in2int, out2int])
        count += 1
        """
        #下面是不考虑地影的做法！！！0201修改
        if count_t == 1:
            out_time = in_out_time[22:-1]
            out_split = out_time.split()
            out_datetime = datetime.datetime(int(out_split[0]),
                                             int(out_split[1]),
                                             int(out_split[2]),
                                             int(out_split[3]),
                                             int(out_split[4]),
                                             round(float(out_split[5])))
            out2int = (out_datetime - base_t).seconds
            if count < 2:
                self.orbits.append([0, out2int])
            else:
                self.orbits.append([self.orbits[int((count - 1) / 2) - 1][1], out2int])
        count += 1
    oi.close()
    self.total_orbit = count / 2
    
    tw = open(c.path_workingdirectory + c.path_timewindow, 'w')#sw2
    #oa = open(c.path_outputattitude , 'w')#sw3
    tl = open(c.path_workingdirectory + c.path_task, 'r')
    tw.write("s1\n")
    tw.close()
    meta_num = [0 for i in range(c.state_size)]
    meta_task = [[] for i in range(c.state_size)]
    for i in range(c.state_size):
        tm = open(c.path_workingdirectory + c.path_temp , 'w')
        line1 = str(self.state[0][i]) + ' ' + str(self.state[1][i]) + ' 0'
        line2 = "2  45.0 45.0 0.00"
        tm.write(line1 + '\n' + line2 + '\n')
        tm.close()
                
        runexe(c.path_workingdirectory + c.path_cc4_2)

        ac = open(c.path_workingdirectory + c.path_access, 'r')#sw4
        #ag = open(c.path_angle, 'r')     
        #统计文件行数
        lines = 0
        while True:
            buffer=ac.read(1024*8192)
            if not buffer:
                break
            lines += buffer.count('\n')
        ac.close()
        tw = open(c.path_workingdirectory + c.path_timewindow , 'w')#sw2
        tw.write(str(i) + " " + str(lines) + "\n")
        meta_num[i] = lines
        #将angle里面的信息逐一写到tw里面去。
        ac = open(c.path_workingdirectory + c.path_access, 'r')#sw4
        vtws = []
        for line in ac.readlines():  
            tw.write(line)
            begins = line[0:28]
            ends = line[28:56]
            begins_split = begins.split()
            begins_datetime = datetime.datetime(int(begins_split[0]),
                                                int(begins_split[1]),
                                                int(begins_split[2]),
                                                int(begins_split[3]),
                                                int(begins_split[4]),
                                                max(0, min(59, round(float(begins_split[5])))))
            ends_split = ends.split()
            ends_datetime = datetime.datetime(int(ends_split[0]),
                                              int(ends_split[1]),
                                              int(ends_split[2]),
                                              int(ends_split[3]),
                                              int(ends_split[4]),
                                              max(0, min(59, round(float(ends_split[5])))))
            begins2int = (begins_datetime - base_t).seconds
            ends2int = (ends_datetime - base_t).seconds
            vtws.append([begins2int, ends2int])##################################这里要搞清楚它的嵌套关系！
        meta_task[i] = vtws
        ac.close()
        #这里预留一个读取每一时刻angle的函数！！！
        #ag.close()
        tw.close()
    #oa.close()
    self.treat = [meta_num, meta_task]
    tl.close()


def cal_remain_vtw(self):#相对的。
    remain_vtw = [0 for i in range(c.state_size)]
    for i in range(c.state_size):
        #if self.state[3][i] >= 0:
        for j in range(self.treat[0][i]):
            if self.treat[1][i][j][1] > self.cur_time:
                remain_vtw[i] += 1
        remain_vtw[i] = remain_vtw[i] / (c.max_orbit * 1.0)
        #else:
        #    remain_vtw[i] = -1
    return remain_vtw


def _get_slew_time(task1 , task2):
    return 20


def _is_in_window(task_list, task_index, t):#对应的开始时间和相应的结束时间都要在轨道内，如果再时间窗内，返回轨道圈号，如果不在，返回-1
    #for i in range(env.treat[0][task_index]):
    if t >= task_list[task_index][1] and t + task_list[task_index][3] <= task_list[task_index][2]:
        return 1
    return -1

def schedule_by_greedy(self):
    self.orbit_result = [[]]
    temp_tasks = [[]]
    orbit_reward = 0
    # self.selected_tasks = list({}.fromkeys(self.selected_tasks).keys())
    for i in range(len(self.selected_tasks)):
        for j in range(len(self.orbit_tasks_origin)):
            if self.selected_tasks[i] == self.orbit_tasks_origin[j][0]:
                temp_tasks.append([self.selected_tasks[i],
                                   self.orbit_tasks_origin[j][1],
                                   self.orbit_tasks_origin[j][2],
                                   self.orbit_tasks_origin[j][3]])
    while [] in temp_tasks:
        temp_tasks.remove([])
    # print(len(temp_tasks), '\n')
    #temp_tasks.sort(key=lambda x: x[2])  # 把orbit_tasks改成了selected_tasks
    ct = self.orbits[self.cur_orbit][0]
    last_task = -1
    for i in range(len(temp_tasks)):
        if last_task == -1:
            cur_start_time = max(ct, temp_tasks[i][1])
            if cur_start_time + self.state[3][temp_tasks[i][0]] <= temp_tasks[i][2]:
                self.orbit_result.append([temp_tasks[i][0],
                                          cur_start_time,
                                          cur_start_time + self.state[3][temp_tasks[i][0]]])
                ct = cur_start_time + self.state[3][temp_tasks[i][0]]
                # self.state[3][temp_tasks[i][0]] = 1
                last_task = temp_tasks[i][0]
                orbit_reward += temp_tasks[i][3]
            # else:
            #    self.state[3][self.orbit_tasks[i][0]] = 0
        else:
            cur_start_time = max(ct + _get_slew_time(temp_tasks[i][0], last_task), temp_tasks[i][1])
            if cur_start_time + self.state[3][temp_tasks[i][0]] <= temp_tasks[i][2]:
                self.orbit_result.append([temp_tasks[i][0],
                                          cur_start_time,
                                          cur_start_time + self.state[3][temp_tasks[i][0]]])
                ct = cur_start_time + self.state[3][temp_tasks[i][0]]
                # self.state[3][temp_tasks[i][0]] = 1
                last_task = temp_tasks[i][0]
                orbit_reward += temp_tasks[i][3]
            # else:
            #    self.state[3][self.orbit_tasks[i][0]] = 0
    # return之前要更新self.cur_time = 这一轨的开始时间。
    while [] in self.orbit_result:
        self.orbit_result.remove([])
    self.orbit_reward = orbit_reward
    return self.orbit_reward


def schedule_by_hadrt(self):
    self.orbit_result = [[]]
    temp_tasks = [[]]
    orbit_reward = 0
    # self.selected_tasks = list({}.fromkeys(self.selected_tasks).keys())
    for i in range(len(self.selected_tasks)):
        for j in range(len(self.orbit_tasks_origin)):
            if self.selected_tasks[i] == self.orbit_tasks_origin[j][0]:
                temp_tasks.append([self.selected_tasks[i],
                                   self.orbit_tasks_origin[j][1],
                                   self.orbit_tasks_origin[j][2],
                                   self.orbit_tasks_origin[j][3]])
    while [] in temp_tasks:
        temp_tasks.remove([])
    # print(len(temp_tasks), '\n')
    temp_tasks.sort(key=lambda x: x[2])  # 把orbit_tasks改成了selected_tasks
    ct = self.orbits[self.cur_orbit][0]
    last_task = -1
    for i in range(len(temp_tasks)):
        if last_task == -1:
            cur_start_time = max(ct, temp_tasks[i][1])
            if cur_start_time + self.state[3][temp_tasks[i][0]] <= temp_tasks[i][2]:
                self.orbit_result.append([temp_tasks[i][0],
                                          cur_start_time,
                                          cur_start_time + self.state[3][temp_tasks[i][0]]])
                ct = cur_start_time + self.state[3][temp_tasks[i][0]]
                # self.state[3][temp_tasks[i][0]] = 1
                last_task = temp_tasks[i][0]
                orbit_reward += temp_tasks[i][3]
            # else:
            #    self.state[3][self.orbit_tasks[i][0]] = 0
        else:
            cur_start_time = max(ct + _get_slew_time(temp_tasks[i][0], last_task), temp_tasks[i][1])
            if cur_start_time + self.state[3][temp_tasks[i][0]] <= temp_tasks[i][2]:
                self.orbit_result.append([temp_tasks[i][0],
                                          cur_start_time,
                                          cur_start_time + self.state[3][temp_tasks[i][0]]])
                ct = cur_start_time + self.state[3][temp_tasks[i][0]]
                # self.state[3][temp_tasks[i][0]] = 1
                last_task = temp_tasks[i][0]
                orbit_reward += temp_tasks[i][3]
            # else:
            #    self.state[3][self.orbit_tasks[i][0]] = 0
    # return之前要更新self.cur_time = 这一轨的开始时间。
    while [] in self.orbit_result:
        self.orbit_result.remove([])
    self.orbit_reward = orbit_reward
    return self.orbit_reward



def single_orbit_schedule(self):
    if c.algorithm == 1:
        result = schedule_by_hadrt(self)
    else:
        result = schedule_by_greedy(self)

    return result  # 返回的是当前轨的reward！！


def ft_by_orbit(environment):#筛选每一轨的任务
#从文件读取这颗星星的进出orbit信息。
#设置一个1-1000的循环，检查每一个任务在这个轨道圈次是否有VTW，并且结束时间大于time_label。如果有，那么加入input_tm1中
    #input_tm1 = [[-1,-1,-1,-1] for i in range(c.state_size)]
    #input_tm1 = input_t
    #ip_count = 0
    ot_count = 0
    #t = [-1 for i in range(c.state_size)]
    environment.orbit_tasks = [[]]
    environment.orbit_task_number = []
    environment.orbit_result = []
    environment.selected_tasks = []
    environment.orbit_reward = 0
    for i in range(c.state_size):
        if environment.state[4][i] == 1:#如果说这个任务已经被做过，就不再选！！！只有在result里面的任务，才标记为1.
            continue
        else:
            for j in range(environment.treat[0][i]):
                if environment.treat[1][i][j][0] > environment.orbits[environment.cur_orbit][0]:
                    if environment.treat[1][i][j][1] < environment.orbits[environment.cur_orbit][1]:
                        if environment.treat[1][i][j][1] - environment.treat[1][i][j][0] >= environment.state[3][i]:
                            #把这个任务加到这一轨中
                            #input_tm1[ip_count][0] = environment.state[0][i]
                            #input_tm1[ip_count][1] = environment.state[1][i]
                            #input_tm1[ip_count][2] = environment.state[2][i]
                            #input_tm1[ip_count][3] = environment.state[3][i]
                            #ip_count += 1
                            environment.orbit_tasks.append([i,
                                                            environment.treat[1][i][j][0],
                                                            environment.treat[1][i][j][1],
                                                            environment.state[2][i]])
                            environment.orbit_task_number.append(i)
                            ot_count += 1
                            #t[i] = 1
                        else:
                            break#如果满足在轨道中，但是成像时长不足，那么这个任务在这一轨就不可能有时间窗。
                    else:
                        break#如果时间窗的结束时间已经大于了这一轨的结束时间，后面的时间窗肯定更晚。
                else:
                    continue#如果时间窗开始时间早于这一轨的开始时间，那么后面还有机会。
    #environment.state[3] = t
    environment.ot_count = ot_count
    while [] in environment.orbit_tasks:
        environment.orbit_tasks.remove([])
    environment.orbit_tasks_origin = copy.deepcopy(environment.orbit_tasks)
    environment.orbit_task_number.append(c.num_actions - 1)
    #return np.reshape(input_tm1, (1, -1))

def loadtxt(self, tt, dd, nn):
    tl = open(c.path_dataset + str(tt) +"\\"+ str(dd) +"\\"+ str(nn) +"\\"+ c.path_task, 'r')
    base_t = tl.readlines()

    n = [-999 for i in range(c.state_size)]
    m = [-999 for i in range(c.state_size)]
    b = [0 for i in range(c.state_size)]
    t = [0 for i in range(c.state_size)]
    y = [1 for i in range(c.state_size)]
    for i in range(int(len(base_t) / 2)):
        base_split = base_t[2 * i].split()
        n[i] = float(base_split[1])
        m[i] = float(base_split[2])
        b[i] = int(base_split[3])
        t[i] = int(base_split[4])
        y[i] = 0
    self.state = [n, m, b, t, y]
    tl.close()

    ob = open(c.path_workingdirectory + c.path_orbit, 'r')  # sw4
    # 下面这一段是为了计算相对时间方便。
    base_t = ob.readlines()
    base_t = base_t[0].split()
    base_t = datetime.datetime(int(base_t[0]),
                               int(base_t[1]),
                               int(base_t[2]),
                               int(base_t[3]),
                               int(base_t[4]),
                               int(base_t[5]))
    ob.close()

    oi = open(c.path_workingdirectory + c.path_shadow, 'r')  # sw2
    count = 0
    for in_out_time in oi.readlines():
        # orbit_num = in_out_time[16:17]
        if len(in_out_time) < 10:
            continue
        count_t = count % 2
        """             #考虑地影的做法。
        if count_t == 0:
            in_time = in_out_time[22:-1]
        elif count_t == 1:
            out_time = in_out_time[22:-1]
            in_split = in_time.split()
            in_datetime = datetime.datetime(int(in_split[0]),
                                            int(in_split[1]),
                                            int(in_split[2]),
                                            int(in_split[3]),
                                            int(in_split[4]),
                                            round(float(in_split[5])))
            out_split = out_time.split()
            out_datetime = datetime.datetime(int(out_split[0]),
                                             int(out_split[1]),
                                             int(out_split[2]),
                                             int(out_split[3]),
                                             int(out_split[4]),
                                             round(float(out_split[5])))
            in2int = (in_datetime - base_t).seconds
            out2int = (out_datetime - base_t).seconds            
            self.orbits.append([in2int, out2int])
        count += 1
        """
        # 下面是不考虑地影的做法！！！0201修改
        if count_t == 1:
            out_time = in_out_time[22:-1]
            out_split = out_time.split()
            out_datetime = datetime.datetime(int(out_split[0]),
                                             int(out_split[1]),
                                             int(out_split[2]),
                                             int(out_split[3]),
                                             int(out_split[4]),
                                             round(float(out_split[5])))
            out2int = (out_datetime - base_t).seconds
            if count < 2:
                self.orbits.append([0, out2int])
            else:
                self.orbits.append([self.orbits[int((count - 1) / 2) - 1][1], out2int])
        count += 1
    oi.close()
    self.total_orbit = count / 2

    meta_num = [0 for i in range(c.state_size)]
    meta_task = [[] for i in range(c.state_size)]
    task_index = 0
    tw = open(c.path_dataset + str(tt) +"\\"+ str(dd) +"\\"+ str(nn) +"\\"+ c.path_timewindow, 'r')
    try:
        tw.readline()
        while True:
            line = tw.readline()
            if len(line) == 0:
                break
            line_split = line.split()
            meta_num[int(line_split[0])] = int(line_split[1])
            task_index = int(line_split[0])
            vtws = []
            for i in range(meta_num[task_index]):
                line = tw.readline()
                begins = line[0:28]
                ends = line[28:56]
                begins_split = begins.split()
                begins_datetime = datetime.datetime(int(begins_split[0]),
                                                    int(begins_split[1]),
                                                    int(begins_split[2]),
                                                    int(begins_split[3]),
                                                    int(begins_split[4]),
                                                    max(0, min(59, round(float(begins_split[5])))))
                ends_split = ends.split()
                ends_datetime = datetime.datetime(int(ends_split[0]),
                                                  int(ends_split[1]),
                                                  int(ends_split[2]),
                                                  int(ends_split[3]),
                                                  int(ends_split[4]),
                                                  max(0, min(59, round(float(ends_split[5])))))
                begins2int = (begins_datetime - base_t).seconds
                ends2int = (ends_datetime - base_t).seconds
                vtws.append([begins2int, ends2int])  ##################################这里要搞清楚它的嵌套关系！
            meta_task[task_index] = vtws
    except EOFError:
        pass

    tw.close()
    self.treat = [meta_num, meta_task]
