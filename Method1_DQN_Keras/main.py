# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:09:52 2019

@author: heyon
"""
import time
import gc
import seaborn
import keras.backend
import config as c
import tandt as tt


def InitPara(tn):
    c.state_size = tn
    c.num_actions = tn + 1
    if tn < 100:
        c.low_lat = 20  # -65#
        c.high_lat = 30  # 65#
        c.low_lon = 108  # -180#
        c.high_lon = 114  # 180#
    else:
        c.low_lat = 3  # -65#
        c.high_lat = 53  # 65#
        c.low_lon = 73  # -180#
        c.high_lon = 133  # 180#

if __name__ == "__main__":
    tw = open("C:\\Users\\Administrator\\Desktop\\yingyong\\res.txt", 'a+')
    tw.write("Algorithm: " + str(c.algorithm) + '\n')
    tw.close()
    for tn in c.task_set_num:
        tw = open("C:\\Users\\Administrator\\Desktop\\yingyong\\res.txt", 'a+')
        tw.write("Task sets: " + str(tn) + '\n')
        tw.close()
        InitPara(tn)
        keras.backend.clear_session()
        seaborn.set()
        model = tt.baseline_model(c.state_size * c.state_dim, c.num_actions, c.hidden_size)
        model.summary()

        if c.trainDQN:
            train_start = time.perf_counter()
            model = tt.train(model, c.epoch)
            train_end = (time.perf_counter() - train_start)
            tw = open("C:\\Users\\Administrator\\Desktop\\yingyong\\res.txt", 'a+')
            tw.write("Training time: " + str(train_end) + '\n')
            tw.close()

        if c.testDQN:
            test_start = time.perf_counter()
            tt.test(model, tn)
            test_end = (time.perf_counter() - test_start)
            print("Time used to test:", test_end)
            tw = open("C:\\Users\\Administrator\\Desktop\\yingyong\\res.txt", 'a+')
            tw.write("Testing time: " + str(test_end) + '\n')
            tw.close()
        del model
        gc.collect()
        time.sleep(60)
