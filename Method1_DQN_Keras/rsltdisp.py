# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:09:52 2019

@author: heyon
"""


def display_orbit_result(environment):
    print('The result of orbit', environment.cur_orbit, ' :')
    for i in range(len(environment.orbit_result)):
        print(environment.orbit_result[i])
