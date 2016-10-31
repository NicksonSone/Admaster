#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function   # log output
import numpy as np
import pandas as pd

def log_output(filename, content, head=None):
    with open(filename, "a") as log:
        if head != None:
            print(head + '\n', file=log)
        print(content, file=log)
        

columns = ['rank', 'dt', 'cookie', 'ip', 'mobile_idfa', 'mobile_imei', 'mobile_android_id', 
	      'mobile_openudid', 'mobile_mac', 'timestamps', 'camp_id', 'creativeid', 'mobile_os', 
	      'mobile_type','mobile_app_key', 'mobile_app_name', 'placement_id', 'user_agent', 
	      'media_id', 'os', 'born_time', 'flag']

# initialize data  
# it is faster to use pandas.read_csv to load large dataset
## chunks = pd.read_csv("training_set", iterator=True, delimiter="	", chunksize=100000)
## training_set = pd.concat([chunk for chunk in chunks], ignore_index=True)
## training_set.columns = columns

# proportion of fake access in training set: 0.49
## percentage_fake = fake_access / np.float(line) 

# check missing values
training_size = len(training_set.index)
log_output('log', training_size , header='training size')
## nulls = (training_size - training_set.count()) / training_size
## log_output("log", head="proportion of missing values", content=nulls)

# proportion of different sources in all views
androids = 1 - np.isnan(training_set['mobile_android_id'])
cnt_android = np.sum(androids)

mobile_idfa = 1 - np.isnan(training_set['mobile_idfa'])
mobile_openudid = 1 - np.isnan(training_set['mobile_openudid'])
cnt_ios = np.sum(npmobile_idfa + mobile_openudid)

webs = 1 - np.isnan(training_set['cookie'])
cnt_web = np.sum(webs)

log_output('log', cnt_android, header='androids')
log_output('log', cnt_ios, header='ios')
log_output('log', cnt_web, header='web')


# categorize fake accesses based on source
# proportion of sources in fake access
# The time when most access occurs


# i wanna check if most of fake access originate from a few devices and burst of fake access tend to cluster
# in certain amount of time

	# time and device id 
	# how to encode categorical data to reflect the heirachy

# if yes, repetited access will be a crucial feature for classification
# otherwise, i can randomly subsample data from training

