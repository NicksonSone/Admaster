#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function   # log output
import numpy as np
import pandas as pd

def log_output(filename, content, head=None):
    with open(filename, "a") as log:
        if head != None:
            print('\n## ' + head + '\n', file=log)
        print(content, file=log)


columns = ['rank', 'dt', 'cookie', 'ip', 'mobile_idfa', 'mobile_imei', 'mobile_android_id',
'mobile_openudid', 'mobile_mac', 'timestamps', 'camp_id', 'creativeid', 'mobile_os',
'mobile_type','mobile_app_key', 'mobile_app_name', 'placement_id', 'user_agent',
'media_id', 'os', 'born_time', 'flag']

# initialize data
# it is faster to use pandas.read_csv to load large dataset
chunks = pd.read_csv("training_set", iterator=True, delimiter="	", chunksize=100000)
training_set = pd.concat([chunk for chunk in chunks], ignore_index=True)
training_set.columns = columns

# proportion of fake access in training set: 0.49
## percentage_fake = fake_access / np.float(line)


# check missing values
training_size = len(training_set.index)
log_output('log', training_size , head='training size')
nulls_ratio = (training_size - training_set.count()) / training_size
## log_output("log", head="proportion of missing values", content=nulls_ratio)


# proportion of different sources in all views
# it turns out every access has a cookie value
androids = 1 - pd.isnull(training_set['mobile_android_id'])
cnt_android = np.sum(androids)

mobile_idfa = 1 - pd.isnull(training_set['mobile_idfa'])
mobile_openudid = 1 - pd.isnull(training_set['mobile_openudid'])
cnt_ios = np.sum(mobile_idfa + mobile_openudid)

## log_output('log', cnt_android, head='androids')
## log_output('log', cnt_ios, head='ios')

# count mobile_type, user_agents, cookies
mobile_types = training_set.mobile_type.value_counts()
user_agents = training_set.user_agents.value_counts()

cookies = training_set.cookie.value_counts()
## log_output('log', content=mobile_types, head='mobile types')
## log_output('log', content=user_agent, head='user agents')
## log_output('log', content=cookies, head='cookies')


# extract all flaged access and watch at the  distribution
flaged_access = training_set[training_set['flag'] == 1]

f_ip = flaged_access.ip.value_counts()

f_cookies = flaged_access.cookie.value_counts()

f_user_agents= flaged_access.user_agent.value_counts()

f_mobile_types = flaged_access.mobile_type.value_counts()

f_mobile_os = flaged_access.mobile_os.value_counts()

f_camp_id = flaged_access.camp_id.value_counts()

f_creativeid = flaged_access.creativeid.value_counts()

f_placement_id = flaged_access.placement_id.value_counts()

f_media_id = flaged_access.media_id.value_counts()

## log_output('flaglog', content=f_ip, head='f_ip')
## log_output('flaglog', content=f_cookies, head='cookies')
## log_output('flaglog', content=f_user_agents, head='flaged_user_agents')
## log_output('flaglog', content=f_mobile_types, head='f_mobile_types')
## log_output('flaglog', content=f_mobile_os, head='f_mobile_os')
## log_output('flaglog', content=f_camp_id, head='f_camp_id')
## log_output('flaglog', content=f_creativeid, head='f_creativeid')
## log_output('flaglog', content=f_placement_id, head='f_placement_id')
## log_output('flaglog', content=f_media_id, head='f_media_id')

# 1152.1151.1165.1103 is one of the most flaged ip, a ip address may has several
# cookies
peek = flaged_access.loc[flaged_access.ip ==
'1152.1151.1165.1103', 'cookie']
## log_output('flaglog', content=peek.value_counts()
, head='cookies of a specific flaged ip')


# make flaged cookies and ip persistent

# proportion of sources in fake access
# The time when most access occurs

# i wanna check if most of fake access originate from a few devices and burst of fake access tend to cluster
# in certain amount of time

	# time and device id
	# how to encode categorical data to reflect the heirachy

# if yes, repetited access will be a crucial feature for classification
# otherwise, i can randomly subsample data from training

