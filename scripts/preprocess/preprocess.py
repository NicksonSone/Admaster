#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import csv as csv
import numpy as np
import pandas as pd

train = pd.read_csv('./data/training_set.csv')

log_column = ['rank', 'dt', 'cookie', 'ip', 'mobile_idfa', 'mobile_imei', 'mobile_android_id', 
	      'mobile_openudid', 'mobile_mac', 'timestamps', 'camp_id', 'creativeid', 'mobile_os', 
	      'mobile_type','mobile_app_key', 'mobile_app_name', 'placement_id', 'user_agent', 
	      'media_id', 'os', 'born_time', 'flag']

media_column = ['mediaid', 'media_id', 'Category', 'firstType_cn', 'secondType_cn', 'tag'] 

# insert code to get a list of categorical columns into a variable say categorical_columns
# insert code to take care of the missing values in the columns in whatever way you like to
# but is is important that missing values are replaced.

# Get the categorical values into a 2D numpy array
train_categorical_values = np.array(train[categorical_columns])

# OneHotEncoder will only work on integer categorical values, so if you have strings in your
# categorical columns, you need to use LabelEncoder to convert them first

# do the first column
enc_label = LabelEncoder()
train_data = enc_label.fit_transform(train_categorcial_values[:,0])

# do the others
for i in range(1, train_categorical_values.shape[1]):
	enc_label = LabelEncoder()
	train_data = np.column_stack((train_data, enc_label.fit_transform(train_categorical_values[:,i])))

train_categorical_values = train_data.astype(float)

# if you have only integers then you can skip the above part from do the first column and uncomment the following line
# train_categorical_values = train_categorical_values.astype(float) 

enc_onehot = OneHotEncoder()
train_cat_data = enc.fit_transform(train_categorical_values)

# play around and print enc.n_values_ features_indices_ to see how many unique values are there in each column

# create a list of columns to help create a DF from np array
# so say if you have col1 and col2 as the categorical columns with 2 and 3 unique values respectively. The following code
# will give you col1_0, col1_1, col2_1,col2_2,col2_3 as the columns

cols = [categorical_columns[i] + '_' + str(j) for i in range(0,len(categorical_columns)) for j in range(0,enc.n_values_[i]) ]
train_cat_data_df = pd.DataFrame(train_cat_data.toarray(),columns=cols)

# get this columns back into the data frame
train[cols] = train_cat_data_df[cols]

# append the target column. Obviously rename it to whatever is your target column 
cols.append('target')
# So now you have a dataframe with only the categorical columns and the target. You can now do whatever you want to do with it :)
train_cat_df = train[cols]
