import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score


# Load media info
print 'Loading media info...'
media_info = pd.read_csv('ccf_media_info.csv')
media_cols = media_info.columns
# Encode media info
enc = LabelEncoder()
for col in media_cols:
    if col == 'id':
        continue
    media_info[col] = enc.fit_transform(media_info[col])
    media_info[col] = media_info[col].astype('category')


# Load training set
# it is faster to use pandas.read_csv to load large dataset
columns = ['rank', 'dt', 'cookie', 'ip', 'mobile_idfa', 'mobile_imei', 'mobile_android_id',
'mobile_openudid', 'mobile_mac', 'timestamps', 'camp_id', 'creativeid', 'mobile_os',
'mobile_type','mobile_app_key', 'mobile_app_name', 'placement_id', 'user_agent',
'media_id', 'os', 'born_time', 'flag']

print "Loading training set"
training_chunks = pd.read_csv("training_set", iterator=True, delimiter="	",
        chunksize=100000, low_memory=False)
training_set = pd.concat([chunk for chunk in training_chunks], ignore_index=True)
training_set.columns = columns



# Load test set & release training set
test_columns = columns[:]; test_columns.remove('flag')
print "Loading test set"
test_chunks = pd.read_csv("test_set", iterator=True, delimiter="	", chunksize=100000, 
        low_memory=False)
test_set = pd.concat([chunk for chunk in test_chunks], ignore_index=True)
test_set.columns = test_columns


# Preporcess training set
cat_cols = ['media_id', 'camp_id', 'creativeid', 'mobile_os', 'mobile_type','mobile_app_key', 
'mobile_app_name', 'placement_id', 'user_agent','os', ]
# Fill missing value
for col in cat_cols:
    training_set[col].fillna('NaN', inplace=True)
    test_set[col].fillna('NaN', inplace=True)

print "preprocessing categorical values: training set"
labelEncoders = {col:LabelEncoder() for col in cat_cols}
cat_df_train = pd.DataFrame() 
for col in cat_cols:
    if col == 'media_id':
        cat_df_train[col] = training_set[col]
    else:
        concat = pd.concat([ training_set[col], test_set[col] ])
        labelEncoders[col].fit(concat)
        cat_df_train[col] = labelEncoders[col].transform(training_set[col])

    cat_df_train[col] = cat_df_train[col].astype('category')
# join categorical values of training set with media info
cat_df_train = cat_df_train.join(media_info.set_index('id'), on='media_id')
cat_df_train.drop('media_id', axis=1, inplace=True)


# Get true y values
y_true = training_set['flag']

# Load model
clf5 = joblib.load('full_fea5/randomForest.pkl')

# Predict y values
y_pred = clf5.predict(cat_df_train)

# f1 score
f1_score(y_true, y_pred, average='binary')
f1_score(y_true, y_pred, average='macro')
f1_score(y_true, y_pred, average='micro')
f1_score(y_true, y_pred, average='weighted')
f1_score(y_true, y_pred, average=None)


