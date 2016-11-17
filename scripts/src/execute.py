import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


columns = ['rank', 'dt', 'cookie', 'ip', 'mobile_idfa', 'mobile_imei', 'mobile_android_id',
'mobile_openudid', 'mobile_mac', 'timestamps', 'camp_id', 'creativeid', 'mobile_os',
'mobile_type','mobile_app_key', 'mobile_app_name', 'placement_id', 'user_agent',
'media_id', 'os', 'born_time', 'flag']

print "Loading training set"
training_chunks = pd.read_csv("training_set", iterator=True, delimiter="	",
        chunksize=100000, low_memory=False)
training_set = pd.concat([chunk for chunk in training_chunks], ignore_index=True)
training_set.columns = columns


# mobile types when hits>500, num = 1000
# mobile os 10, 154
# creativeid 10, 88
# placementid 3500, 1000
# media id 10, 154
# os 10, 154
# mobile app key 10, 154
# mobile app name 10, 154


