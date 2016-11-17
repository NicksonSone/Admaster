import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


# Load media info
print 'Loading media info...'
media_info = pd.read_csv('ccf_media_info.csv')

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
test_columns = ['rank', 'dt', 'cookie', 'ip', 'mobile_idfa', 'mobile_imei', 'mobile_android_id',
'mobile_openudid', 'mobile_mac', 'timestamps', 'camp_id', 'creativeid', 'mobile_os',
'mobile_type','mobile_app_key', 'mobile_app_name', 'placement_id', 'user_agent',
'media_id', 'os', 'born_time']

print "Loading test set"
test_chunks = pd.read_csv("test_set", iterator=True, delimiter="	", chunksize=100000, 
        low_memory=False)
test_set = pd.concat([chunk for chunk in test_chunks], ignore_index=True)
test_set.columns = test_columns


# TODO: media_id has not been used
# Preporcess training set
cat_cols = ['media_id', 'camp_id', 'creativeid', 'mobile_os', 'mobile_type','mobile_app_key', 
'mobile_app_name', 'placement_id', 'user_agent','os', ]

# Fill missing value
training_set[cat_cols] = training_set[cat_cols].fillna('NaN')
test_set[cat_cols] = test_set[cat_cols].fillna('NaN')

print "preprocessing categorical values: training set"
labelEncoders = {col:LabelEncoder() for col in cat_cols}
cat_df_train = pd.DataFrame() 
for col in cat_cols:
    if col == 'media_id':
        labelEncoders[col].fit(media_info['id'])
        cat_df_train[col] = labelEncoders[col].transform(training_set[col]) 
    else:
        concat = pd.concat([ training_set[col], test_set[col] ])
        labelEncoders[col].fit(concat)
        cat_df_train[col] = labelEncoders[col].transform(training_set[col])

    cat_df_train[col] = cat_df_train[col].astype('category')


##  cat_dict_train = cat_df_train.T.to_dict().values()
##  
##  print 'transforming....'
##  vectorizer = DictVectorizer(sparse=False)
#  vec_x_cat_train = vectorizer.fit_transform(cat_dict_train)  


# Save processed training set
# fit_transform return a numpy.ndarray data, use numpy.savetxt() to save data
## np.savetxt("processed_categories_training_set,csv", vec_x_cat_train,
## delimiter="	")


# Build random forest
print "constructing tree"
rf = RandomForestClassifier(n_estimators=20)
rf.fit(cat_df_train, training_set['flag'])

# Save model
joblib.dump(rf, './randomForestModel/randomForest.pkl')
# To load a model use: model = joblib.load('filename.pkl') 

# Release training set memory
## del cat_df_train; del training_set
## gc.collect()


# Preporcess test set
print "preprocessing categorical values: test set"
cat_df_test = pd.DataFrame()
for col in cat_cols:
    cat_df_test[col] = labelEncoders[col].transform(test_set[col])
    cat_df_test[col] = cat_df_test[col].astype('category')


## cat_dict_test = cat_df_test.T.to_dict().values()
## 
## print 'transforming....'
## vec_x_cat_test = vectorizer.fit_transform(cat_dict_test)


# Save processed test set
## np.savetxt("processed_categories_test_set,csv", vec_x_cat_test,
## delimiter="	")


# Predict
print "predicting"
y = rf.predict(cat_df_test)
print "saving"
output = test_set.loc[y == 1, 'rank']
output.to_csv("output2.csv", index=False, encoding='utf-8')
