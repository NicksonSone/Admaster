import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score


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

# Preporcess test set
print "preprocessing categorical values: test set"
cat_df_test = pd.DataFrame()
for col in cat_cols:
    if col == 'media_id':
        cat_df_test[col] = test_set[col]
    else:
        cat_df_test[col] = labelEncoders[col].transform(test_set[col])
        cat_df_test[col] = cat_df_test[col].astype('category')
# join categorical values of test set with media info
cat_df_test = cat_df_test.join(media_info.set_index('id'), on='media_id')
cat_df_test.drop('media_id', axis=1, inplace=True)

for col in media_cols:
    if col == 'id':
        continue
    else:
        cat_df_train[col] = cat_df_train[col].astype('category')
        cat_df_test[col] = cat_df_test[col].astype('category')


# Build models 
num_estimators = [3, 5, 7, 10, 11, 15, 30]
classifiers = {}
for estimator in num_estimators:
    print "constructing tree " + str(estimator) + "..."
    classifiers[estimator] = RandomForestClassifier(n_estimators=estimator,
            oob_score=True)
    classifiers[estimator].fit(cat_df_train, training_set['flag'])

    
    # Predict & Save results
    print "Predicting..."
    y = classifiers[estimator].predict(cat_df_test)
    print "Saving predictions..."
    output = test_set.loc[y == 1, 'rank']
    output.to_csv("full_fea"+str(estimator)+'.csv', header=False, index=False, encoding='utf-8')


    # Save model
    # To load a model use: model = joblib.load('filename.pkl') 
    print "Model persistence..."
    joblib.dump(classifiers[estimator], './full_fea'+str(estimator)+'/randomForest.pkl')


# Cross evaluate
for estimator in num_estimators:
    print(cross_val_score(classifiers[estimator], cat_df_train,
        training_set['flag']))

# Save processed training set
cat_df_train.to_csv('cat_df_train.csv', index=False, encoding='utf-8')
# Save processed test set
cat_df_test.to_csv('cat_df_test.csv', index=False, encoding='utf-8')


# mobile types when hits>500, num = 1000
# mobile os 10, 154
# creativeid 10, 88
# placementid 3500, 1000
# media id 10, 154
# os 10, 154
# mobile app key 10, 154
# mobile app name 10, 154


