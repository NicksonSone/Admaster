import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

rough = pd.read_csv('r.csv')
frst = pd.read_csv('f.csv')

cmbnd = pd.concat([rough, frst])
cmbnd = cmbnd.sort_values('rank')
cmbnd = cmbnd.drop_duplicates()
cmbnd.to_csv("combined.csv", index=False,encoding='utf-8')
