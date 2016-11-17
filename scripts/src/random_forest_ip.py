import numpy as np
import pandas as pd

tree = pd.read_csv('full_fea10.csv', header=None)
ip = pd.read_csv('ip.csv', header=None)

cmbnd = pd.concat([tree, ip])
cmbnd = cmbnd.sort_values(0)
cmbnd = cmbnd.drop_duplicates()
cmbnd.to_csv("fea10_ip.csv",header=False,  index=False,encoding='utf-8')
