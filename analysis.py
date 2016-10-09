import csv as csv 
import numpy as np

# initialize data
file = csv.reader(open("train"), delimiter="	")
line = 500000
data = []

for i in xrange(line):
	row = file.next()
	data.append(row)
data = np.array(data)

# proportion of fake access
fake_access = np.sum(data[0::, 21].astype(np.float)) # the last field is 21. 1 for fake access, 0 for normal one
proportion_fake = fake_access / np.float(line) # the proportion is 0.0324759998
