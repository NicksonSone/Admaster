import numpy as np
import pandas as pd

test_chunks = pd.read_csv("test_set", iterator=True, delimiter="	", chunksize=100000)
test_set = pd.concat([chunk for chunk in test_chunks], ignore_index=True)
test_set.columns = columns

ip_index = test_set.ip.isin(flaged_access.ip)

