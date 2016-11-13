import numpy as np
import pandas as pd

# this is the best result I can get by just memorizing all the hard details of
# abnormal accesses. This algorithm is for fun and does not generalize well


test_columns = ['rank', 'dt', 'cookie', 'ip', 'mobile_idfa', 'mobile_imei', 'mobile_android_id',
'mobile_openudid', 'mobile_mac', 'timestamps', 'camp_id', 'creativeid', 'mobile_os',
'mobile_type','mobile_app_key', 'mobile_app_name', 'placement_id', 'user_agent',
'media_id', 'os', 'born_time']

# load test set
test_chunks = pd.read_csv("test_set", iterator=True, delimiter="	", chunksize=100000, low_memory=False)
test_set = pd.concat([chunk for chunk in test_chunks], ignore_index=True)
test_set.columns = test_columns

# filter by ip, cookie, idfa, imei, android_id openudid, mac
ip_index = test_set.ip.isin(flaged_access.ip)
cookie_index = test_set.cookie.isin(flaged_access.cookie)

f_mac = flaged_access[flaged_access.mobile_mac .notnull()]
f_idfa = flaged_access[flaged_access.mobile_idfa.notnull()]
f_imei = flaged_access[flaged_access.mobile_imei.notnull()]
f_openudid = flaged_access[flaged_access.mobile_openudid.notnull()]
f_android_id = flaged_access[flaged_access.mobile_android_id.notnull()]

mac_index = test_set.mobile_mac.isin(f_mac)
idfa_index = test_set.mobile_idfa.isin(f_idfa)
imei_index = test_set.mobile_imei.isin(f_imei)
openudid_index = test_set.mobile_openudid.isin(f_openudid)
android_id_index = test_set.mobile_android_id.isin(f_android_id)

# classify test set
flaged = ip_index | cookie_index | idfa_index | imei_index | android_id_index \
       | openudid_index | mac_index
## flaged = cookie_index | ip_index
output = test_set.loc[flaged, 'rank']
output.to_csv("output2.csv", index=False, encoding='utf-8')
