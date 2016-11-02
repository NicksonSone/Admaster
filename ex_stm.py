log_output('flaglog', content=peek.value_counts()
, head='cookies of a specific flaged ip')

ip = training_set.ip.value_counts()

log_output('log', content=ip, head='ip')

# relationship between ip and cookie

