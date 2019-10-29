import pandas as pd
import numpy as np
df = pd.read_csv('menu_items_details_complete.csv')
description = df['directions'].values
keyword = ['hours', 'hour', 'minutes', 'minute', 'seconds' , 'second'
           'hours,', 'hour,', 'minutes,', 'minute,', 'seconds,' , 'second,'
           'hours.', 'hour.', 'minutes.', 'minute.', 'seconds.' , 'second.']
timeDescribe = []
for i in range(description.shape[0]):
    line = str(description[i]).split()
    token = []
    for j in range(len(line)):
        if line[j] in keyword:
            token.append([line[j-1], line[j]])
    timeDescribe.append(token)

for i in range(len(timeDescribe)):
    for j in range(len(timeDescribe[i])):
        if 'min' in timeDescribe[i][j][1]:
            timeDescribe[i][j][1] = 60
        elif 'sec' in timeDescribe[i][j][1]:
            timeDescribe[i][j][1] = 1
        elif 'hou' in timeDescribe[i][j][1]:
            timeDescribe[i][j][1] = 3600
        if 'ty' in timeDescribe[i][j][0]:
            token = timeDescribe[i][j][0].split('ty')
            timeDescribe[i][j][1]*=10
        elif 'sev' in timeDescribe[i][j][0]:
            timeDescribe[i][j][1]*=5
            
        
