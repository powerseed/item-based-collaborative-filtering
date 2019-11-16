import pandas as pd
import numpy as np
import math
from fuzzyRoughSetCode import clustering, TFRS
from fuzzyRoughSetCode.TFRS import I_TFRS

data = pd.read_csv('../data/weather_features.csv')

for column in data:
    if ((column != 'weather_main') and (column != 'dt_iso')):
        data[column] = clustering.cluster(data[column]).astype(str)

# print(data['temp'])
# data.to_csv(r'C:\4710project\item-based-collaborative-filtering\data\weather_clustered.csv',
#                                       index=None, header=True)

tfrs = I_TFRS()
tfrs.fit(data[:500], 'weather_main')
reduct = tfrs.reduct_attr
print(reduct)

#
# print(data['humidity'])
