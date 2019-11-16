import pandas as pd
import numpy as np
import math
from fuzzyRoughSetCode import clustering, TFRS
from fuzzyRoughSetCode.TFRS import I_TFRS

data = pd.read_csv('../data/weather_features.csv')

tfrs = I_TFRS()
tfrs.fit(data[:500], 'weather_main')
reduct = tfrs.reduct_attr
print(reduct)

# for column in data:
#     if ((column != 'weather_main') and (column != 'dt_iso')):
#         data[column] = clustering.cluster(data[column])
#
# print(data['humidity'])
