# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
complex_df = pd.read_csv('train_orders_data_with_targets.csv')
complex_df = complex_df.fillna(0)
s_df = complex_df[['order_id','datetime','food_prep_time_minutes']]
s_df['datetime'] = pd.to_datetime(s_df['datetime'])
s_df['day_of_week'] = s_df['datetime'].dt.day_name()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
s_df['day_of_week'] = pd.Categorical(s_df['day_of_week'], categories=days)

s_df['hour'] = s_df['datetime'].dt.hour + 0.25*s_df['datetime'].dt.quarter
item_max = 10
quantities = []
item_count = []
for i in range(complex_df.shape[0]):
    quantity = 0
    item = 0
    for j in range(item_max):
        col_name = "quantity_{index}".format(index = str(j+1))
        quantity = quantity +  complex_df.loc[i][col_name]
        if complex_df.loc[i][col_name] != 0.0:
            item = item+1
    quantities.append(quantity)
    item_count.append(item)
s_df['quantities'] =  quantities  
s_df['item_count'] = item_count
mat_val = s_df[['day_of_week', 'hour', 'quantities', 'item_count', 'food_prep_time_minutes']]
mat_val.to_csv("data_after_cleaning.csv")
mat = mat_val.values
mat_output = open('trian_mat.npy','wb')
np.save(mat_output,mat)

