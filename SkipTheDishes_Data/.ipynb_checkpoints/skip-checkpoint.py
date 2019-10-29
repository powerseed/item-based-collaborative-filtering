# -*- coding: utf-8 -*-
import pandas as pd
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
        quantity = quantity + complex_df.loc[i][col_name]
        if complex_df.loc[i][col_name] != 0.0:
            item = item+1
    quantities.append(quantity)
    item_count.append(item)
s_df['quantities'] =  quantities  
s_df['item_count'] = item_count
mat = s_df.values



from matplotlib import pyplot as plt
plt.figure()
s_df.groupby('day_of_week').count()['order_id'].plot(kind='bar')
plt.show()
plt.figure()
s_df.groupby('day_of_week').sum()['quantities'].plot(kind='bar')
plt.show()
plt.figure()
s_df.groupby('day_of_week').mean()['food_prep_time_minutes'].plot(kind='bar')
plt.show()

plt.figure()
s_df.groupby('hour').count()['order_id'].plot(kind='bar')
plt.show()
plt.figure()
s_df.groupby('hour').sum()['quantities'].plot(kind='bar')
plt.show()
plt.figure()
s_df.groupby('hour').mean()['food_prep_time_minutes'].plot(kind='bar')
plt.show()
plt.figure()
s_df.groupby('quantities').mean()['food_prep_time_minutes'].plot(kind='bar')
plt.show()
plt.figure()
s_df.groupby('quantities').var()['food_prep_time_minutes'].plot(kind='bar')
plt.show()
plt.figure()
s_df.groupby('item_count').mean()['food_prep_time_minutes'].plot(kind='bar')
plt.show()
plt.figure()
s_df.groupby('item_count').var()['food_prep_time_minutes'].plot(kind='bar')
plt.show()
