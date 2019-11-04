from sklearn.externals import joblib
import catboost as cb
import numpy as np
catboost_regr = joblib.load('cat_boost.pkl')
mat_input = open('test_mat.npy','rb')
X = np.load(mat_input, allow_pickle = True)
Y = catboost_regr.predict(X)
import pandas as pd
order_id = pd.read_csv('test_orders_data_without_targets.csv')
order_id['food_prep_time_minutes'] = Y
order_id = order_id[['order_id','food_prep_time_minutes']]
order_id.to_csv('NoName_skipthedishes_food_prep_time_prediction_file.csv')