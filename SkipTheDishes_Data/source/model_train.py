import catboost as cb
import numpy as np
mat_input = open('train_mat.npy','rb')
mat = np.load(mat_input, allow_pickle = True)
X = mat[:,:-1]
Y = mat[:,-1]
cat_features = [0]
regr = cb.CatBoostRegressor(depth = 4, learning_rate = 0.5)   
regr.fit(X,Y,cat_features)
from sklearn.externals import joblib
joblib.dump(regr,'cat_boost.pkl')

