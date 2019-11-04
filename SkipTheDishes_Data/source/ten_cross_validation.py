from sklearn.model_selection import KFold
import catboost as cb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
mat_input = open('train_mat.npy','rb')
mat = np.load(mat_input, allow_pickle = True)
X = mat[:,:-1]
Y = mat[:,-1]
kf = KFold(n_splits=10)
mae=[]
rmse=[]
r2=[]
for train_index, test_index in kf.split(X):
    X_train,X_test = X[train_index],X[test_index]
    Y_train,Y_test = Y[train_index],Y[test_index]
    cat_features = [0]
    regr = cb.CatBoostRegressor(depth = 4, learning_rate = 0.5)   
    regr.fit(X_train,Y_train,cat_features)
    prediction = regr.predict(X_test)
    mae.append(mean_absolute_error(Y_test,prediction))
    rmse.append(np.sqrt(mean_squared_error(Y_test,prediction)))
    r2.append(r2_score(Y_test,prediction))
print("MAE=:{}".format(np.mean(mae)))
print("RMSE=:{}".format(np.mean(rmse)))
print("RMSE=:{}".format(np.mean(r2)))
