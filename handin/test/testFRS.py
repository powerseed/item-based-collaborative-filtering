import pandas as pd
import FRS as mrs
import IVFRS as imrs
from sklearn.model_selection import KFold
df = pd.read_csv('data/fuzzy_rough_set_data/vehicle.csv')
X = df.drop(columns = ['Class']).values
Y = df['Class'].values
corrects = []
for i in range(2):
    kf = KFold(n_splits=10)
    for train_id, test_id in kf.split(X):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = Y[train_id], Y[test_id]
        mr = imrs.IVFRS(100)
        mr.fit(X_train,y_train)
        correct = 0
        for j in range(X_test.shape[0]):
            p = mr.predict(X_test[j])
            if p == y_test[j]:
                correct = correct + 1
        corrects.append(correct/X_test.shape[0])