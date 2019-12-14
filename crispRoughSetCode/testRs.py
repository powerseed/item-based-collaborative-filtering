import pandas as pd
import RS as mrs
from sklearn.model_selection import KFold
import clustering as cluster
df = pd.read_csv('data/fuzzy_rough_set_data/ionosphere.data').drop(columns = ['1'])
df = cluster.cluster_table(df,'class',5)
X = df.drop(columns = ['class']).values
Y = df['class'].values
corrects = []
for i in range(2):
    kf = KFold(n_splits=10)
    for train_id, test_id in kf.split(X):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = Y[train_id], Y[test_id]
        mr = mrs.I_RS()
        mr.fit(X_train,y_train)
        correct = 0
        for j in range(X_test.shape[0]):
            p = mr.predict(X_test[j])
            if p[0] == y_test[j]:
                correct = correct + 1
        corrects.append(correct/X_test.shape[0])# -*- coding: utf-8 -*-

