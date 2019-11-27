import FRS
import pandas as pd
df = pd.read_csv('csv_result-hyperplane8.csv')
Y = df['output'].values
X = df.drop(columns = ['id','output']).values
imrs = FRS.I_FRS()
imrs.fit(X[:2000],Y[:2000])
tfmrs = FRS.TFMRS(100,0.9)
tfmrs.fit(X[:2000],Y[:2000])
test_x = X[2000:]
test_y = Y[2000:]
correct = [0,0,0,0]
for i in range(500):
    y1,cover = imrs.predict(test_x[i])
    if y1 == test_y[i]:
        correct[0] = correct[0] + 1
    y3 = tfmrs.predict(test_x[i])
    if y3 == test_y[i]:
        correct[2] = correct[2] + 1