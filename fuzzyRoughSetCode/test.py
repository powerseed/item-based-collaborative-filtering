import FRS
import pandas as pd
df = pd.read_csv('electricity_csv.csv')
Y = df['class'].values
X = df.drop(columns = ['class']).values
imrs = FRS.I_FRS()
imrs.fit(X[:2000],Y[:2000],cat=[0])
tfmrs = FRS.TFMRS(200,0.7)
tfmrs.fit(X[:2000],Y[:2000],cat=[0])
test_x = X[2000:]
test_y = Y[2000:]
correct = [0,0,0,0]
for i in range(1000):
    y1,cover = imrs.predict(test_x[i])
    if y1 == test_y[i]:
        correct[0] = correct[0] + 1
    y3 = tfmrs.predict(test_x[i])
    if y3 == test_y[i]:
        correct[2] = correct[2] + 1