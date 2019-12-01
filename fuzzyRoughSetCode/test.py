import FRS
import pandas as pd
<<<<<<< Updated upstream
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
=======
df = pd.read_csv('hyperplane30.csv')
Y = df['response'].values
X = df.drop(columns = ['response']).values
imrs = FRS.I_FRS()
imrs.fit(X[:500],Y[:500])
tfmrs = FRS.TFMRS(100,0.9)
tfmrs.fit(X[:500],Y[:500])
test_x = X[500:]
test_y = Y[500:]
correct = [0,0,0,0]
for i in range(100):
>>>>>>> Stashed changes
    y1,cover = imrs.predict(test_x[i])
    if y1 == test_y[i]:
        correct[0] = correct[0] + 1
    y3 = tfmrs.predict(test_x[i])
    if y3 == test_y[i]:
        correct[2] = correct[2] + 1