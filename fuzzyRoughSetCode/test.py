import FRS
import pandas as pd
df = pd.read_csv('../crispRoughSetCode/rh.csv')
Y = df['response'].values
X = df.drop(columns = ['response']).values
#imrs = FRS.I_FRS()
#imrs.fit(X[:2000],Y[:2000],cat = [0])
tfmrs = FRS.TFMRS(200,0.95)
tfmrs.fit(X[:2000],Y[:2000])
test_x = X[2000:3000]
test_y = Y[2000:3000]
correct = [0,0,0,0]
for i in range(200):
#   y1,cover = imrs.predict(test_x[i])
#   if y1 == test_y[i]:
#       correct[0] = correct[0] + 1
   y3 = tfmrs.predict(test_x[i])
   if y3 == test_y[i]:
       correct[2] = correct[2] + 1