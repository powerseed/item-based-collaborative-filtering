import TFMRS as tf
import IVMRS as iv
import ISwVMRS as sw
import pandas as pd
df = pd.read_csv('../crispRoughSetCode/data/stream_data/movingRBF.csv')
#df = df.drop(columns=['date'])
Y = df['target'].values
X = df.drop(columns = ['target']).values
#a = tf.TFMRS(60,0.95)
#a = iv.IVMRS(100)
a = sw.ISwMRS(100,15)
a.fit(X[:2000],Y[:2000])
#b.fit(X[:2000],Y[:2000])
#c.fit(X[:2000],Y[:2000])
testX = X[2000:10000]
testY = Y[2000:10000]
correct = [0,0,0,0]
i = 1
while testX.shape[0] >= 60:
    print(i)
    i = i + 1
    for j in range(60):
        d1 = a.predict(testX[j])
        if d1 == testY[j]:
            correct[0] = correct[0]+1
#        d2 = b.predict(testX[j])
#        if d2 == testY[j]:
#            correct[1] = correct[1]+1
#        d3 = c.predict(testX[j])
#        if d3 == testY[j]:
#            correct[2] = correct[2]+1
    a.update_group(testX[:60],testY[:60])
#    b.update_group(testX[:100],testY[:100])
#    c.update_group(testX[:100],testY[:100])
    testX = testX[60:]
    testY = testY[60:]
