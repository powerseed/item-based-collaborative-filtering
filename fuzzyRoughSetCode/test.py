import TFMRS as tf
import IVMRS as iv
import ISwVMRS as sw
import pandas as pd
df = pd.read_csv('../crispRoughSetCode/data/em.csv')
Y = df['label'].values
X = df.drop(columns = ['label']).values
cat = []
a = tf.TFMRS(200,0.95)
b = iv.IVMRS(200)
c = sw.ISwMRS(200,15)
a.fit(X[:2000],Y[:2000],cat=cat)
b.fit(X[:2000],Y[:2000],cat=cat)
c.fit(X[:2000],Y[:2000],cat=cat)
testX = X[2000:10000]
testY = Y[2000:10000]
correct = [0,0,0,0]
i = 1
while testX.shape[0] >= 200:
    print(i)
    i = i + 1
    for j in range(200):
        d1 = a.predict(testX[j])
        if d1 == testY[j]:
            correct[0] = correct[0]+1
        d2 = b.predict(testX[j])
        if d2 == testY[j]:
            correct[1] = correct[1]+1
        d3 = c.predict(testX[j])
        if d3 == testY[j]:
            correct[2] = correct[2]+1
    a.update_group(testX[:200],testY[:200])
    b.update_group(testX[:200],testY[:200])
    c.update_group(testX[:200],testY[:200])
    testX = testX[200:]
    testY = testY[200:]
