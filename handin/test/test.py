import pandas as pd
import numpy as np
import ICRS as ic
import IVRS as iv
import SwCRS as iswc
import SwVRS as isr
import TFCRS as tfc
import TFVRS as tfv
import clustering
df = pd.read_csv('data/concept_data/circle.csv')
df = clustering.cluster_table(df,'class',15)
condition = df.drop(columns = ['class']).values
decision = df['class'].values
X = condition[:2000]
Y = decision[:2000]
testX = condition[2000:]
testY = decision[2000:]

testX = condition[2000:]
testY = decision[2000:]
a = iswc.SwCRS(50,20)
b = tfc.TFCRS(50,0.95)
c = iv.IVRS(50)
d = isr.SwVRS(50,20)
e = tfv.TFVRS(50,0.95)
f = ic.ICRS(50)
a.fit(X,Y)
b.fit(X,Y)
c.fit(X,Y)
d.fit(X,Y)
e.fit(X,Y)
f.fit(X,Y)
correct = [0,0,0,0,0,0]
i = 1
while testX.shape[0] >= 20:
    print(i)
    i = i + 1
    for j in range(20):
        d1 = a.predict(testX[j])
        if d1 == testY[j]:
            correct[0] = correct[0] + 1
        d1 = b.predict(testX[j])
        if d1 == testY[j]:
            correct[1] = correct[1] + 1
        d1 = c.predict(testX[j])
        if d1 == testY[j]:
            correct[2] = correct[2] + 1
        d1 = d.predict(testX[j])
        if d1 == testY[j]:
            correct[3] = correct[3] + 1
        d1 = e.predict(testX[j])
        if d1 == testY[j]:
            correct[4] = correct[4] + 1
        d1 = f.predict(testX[j])
        if d1 == testY[j]:
            correct[5] = correct[5] + 1
    a.update_group(testX[:20],testY[:20])
    b.update_group(testX[:20],testY[:20])
    c.update_group(testX[:20],testY[:20])
    d.update_group(testX[:20],testY[:20])
    e.update_group(testX[:20],testY[:20])
    f.update_group(testX[:20],testY[:20])
    testX = testX[20:]
    testY = testY[20:]  
    
    
    
    
    