import pandas as pd
import numpy as np
import ICRS as ic
import IVRS as iv
import ISwCRS as iswc
import ISwRS as isr
import TFCRS as tfc
import TFVRS as tfv
import clustering
import IRS1 as irs
df = pd.read_csv('data/electricity_csv.csv')
df = df.drop(columns=['date'])
df = clustering.cluster_table(df,'class',10, cat = ['day'])
decision = df['class'].values
condition = df.drop(columns=['class']).values
X = condition[:2000]
Y = decision[:2000]
testX = condition[2000:]
testY = decision[2000:]
i1 = irs.I_RS()
i1.fit(X,Y)
#a = ic.ICRS(200)
#b = iswc.ISwCRS(200,15)
#c = tfc.TFCRS(200,0.95)
#d = iv.IVRS(200)
#e = isr.ISwRS(200,15)
#f = tfv.TFVRS(200,0.95)
#a.fit(X,Y)
#b.fit(X,Y)
#c.fit(X,Y)
#d.fit(X,Y)
#e.fit(X,Y)
#f.fit(X,Y)
correct = [0,0,0,0,0,0]
d_l = []
unknow = 0
i = 1
while i < 5:
    print(i)
    i = i + 1
    for j in range(200):
        d = i1.predict(testX[j])
        d_l.append(d)
        if d == testY[j]:
            correct[0] = correct[0]+1
#        d1 = a.predict(testX[j])
#        if d1 == testY[j]:
#            correct[0] = correct[0]+1
#        d2 = b.predict(testX[j])
#        if d2 == testY[j]:
#            correct[1] = correct[1]+1
#        d3 = c.predict(testX[j])
#        if d3 == testY[j]:
#            correct[2] = correct[2]+1
#        d4 = d.predict(testX[j])
#        if d4 == testY[j]:
#            correct[3] = correct[3]+1
#        d5 = e.predict(testX[j])
#        if d5 == testY[j]:
#            correct[4] = correct[4]+1
#        d6 = f.predict(testX[j])
#        if d6 == testY[j]:
#            correct[5] = correct[5]+1
    i1.update_group(testX[:200],testY[:200])
#    a.update_group(testX[:200],testY[:200])
#    b.update_group(testX[:200],testY[:200])
#    c.update_group(testX[:200],testY[:200])
#    d.update_group(testX[:200],testY[:200])
#    e.update_group(testX[:200],testY[:200])
#    f.update_group(testX[:200],testY[:200])
    testX = testX[200:]
    testY = testY[200:]
