import pandas as pd
import numpy as np
#import ICRS as ic
#import IVRS as iv
#import ISwCRS as iswc
#import ISwVRS as isr
#import TFCRS as tfc
#import TFVRS as tfv
import clustering
import IRS as irs
df = pd.read_csv('data/electricity_csv.csv')
df = df.drop(columns=['date'])
df = clustering.cluster_table(df,'class',10, cat = ['day'])
decision = df['class'].values
condition = df.drop(columns=['class']).values
X = condition[:2000]
Y = decision[:2000]
testX = condition[2000:10000]
testY = decision[2000:10000]
l = np.where(testY == np.unique(testY)[0])[0]
print(len(l))
a = irs.I_RS()
a.fit(X,Y)
#a = ic.ICRS(200)
#a = iswc.ISwCRS(200,15)
#a = tfc.TFCRS(200,0.95)
#a = iv.IVRS(200)
#a = isr.ISwVRS(200,15)
#a = tfv.TFVRS(200,0.95)
#a.fit(X,Y)
correct = [0,0,0,0,0,0]
d_l = []
unknow = 0
i = 1
while testX.shape[0] >= 200:
    print(i)
    i = i + 1
    for j in range(200):

        d1,_ = a.predict(testX[j])
        if d1 == testY[j]:
            correct[0] = correct[0]+1
    a.update_group(testX[:200],testY[:200])
    testX = testX[200:]
    testY = testY[200:]
