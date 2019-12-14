import pandas as pd
import numpy as np
import ICRS as ic
import IVRS as iv
import ISwCRS as iswc
import ISwVRS as isr
import TFCRS as tfc
import TFVRS as tfv
import clustering
df = pd.read_csv('data/stream_data/covtype.csv')[:50000]
df[df.columns[:10]] = clustering.cluster_table(df[df.columns[:10]],'target',10)
condition = df.drop(columns = ['target']).values
decision = df['target'].values
X = condition[:2000]
Y = decision[:2000]
testX = condition[2000:]
testY = decision[2000:]
corrects_l = []
for batch_size in [50,100,160,200,250]:
    corrects = []
    for fd in range(1,50):
        
        testX = condition[2000:]
        testY = decision[2000:]
        a = tfc.TFCRS(batch_size,fd*0.02)
        d = tfv.TFVRS(batch_size,fd*0.02)
        a.fit(X,Y)
        d.fit(X,Y)
        correct = [0,0]
        i = 1
        while testX.shape[0] >= batch_size:
            print(i)
            i = i + 1
            for j in range(batch_size):
                d1 = a.predict(testX[j])
                if d1 == testY[j]:
                    correct[0] = correct[0] + 1
                d1 = d.predict(testX[j])
                if d1 == testY[j]:
                    correct[1] = correct[1] + 1
            a.update_group(testX[:batch_size],testY[:batch_size])
            d.update_group(testX[:batch_size],testY[:batch_size])
            testX = testX[batch_size:]
            testY = testY[batch_size:]  
        corrects.append(correct)
    corrects_l.append(corrects)
    # -*- coding: utf-8 -*-

