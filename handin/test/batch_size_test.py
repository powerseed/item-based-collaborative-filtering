import pandas as pd
import numpy as np
import ICRS as ic
import IVRS as iv
import SwCRS as iswc
import SwVRS as isr
import TFCRS as tfc
import TFVRS as tfv
import clustering
df = pd.read_csv('data/stream_data/hyperplane20.csv')
df = clustering.cluster_table(df,'response',10)
condition = df.drop(columns = ['response']).values
decision = df['response'].values
X = condition[:2000]
Y = decision[:2000]
testX = condition[2000:]
testY = decision[2000:]
corrects = []
for size in range(20,500,10):
    testX = condition[2000:10000]
    testY = decision[2000:10000]
    a = iswc.SwCRS(size,20)
    b = tfc.TFCRS(size,0.95)
    d = isr.SwVRS(size,20)
    e = tfv.TFVRS(size,0.95)
    a.fit(X,Y)
    b.fit(X,Y)
    d.fit(X,Y)
    e.fit(X,Y)
    correct = [0,0,0,0]
    i = 1
    while testX.shape[0] > 0:
        print(i)
        i = i + 1
        for j in range(min(size,testX.shape[0])):
            d1 = a.predict(testX[j])
            if d1 == testY[j]:
                correct[0] = correct[0] + 1
            d1 = b.predict(testX[j])
            if d1 == testY[j]:
                correct[1] = correct[1] + 1
            d1 = d.predict(testX[j])
            if d1 == testY[j]:
                correct[2] = correct[2] + 1
            d1 = e.predict(testX[j])
            if d1 == testY[j]:
                correct[3] = correct[3] + 1
        a.update_group(testX[:min(size,testX.shape[0])],testY[:min(size,testX.shape[0])])
        b.update_group(testX[:min(size,testX.shape[0])],testY[:min(size,testX.shape[0])])
        d.update_group(testX[:min(size,testX.shape[0])],testY[:min(size,testX.shape[0])])
        e.update_group(testX[:min(size,testX.shape[0])],testY[:min(size,testX.shape[0])])
        testX = testX[min(size,testX.shape[0]):]
        testY = testY[min(size,testX.shape[0]):]  
    corrects.append(correct)
    
    
    
    # -*- coding: utf-8 -*-

