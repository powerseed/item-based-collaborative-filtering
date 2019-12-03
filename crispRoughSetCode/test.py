import pandas as pd
import numpy as np
import IRS2 as irs
import IRS1 as ir1
import clustering
def test(X,Y,testX,testY,batch_size,window_size,fading_factor):
#    i = irs.I_RS()
#    i.fit(X,Y)
    lm = irs.VIRS(batch_size)
    lm.fit(X,Y)
    tf = irs.TFVRS(batch_size,fading_factor)
    tf.fit(X,Y)
    sw = irs.ISwRS(batch_size,window_size)
    sw.fit(X,Y)
    correct = [0,0,0,0]
    for j in range(testX.shape[0]):
        d1 = tf.predict(testX[j])
        if d1 == testY[j]:
            correct[0] = correct[0] + 1
        d2 = lm.predict(testX[j])
        if d2 == testY[j]:
            correct[1] = correct[1] + 1        
        d3 = sw.predict(testX[j])
        if d3 == testY[j]:
            correct[2] = correct[2] + 1
#        t,_ = i.predict(testX[j])
#        if t == testY[j]:
#            correct[3] = correct[3] + 1
    for i in range(len(correct)):
        correct[i] = correct[i]/testY.shape[0]
    return correct

df = pd.read_csv('electricity_csv.csv')
df = df.drop(columns=['date'])
df = clustering.cluster_table(df,'class',10,cat = ['day'])
decision = df['class'].values
condition = df.drop(columns=['class']).values
X = condition[:2000]
Y = decision[:2000]
testX = condition[2000:]
testY = decision[2000:]
lm = irs.VIRS(200)
lm.fit(X,Y)
lm2 = ir1.ICRS(300)
lm2.fit(X,Y)
#tf = irs.TFVRS(300,0.9)
#tf.fit(X,Y)
#sw = irs.ISwRS(300,15)
#sw.fit(X,Y)
correct = [0,0,0,0]
unknow = 0
i = 1
while testX.shape[0]>200:
    #print(i)
    i = i + 1
    for j in range(200):
#        d1 = tf.predict(testX[j])
#        if d1 == testY[j]:
#            correct[0] = correct[0] + 1 
        d2 = lm.predict(testX[j])
        if d2 == testY[j]:
            correct[1] = correct[1] + 1   
        if d2 == 'Unknown':
            unknow = unknow + 1
#        d3 = sw.predict(testX[j])
#        if d3 == testY[j]:
#            correct[2] = correct[2] + 1
        d4 = lm2.predict2(testX[j])
        if d4 == testY[j]:
            correct[3] = correct[3] + 1
    lm.update_group(testX[:200],testY[:200])
#    tf.update_group(testX[:200],testY[:200])
#    sw.update_group(testX[:200],testY[:200])
    lm2.update_group(testX[:200],testY[:200])
    testX = testX[200:]
    testY = testY[200:]
