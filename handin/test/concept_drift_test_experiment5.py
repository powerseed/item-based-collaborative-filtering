import pandas as pd
import numpy as np
import ICRS as ic
import IVRS as iv
import SwCRS as swc
import SwVRS as swv
import TFCRS as tfc
import TFVRS as tfv
import clustering
batch_size = 50
fading_factor = 0.95
window_size = 20
n_cluster = 10


#####################circle###########################
df = pd.read_csv('data/concept_data/circle.csv')
df = clustering.cluster_table(df,'class',n_cluster)
condition = df.drop(columns = ['class']).values
decision = df['class'].values
#####################led############################

#df = pd.read_csv('data/concept_data/led.csv')
#condition = df.drop(columns = ['class']).values
#decision = df['class'].values

#####################sin#############################

#df = pd.read_csv('data/concept_data/sin.csv')
#df = clustering.cluster_table(df,'class',n_cluster)
#condition = df.drop(columns = ['class']).values
#decision = df['class'].values

#####################stagger#############################

#df = pd.read_csv('data/concept_data/stagger.csv')
#condition = df.drop(columns = ['class']).values
#decision = df['class'].values

#######################################################

X = condition[:2000]
Y = decision[:2000]
testX = condition[2000:]
testY = decision[2000:]
testX = condition[2000:]
testY = decision[2000:]

a = swc.SwCRS(batch_size,window_size)
b = tfc.TFCRS(batch_size,fading_factor)
c = ic.ICRS(batch_size)
d = swv.SwVRS(batch_size,window_size)
e = tfv.TFVRS(batch_size,fading_factor)
f = iv.IVRS(batch_size)
a.fit(X,Y)
b.fit(X,Y)
c.fit(X,Y)
d.fit(X,Y)
e.fit(X,Y)
f.fit(X,Y)
correct = [0,0,0,0,0,0]
i = 1
while testX.shape[0] >= batch_size:
    print(i)
    i = i + 1
    for j in range(batch_size):
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
    a.update_group(testX[:batch_size],testY[:batch_size])
    b.update_group(testX[:batch_size],testY[:batch_size])
    c.update_group(testX[:batch_size],testY[:batch_size])
    d.update_group(testX[:batch_size],testY[:batch_size])
    e.update_group(testX[:batch_size],testY[:batch_size])
    f.update_group(testX[:batch_size],testY[:batch_size])
    testX = testX[batch_size:]
    testY = testY[batch_size:]  
    
print(correct)     
    
    
    