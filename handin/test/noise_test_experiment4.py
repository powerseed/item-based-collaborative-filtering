import pandas as pd
import numpy as np
import ICRS as ic
import IVRS as iv
import RS as rs
import clustering
batch_size = 50
n_cluster = 10
#####################hyperplane 10#############################

#df = pd.read_csv('data/noise_data/hyperplane_noise_0.01_10.csv')[:2500]
#df = clustering.cluster_table(df,'response',n_cluster)
#condition = df.drop(columns = ['response']).values
#decision = df['response'].values

#####################hyperplane 20#############################

df = pd.read_csv('data/noise_data/hyperplane_noise_0.02_10.csv')[:2500]
df = clustering.cluster_table(df,'response',n_cluster)
condition = df.drop(columns = ['response']).values
decision = df['response'].values

#####################hyperplane 20#############################

#df = pd.read_csv('data/noise_data/hyperplane_noise_0.05_10.csv')[:2500]
#df = clustering.cluster_table(df,'response',n_cluster)
#condition = df.drop(columns = ['response']).values
#decision = df['response'].values

#####################hyperplane 20#############################

#df = pd.read_csv('data/noise_data/hyperplan_noise_0.1_10.csv')[:2500]
#df = clustering.cluster_table(df,'response',n_cluster)
#condition = df.drop(columns = ['response']).values
#decision = df['response'].values

#####################hyperplane 20#############################

#df = pd.read_csv('data/noise_data/hyperplane_noise_0.2_10.csv')[:2500]
#df = clustering.cluster_table(df,'response',n_cluster)
#condition = df.drop(columns = ['response']).values
#decision = df['response'].values

################################################################

X = condition[:2000]
Y = decision[:2000]
testX = condition[2000:]
testY = decision[2000:]
testX = condition[2000:]
testY = decision[2000:]
test_size = testX.shape[0]
a = ic.ICRS(batch_size)
b = iv.IVRS(batch_size)
c = rs.RS()
a.fit(X,Y)
b.fit(X,Y)
c.fit(X,Y)
correct = [0,0,0]
for j in range(testX.shape[0]):
    d1 = a.predict(testX[j])
    print(d1)
    if d1 == testY[j]:
        correct[0] = correct[0] + 1
    d1 = b.predict(testX[j])
    print(d1)
    if d1 == testY[j]:
        correct[1] = correct[1] + 1
    d1 = c.predict(testX[j])
    print(d1)
    if d1[0] == testY[j]:
        correct[2] = correct[2] + 1
for i in range(len(correct)):
    corrct[i] = correct[i]/test_size
print(correct)
