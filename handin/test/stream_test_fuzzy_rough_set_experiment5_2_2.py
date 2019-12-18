import TFVFRS as tf
import IVFRS as iv
import SwVFRS as sw
import pandas as pd
batch_size = 50
fading_factor = 0.95
window_size = 20
n_cluster = 10

#####################electricity############################

df = pd.read_csv('data/stream_data/electricity_csv.csv')[:10000]
df = df.drop(columns = ['date'])

condition = df.drop(columns = ['class']).values
decision = df['class'].values

#####################hyperplane 10#############################

#df = pd.read_csv('data/stream_data/rh.csv')[:10000]

#condition = df.drop(columns = ['label']).values
#decision = df['class'].values

#####################hyperplane 20#############################

#df = pd.read_csv('data/stream_data/hyperplane20.csv')[:10000]

#condition = df.drop(columns = ['response']).values
#decision = df['response'].values

#######################moving RBF#############################

#df = pd.read_csv('data/stream_data/movingRBF.csv')[:10000]

#condition = df.drop(columns = ['target']).values
#decision = df['target'].values

##########################weather##############################

#df = pd.read_csv('data/stream_data/weather.csv')[:10000]

#condition = df.drop(columns = ['target']).values
#decision = df['target'].values

###############################################################
X = condition[:2000]
Y = decision[:2000]
a = sw.SwVFRS(batch_size,window_size)
b = iv.IVFRS(batch_size)
c = tf.TFVFRS(batch_size,fading_factor)
a.fit(X[:2000],Y[:2000])
b.fit(X[:2000],Y[:2000])
c.fit(X[:2000],Y[:2000])
testX = X[2000:]
testY = Y[2000:]
test_size = testX.shape[0]
correct = [0,0,0,0]
i = 1
while testX.shape[0] >= batch_size:
    print(i)
    i = i + 1
    for j in range(batch_size):
        d1 = a.predict(testX[j])
        if d1 == testY[j]:
            correct[0] = correct[0]+1
        d2 = b.predict(testX[j])
        if d2 == testY[j]:
            correct[1] = correct[1]+1
        d3 = c.predict(testX[j])
        if d3 == testY[j]:
            correct[2] = correct[2]+1
    a.update_group(testX[:batch_size],testY[:batch_size])
    b.update_group(testX[:batch_size],testY[:batch_size])
    c.update_group(testX[:batch_size],testY[:batch_size])
    testX = testX[batch_size:]
    testY = testY[batch_size:]
for i in range(len(correct)):
    correct[i] = correct[i]/test_size
    
print(correct)
