import pandas as pd
import numpy as np
import IRS as irs
import clustering
df = pd.read_csv('luWEKA.csv')
#df = df.drop(columns=[''])
df['at31'] = clustering.cluster(df['at31'],10).astype(str)

Y = df['label'].values
X = df.drop(columns=['label']).values
i = irs.I_RS()
i.fit(X[:1500],Y[:1500])

tf = irs.TFVRS(200,0.7)

tf.fit(X[:1500],Y[:1500])
correct = [0,0]
for j in range(400):
    d = tf.predict(X[1000+j])
    if d == Y[1000+j]:
        correct[0] = correct[0] + 1
    t,_ = i.predict(X[1000+j])
    if t == Y[1000+j]:
        correct[1] = correct[1] + 1