import scipy as sp
import pandas as pd
import numpy as np
import mflibrary as mf
import math
#r_csv = pd.read_csv('../data/training_ratings.csv')[['userId','movieId','rating','timestamp']].rename(columns={'movieId':'movieId1'})
#m_csv = pd.read_csv('../data/movies.csv', )[['movieId']]
#r_csv = pd.merge(m_csv,r_csv, left_on = 'movieId', right_on = 'movieId1', how ='left')
#r_csv = r_csv.fillna(0)
#r_mat = np.asarray(r_csv.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0))[:,1:]
#K = 5
#I = np.random.rand(r_mat.shape[0],K)
#U = np.random.rand(K,r_mat.shape[1])
#(I,U) = mf.matrix_fatorization(r_mat,I,U,K)
#u_output = open('u.npy','wb')
#i_output = open('i.npy','wb')
#np.save(u_output,U)       
#np.save(i_output,I)  
u_input = open('u.npy','rb')
i_input = open('i.npy','rb')
U = np.load(u_input)       
I = np.load(i_input)         

test_csv = pd.read_csv('../data/test_ratings.csv')[['userId','movieId','rating','timestamp']].rename(columns={'movieId':'movieId1'})
test_csv = pd.merge(m_csv,test_csv, left_on = 'movieId', right_on = 'movieId1', how ='left')
test_csv = test_csv.fillna(0)
test_mat = np.asarray(test_csv.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0))[:,1:]
esq = 0
e = 0
count = 0
for i in range(test_mat.shape[0]):
    for j in range(test_mat.shape[1]):
        if test_mat[i,j] > 0:
            esq = esq + pow(test_mat[i,j] - np.sum(I[i,:] * U[:,j]),2)
            e = e + math.fabs(test_mat[i,j] - np.sum(I[i,:] * U[:,j]))
            count += 1

mesr = esq/count
mae = e/count
print(mesr)
print(mae)