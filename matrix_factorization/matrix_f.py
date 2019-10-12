import scipy as sp
import pandas as pd
import numpy as np
import mflibrary as mf
r_csv = pd.read_csv('../data/training_ratings.csv')[['userId','movieId','rating','timestamp']].rename(columns={'movieId':'movieId1'})
m_csv = pd.read_csv('../data/movies.csv', )[['movieId']]
r_csv = pd.merge(m_csv,r_csv, left_on = 'movieId', right_on = 'movieId1', how ='left')
r_csv = r_csv.fillna(0)
r_mat = np.asarray(r_csv.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0))[:,1:]
K = 10
I = np.random.rand(r_mat.shape[0],K)
U = np.random.rand(K,r_mat.shape[1])
(I,U) = mf.matrix_fatorization(r_mat,I,U,K)
