import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import math
train_csv = pd.read_csv('training_ratings.csv')
test_csv = pd.read_csv('test_ratings.csv')
movie_csv = pd.read_csv('movies.csv')
Means = train_csv.groupby(['movieId'],as_index = False).mean().rename(columns = {'rating':'rating_mean'})[['movieId','rating_mean']]
train_csv = pd.merge(train_csv,Means,on='movieId', how = 'left')
train_csv['adjusted_rating'] = train_csv['rating'] - train_csv['rating_mean']
train_csv = train_csv.rename(columns = {'movieId':'movieId1'})
train_csv = pd.merge(movie_csv,train_csv, left_on = 'movieId', right_on = 'movieId1', how ='left')
train_csv.fillna(0)
train_pivot = train_csv.pivot(index = 'movieId', columns = 'userId', values = 'adjusted_rating').fillna(0)
train_mat = np.asarray(train_pivot)
i = 0
sum_residual = 0
for index,row in test_csv.iterrows():
    if i % 10 == 0:
        print(i)
    i = i + 1
    ##the user we are predicting
    predict_user_id = int(row['userId'])-1
    ##the movie we are predicting
    predict_movie_id = int(row['movieId'])-1
    similarity = []
    rated_index = []
    user_rated = []
    for movie_id in range(train_mat.shape[0]):
        if train_mat[movie_id,predict_user_id] != 0:
            rated_index.append(movie_id)
            user_rated.append(train_mat[movie_id,predict_user_id])
            rated1 = []
            rated2 = []
            for user_id in range(train_mat.shape[1]):
                if train_mat[movie_id,user_id] != 0 and train_mat[predict_movie_id,user_id] != 0:
                    rated1.append(train_mat[movie_id,user_id])
                    rated2.append(train_mat[predict_movie_id,user_id])
            rated1 = np.asarray(rated1)
            rated2 = np.asarray(rated2)
            similarity.append(np.corrcoef(rated1,rated2)[0,1])
    ##after all the similarity is computed
    similarity = np.asarray(similarity)
    sorted_index = np.argsort(similarity)
    sum_sim = np.sum(similarity)
    w_sum = 0
    for j in range(min(len(rated_index),20)):
        w_sum += similarity[sorted_index[j]]*user_rated[sorted_index[j]]
        
    prediction = w_sum/sum_sim
    residual = prediction - row['rating']
    sum_residual += residual
mae = sum_residual/i
print(mae)
    
    
    


