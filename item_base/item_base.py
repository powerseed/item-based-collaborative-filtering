import pandas as pd
import numpy as np
import math
train_csv = pd.read_csv('../data/training_ratings.csv')
test_csv = pd.read_csv('../data/test_ratings.csv')
movie_csv = pd.read_csv('../data/movies.csv')
Means = train_csv.groupby(['movieId'],as_index = False).mean().rename(columns = {'rating':'rating_mean'})[['movieId','rating_mean']]
train_csv = pd.merge(train_csv,Means,on='movieId', how = 'left')
train_csv['adjusted_rating'] = train_csv['rating'] - train_csv['rating_mean']
train_csv = train_csv.rename(columns = {'movieId':'movieId1'})
train_csv = pd.merge(movie_csv,train_csv, left_on = 'movieId', right_on = 'movieId1', how ='left')
train_csv = train_csv.fillna(0)
train_pivot = train_csv.pivot(index = 'movieId', columns = 'userId', values = 'adjusted_rating').fillna(0)
train_mat = np.asarray(train_pivot)
train_mat = train_mat[:,1:]
i = 0
sum_residual = 0##the sum of the residual over all the prediction
prediction = []## storing the prediction, basically necessary, but for me to debugging use
mean_mat = {}## a python dictionary, contain key:value, key will be movieId and value will be mean_rating
for index,row in Means.iterrows():##each row in the mean data frame
    mean_mat[row['movieId']]=row['rating_mean'] ## we retrieve the movieId and the mean_rating to the dictionary
for index,row in test_csv.iterrows():##each row in the data we need to predict
    if i % 10 == 0:##just showing me the progress of my program
        print("{} out of {}".format(i, test_csv.shape[0]))
    i = i + 1
    ##the user we are predicting
    predic_user_index = int(row['userId'])-1
    ##the movie we are predicting
    predic_movie_index = int(row['movieId'])-1
    similarity = []## contain the similarity of the predicting movie to the movie the user already watch
    rated_index = []## cotain the index of the movie
    user_rated = []## contain the rating of user
    for movie_id in range(train_mat.shape[0]):
        if train_mat[movie_id,predic_user_index] != 0: ## if this movie the user watch before
            rated_index.append(movie_id)
            user_rated.append(train_mat[movie_id,predic_user_index])##append the rating
            rated1 = []##the rating for the 
            rated2 = []## the rating for the predict movie
            for user_id in range(train_mat.shape[1]):
                if train_mat[movie_id,user_id] != 0 and train_mat[predic_movie_index,user_id] != 0:
                    rated1.append(train_mat[movie_id,user_id])
                    rated2.append(train_mat[predic_movie_index,user_id])
            rated1 = np.asarray(rated1)
            rated2 = np.asarray(rated2)
            similarity.append(round(np.corrcoef(rated1,rated2)[0,1],2))
    ##after all the similarity is computed
    similarity = np.asarray(similarity)
    similarity[np.isnan(similarity)] = 0
    sorted_index = np.argsort(similarity)
    sum_sim = np.sum(similarity)
    pred = 0
    if sum_sim == 0:
        pred = mean_mat.get(row['movieId'],2.5)
        
    else:    
        w_sum = 0
        for j in range(min(len(rated_index),20)):
            w_sum += similarity[sorted_index[j]]*user_rated[sorted_index[j]]    
        pred = w_sum/sum_sim + mean_mat.get(row['movieId'],2.5)
    residual = pred - row['rating']
    prediction.append(pred)
    sum_residual = sum_residual +  math.fabs(residual)
mae = sum_residual/i
print(mae)
    
    
    


