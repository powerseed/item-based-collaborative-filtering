#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:55:07 2019

@author: weiyidong
"""

import pandas as pd
import numpy as np
import sys
Ratings = pd.read_csv('ratings.csv')
Movies = pd.read_csv('movies.csv')
Means = Ratings.groupby(['movieId'],as_index = False).mean().rename(columns = {'rating':'rating_mean'})[['movieId','rating_mean']]
Ratings = pd.merge(Ratings,Means,on='movieId', how = 'left')
Ratings['adjusted_rating'] = Ratings['rating'] - Ratings['rating_mean']
#try to see the movie that user 2 not rate
user_rated = Ratings[Ratings['userId'] == sys.argv[1]]##the movieId that user2 rated
distinct_rated =np.unique(user_rated['movieId'])
user_r = []
for movieId in user_rated['movieId']:
    user_r.append(movieId)

user2_not_rated = []
for movieId in Movies['movieId']:
    if not movieId in user_r:
        user2_not_rated.append(movieId)
i = 0
movie_unrated = pd.DataFrame()
for movieId1 in user2_not_rated:
    if i%10 == 0:
        print("{}out of{}".format(i,len(user2_not_rated)))
    i = i + 1
    movie_data1 = Ratings[Ratings['movieId']== movieId1]
    movie_data1 = movie_data1[['userId','movieId','adjusted_rating']].drop_duplicates()##the item the user not rate
    movie_data1 = movie_data1.rename(columns={'movieId':'movieId1'})
    movie_data1 = movie_data1.rename(columns={'adjusted_rating':'adjusted_rating1'})
    vector_length1 = np.sqrt(np.sum(np.square(movie_data1['adjusted_rating1'])))
    movie_sim = pd.DataFrame()
    for movieId2 in distinct_rated:
        movie_data2 = Ratings[Ratings['movieId'] == movieId2]
        movie_data2 = movie_data2[['userId','movieId','adjusted_rating']].drop_duplicates()##all the data about one items the user2 has rate
        movie_data2 = movie_data2.rename(columns={'movieId':'movieId2'})
        movie_data2 = movie_data2.rename(columns={'adjusted_rating':'adjusted_rating2'})
        vector_length2 = np.sqrt(np.sum(np.square(movie_data2['adjusted_rating2'])))
        join_data = pd.merge(movie_data1,movie_data2,on='userId',how='inner')
        join_data['vector_product'] = join_data['adjusted_rating1']*join_data['adjusted_rating2']
        join_data = join_data.groupby(['movieId1','movieId2'],as_index = False).sum()
        join_data['similarity'] = join_data['vector_product']/(vector_length1*vector_length2)
        movie_sim = movie_sim.append(join_data, ignore_index = True, sort = False)
    
    movie_sim = movie_sim[movie_sim['similarity']<1].sort_values(['similarity'],ascending = False)
    movie_sim = movie_sim.head(20)
    movie_unrated = movie_unrated.append(movie_sim, ignore_index = True, sort = False)

sum_sim = movie_unrated.groupby(['movieId1'],as_index = False).sum().rename(columns = {'similarity':'sum_sim'})[['movieId1','sum_sim']]
temp = pd.merge(Ratings,movie_unrated[['movieId1','movieId2','similarity']],left_on='movieId', right_on = 'movieId2', how = 'inner')
temp['weighted_rate'] = temp['similarity'] * temp['ajusted_rating']
temp = temp.groupby(['movieId1'],as_index = False).sum().rename(columns = {'weighted_rate':'sum_w_rate'})[['movieId1','sum_w_rate']]
prediction = pd.merge(sum_sim,temp, on = 'movieId1', how = 'inner')
prediction['pred_rate'] = prediction['sum_w_rate']/prediction['sum_sim']


    
#    
#    
#    
#    
