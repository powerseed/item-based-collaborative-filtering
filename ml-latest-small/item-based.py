#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:55:07 2019

@author: weiyidong
"""

import pandas as pd
import numpy as np
import math
Ratings = pd.read_csv('ratings.csv')
Movies = pd.read_csv('movies.csv')
Tags = pd.read_csv('tags.csv')
Means = Ratings.groupby(['movieId'],as_index = False, sort = False).mean().rename(columns = {'rating':'rating_mean'})[['movieId','rating_mean']]
Ratings = pd.merge(Ratings,Means,on='movieId', how = 'left', sort = False)
Ratings['adjusted_rating'] = Ratings['rating'] - Ratings['rating_mean']
##try to see the movie that user 2 not rate
user2_rated = Ratings[Ratings['userId'] == 1]##the movieId that user2 rated
distinct_rated =np.unique(user2_rated['movieId'])
user2_r = []
for movieId in user2_rated['movieId']:
    user2_r.append(movieId)

user2_not_rated = []
for movieId in Movies['movieId']:
    if not movieId in user2_r:
        user2_not_rated.append(movieId)

movie_unrated = pd.DataFrame()
for movieId1 in user2_not_rated:
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
        join_data = pd.merge(movie_data1,movie_data2,on='userId',how='inner',sort = False)
        join_data['vector_product'] = join_data['adjusted_rating1']*join_data['adjusted_rating2']
        join_data = join_data.groupby(['movieId1','movieId2'],as_index = False, sort = False).sum()
        join_data['similarity'] = join_data['vector_product']/(vector_length1*vector_length2)
        movie_sim = movie_sim.append(join_data, ignore_index = True)
    
    movie_sim = movie_sim[movie_sim['similarity']<1].sort_values(['similarity'],ascending = False)
    movie_sim = movie_sim.head(20)
    movie_unrated = movie_unrated.append(movie_sim, ignore_index = True)

    
    
    
    
    
