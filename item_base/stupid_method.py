import pandas as pd
import numpy as np
Train_ratings = pd.read_csv('training_ratings.csv')
Test_ratings = pd.read_csv('test_ratings.csv')
##calculate the mean of the rating in every movie
Means = Train_ratings.groupby(['movieId'],as_index = False).mean().rename(columns = {'rating':'rating_mean'})[['movieId','rating_mean']]
##merge the mean table and rating table by the movie id
Train_ratings = pd.merge(Train_ratings,Means,on='movieId', how = 'left')
##adjust the rating, grater than the mean means good, otherwise bad
Train_ratings['adjusted_rating'] = Train_ratings['rating'] - Train_ratings['rating_mean']
## the dataframe hold the top 20 similar movie
top_20 = pd.DataFrame()
##counter
i = 0
for index,row in Test_ratings.iterrows():
    if i % 10 == 0:
        print(i)
    i=i+1
    ##the user we are predicting
    predict_user = row['userId']
    ##the movie we are predicting
    predict_movie = row['movieId']
    ##table contain all the rating about our predicted movie
    movie_data1 = Train_ratings[Train_ratings['movieId'] == predict_movie]
    ##just retrieve the attribute we need: userid, movieid and rating, and drop the duplicate as well
    movie_data1 = movie_data1[['userId','movieId','adjusted_rating']].drop_duplicates()
    ##rename the attribute for later join
    movie_data1 = movie_data1.rename(columns={'movieId':'movieId1'})
    movie_data1 = movie_data1.rename(columns={'adjusted_rating':'adjusted_rating1'})
    ##calculate the length of the rating vector
    vector_length1 = np.sqrt(np.sum(np.square(movie_data1['adjusted_rating1'])))
    ##dataframe that hold the similarity value
    movie_sim = pd.DataFrame()
    ##the movie that the user rated
    predict_user_rated = Train_ratings[Train_ratings['userId']==predict_user].drop_duplicates()
    for rated_movie in predict_user_rated['movieId']:##for each movie the user rated
        ##get all the rating about this rated movie
         movie_data2 = Train_ratings[Train_ratings['movieId']== rated_movie]
        ##rename the attribute name
         movie_data2 = movie_data2[['userId','movieId','adjusted_rating']].drop_duplicates()##the item the user not rate
         movie_data2 = movie_data2.rename(columns={'movieId':'movieId2'})
         movie_data2 = movie_data2.rename(columns={'adjusted_rating':'adjusted_rating2'})
         ##calculate the rating vector length
         vector_length2 = np.sqrt(np.sum(np.square(movie_data2['adjusted_rating2'])))
         ## join the unrated table with the rated table by the userId, that is to find the user who rated both
         join_data = pd.merge(movie_data1,movie_data2,on='userId',how='inner')
         ## the dot product, since we are doing the cosin similarity
         join_data['vector_product'] = join_data['adjusted_rating1']*join_data['adjusted_rating2']
         ##group by the movieIds, and sum up the product
         join_data = join_data.groupby(['movieId1','movieId2'],as_index = False).sum()
         ##calculate the cosin similarity note sim(x,y) = cos(theta) = x dot y /length(x)length(y)
         join_data['similarity'] = join_data['vector_product']/(vector_length1*vector_length2)
         ##since we only interested in the similarity, just add it and the ids to the movie_sim
         ##so movie_sim will contains the similarity between the current movie and the moview user rated before
         movie_sim = movie_sim.append(join_data[['movieId1','movieId2','similarity']], ignore_index = True, sort = False)##similarity between movie i and j
    
    ##now the movie_sim contain all the  similarity between the current movie and the moview user rated before, and we sore it by ascending order   
    movie_sim = movie_sim[movie_sim['similarity']<1].sort_values(['similarity'],ascending = False)
    ##top 20 similar item rate by user
    movie_sim = movie_sim.head(20) 
    movie_sim['userId'] = predict_user
    #add these top 20 to the top_20 table, note top_20 table after the outer loop will contain all the top 20 for all the movie we will predict
    top_20 = top_20.append(movie_sim, ignore_index = True, sort = False)
#prediction step
#the sum of similarity in the specific movie for specific user    
sum_sim = top_20.groupby(['movieId1','userId'],as_index = False).sum().rename(columns = {'similarity':'sum_sim'})[['movieId1','sum_sim','userId']]
# temp is the join of the Train_ratings with the top 20 similarity, by the key userID and movieId
temp = pd.merge(Train_ratings,top_20,left_on=['movieId','userId'], right_on = ['movieId2','userId'], how = 'inner')
# calculate the weighted rating
temp['weighted_rate'] = temp['similarity'] * temp['adjusted_rating']
# group the temp by movie1 and userId(movie1 is the one we want to predict), and sum up all the weighted rating that is similar to movie1
temp = temp.groupby(['movieId1','userId'],as_index = False).sum().rename(columns = {'weighted_rate':'sum_w_rate'})[['movieId1','sum_w_rate','userId']]
#join the sum_sim with temp, on movie1 and userid
prediction = pd.merge(sum_sim,temp, on = ['movieId1','userId'], how = 'inner')
prediction['pred_rate'] = prediction['sum_w_rate']/prediction['sum_sim']
prediction = prediction[['pred_rate','userId','movieId1']]
prediction = prediction.rename(columns={'movieId1':'movieId'})
prediction = pd.merge(Means[['movieId','rating_mean']],prediction, on=['movieId'],how = 'inner')
prediction['pred_rate'] = prediction['pred_rate'] + prediction['rating_mean']
comparison = pd.merge(Test_ratings,prediction[['pred_rate','userId','movieId']], on = ['movieId','userId'], how = 'inner')
comparison['diff'] = comparison['pred_rate'] - comparison['rating']