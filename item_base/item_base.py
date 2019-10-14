import pandas as pd
import numpy as np
import math

# Unnamed: 0, userId, movieId, rating, timestamp
train_csv = pd.read_csv('../data/training_ratings.csv')

# Unnamed: 0, userId, movieId, rating, timestamp
test_csv = pd.read_csv('../data/test_ratings.csv')

# movieId, title, genres
movie_csv = pd.read_csv('../data/movies.csv')

# calculate the mean rating for each movie.
# Means table contains columns:
# movieId, rating_mean
Means = train_csv.groupby(['movieId'], as_index=False).mean().rename(columns={'rating': 'rating_mean'})[
    ['movieId', 'rating_mean']]

# after the merge, the columns in train_csv are:
# Unnamed: 0, userId, movieId, rating, timestamp, rating_mean
train_csv = pd.merge(train_csv, Means, on='movieId', how='left')

# add a new column 'adjusted_rating', now the columns in train_csv are:
# Unnamed: 0, userId, movieId, rating, timestamp, rating_mean, adjusted_rating
train_csv['adjusted_rating'] = train_csv['rating'] - train_csv['rating_mean']

# train_csv: Unnamed: 0, userId, movieId1, rating, timestamp, rating_mean, adjusted_rating
train_csv = train_csv.rename(columns={'movieId': 'movieId1'})

# Before the merge:
# movie_csv: movieId, title, genres
# train_csv: Unnamed: 0, userId, movieId1, rating, timestamp, rating_mean, adjusted_rating
# after the merge:
# train_csv:
# movieId       title                genres               Unnamed   userId   movieId1    rating    timestamp     rating_mean     adjusted_rating
#   1     Toy Story (1995)  Animation|Children's|Comedy      15       15        1           5      978300760         3.6               1.4
#   1     Toy Story (1995)  Animation|Children's|Comedy      36       36        1           4      978300275         3.6               0.4
#   2     Jumanji (1995)    Adventure|Children's|Fantasy     120      120       2           1      978824268         2.5               -1.5
#   2     Jumanji (1995)    Adventure|Children's|Fantasy     255      255       2           5      978301752         2.5               2.5
#   2     Jumanji (1995)    Adventure|Children's|Fantasy     333      333       2           2      978302281         2.5               -0.5
#   3     Sabrina (1995)    Comedy|Romance                   15       15        3           3      978302124         4.4               -1.4
#   3     Sabrina (1995)    Comedy|Romance                   120      120       3           4      978302188         4.4               -0.4
train_csv = pd.merge(movie_csv, train_csv, left_on='movieId', right_on='movieId1', how='left')

# fill all NaN elements with 0s.
train_csv = train_csv.fillna(0)

# movieId     0     15       36      120     255     333 (userID)
#    1        0     1.4      0.4      0       0       0
#    2        0     0        0       -1.5    2.5     -0.5
#    3        0     -1.4     0       -0.4     0       0
train_pivot = train_csv.pivot(index='movieId', columns='userId', values='adjusted_rating').fillna(0)

# transform the table train_pivot into a 2D array:
#   0      1.4      0.4      0       0       0
#   0      0        0       -1.5    2.5     -0.5
#   0      -1.4     0       -0.4     0       0
train_mat = np.asarray(train_pivot)

# extract the 2nd and further columns from every each row.
# The reason why abandon the 1st column is that it is userID = 0, but there is not userID = 0.
# userID starts from 1.
# The reason why a column of userID = 0 appear is that the previous merge operation merged movie_csv and train_csv,
# if a movie has no rating, then it exists in movie_csv but not in train_csv.
# So when merging, all columns in train_csv will be filled by 0,
# and userID is one of the columns, so userID = 0 appear.
train_mat = train_mat[:, 1:]

sum_residual = 0  # the sum of the residual over all the prediction
prediction = []  # storing the prediction, basically necessary, but for me to debugging use
mean_mat = {}  # a python dictionary, contain key:value, key will be movieId and value will be mean_rating

for index, row in Means.iterrows():  # each row in the mean data frame
    # mean_mat: movieId, rating_mean
    mean_mat[row['movieId']] = row['rating_mean']  # we retrieve the movieId and the mean_rating to the dictionary

i = 0
for index, row in test_csv.iterrows():  # each row in the data we need to predict

    if i % 10 == 0:  # just showing me the progress of my program
        print("{} out of {}".format(i, test_csv.shape[0]))
    i = i + 1

    # the user we are predicting
    predict_user_index = int(row['userId']) - 1

    # the movie we are predicting
    predict_movie_index = int(row['movieId']) - 1

    similarity = []  # contain the similarity of the predicting movie to the movies the user already watched.
    rated_index = []  # contain the index of the movie.
    user_rated = []  # contain the rating of user.

    # train_mat.shape[0]: the amount of rows in train_mat
    # range: from 0 to a number
    # It as a whole means that take number from 0 to the amount of rows in train_mat
    # and consider each of them as a movie_id.
    for movie_id in range(train_mat.shape[0]):
        if train_mat[movie_id, predict_user_index] != 0:  # if this is a movie that the user has watched before.
            rated_index.append(movie_id)
            user_rated.append(train_mat[movie_id, predict_user_index])  # append the rating
            rated1 = []  # the rating for the
            rated2 = []  # the rating for the predict movie
            for user_id in range(train_mat.shape[1]):
                if train_mat[movie_id, user_id] != 0 and train_mat[predict_movie_index, user_id] != 0:
                    rated1.append(train_mat[movie_id, user_id])
                    rated2.append(train_mat[predict_movie_index, user_id])
            rated1 = np.asarray(rated1)
            rated2 = np.asarray(rated2)
            similarity.append(round(np.corrcoef(rated1, rated2)[0, 1], 2))
    ##after all the similarity is computed
    similarity = np.asarray(similarity)
    similarity[np.isnan(similarity)] = 0
    sorted_index = np.argsort(similarity)
    sum_sim = np.sum(similarity)
    pred = 0
    if sum_sim == 0:
        pred = mean_mat.get(row['movieId'], 2.5)

    else:
        w_sum = 0
        for j in range(min(len(rated_index), 20)):
            w_sum += similarity[sorted_index[j]] * user_rated[sorted_index[j]]
        pred = w_sum / sum_sim + mean_mat.get(row['movieId'], 2.5)
    residual = pred - row['rating']
    prediction.append(pred)
    sum_residual = sum_residual + math.fabs(residual)
mae = sum_residual / i
print(mae)
