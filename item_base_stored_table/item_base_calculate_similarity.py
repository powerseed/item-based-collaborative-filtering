from random import randint

import pandas as pd
import numpy as np
import math

# Unnamed: 0, userId, movieId, rating, timestamp
from pandas import DataFrame

train_csv = pd.read_csv('../data/training_ratings_trimmed.csv')
# train_csv = pd.read_csv('../data/training_ratings.csv')

# Unnamed: 0, userId, movieId, rating, timestamp
test_csv = pd.read_csv('../data/test_ratings_trimmed.csv')
# test_csv = pd.read_csv('../data/test_ratings.csv')

# movieId, title, genres
movie_csv = pd.read_csv('../data/movies_trimmed.csv')
# movie_csv = pd.read_csv('../data/movies.csv')

# calculate the mean rating for each movie.
# Means table contains columns:
# movieId, rating_mean
Means = train_csv.groupby(['movieId'], as_index=False).mean().rename(columns={'rating': 'rating_mean'})[
    ['movieId', 'rating_mean']]

table_similarities = pd.DataFrame(columns = ['movieId_i', 'movieId_j', 'similarity'])
# table_similarities.loc[table_similarities.shape[0]+1] = [1, 2, 3]
# print(table_similarities)
# df = DataFrame(columns=('lib', 'qty1', 'qty2'))#生成空的pandas表
# df.loc[df.shape[0]+1] = [randint(-1,1) for n in range(3)]
# print(df)

# table_similarities.to_csv(r'C:\4710project\item-based-collaborative-filtering\data\test.csv', index=None, header=True)

i = 0
for index, row_one_movie in movie_csv.iterrows():
    if i % 10 == 0:
        print(i)
    i = i + 1

    movieId_i = int(row_one_movie["movieId"])
    movieId_i_rating_mean = Means[Means["movieId"] == movieId_i]["rating_mean"]
    # print(movieId_i)
    # print(movieId_i, ": ", movieId_i_rating_mean)

    users_rated_movie_i = train_csv[ train_csv['movieId'] == movieId_i][["userId"]]
    # print(users_rated_movie_i)

    for index1, row_another_movie in movie_csv.iterrows():
        movieId_j = int(row_another_movie["movieId"])

        if movieId_j == movieId_i:
            break
        else:
            movieId_j_rating_mean = Means[Means["movieId"] == movieId_j]["rating_mean"]

            users_rated_movie_j = train_csv[train_csv['movieId'] == movieId_j][["userId"]]

            users_rated_both_i_and_j = pd.merge(users_rated_movie_i, users_rated_movie_j, how="inner", on="userId")[["userId"]]

            if(users_rated_both_i_and_j.shape[0] == 0):
                break
            else:
                # print("movieId_i: ", movieId_i, ", movieId_j: " , movieId_j, "users_rated_both_i_and_j: ", users_rated_both_i_and_j)

                numerator = 0
                denominator_left = 0
                denominator_right = 0

                ratings_of_movie_i = train_csv[train_csv['movieId'] == movieId_i]
                ratings_of_movie_j = train_csv[train_csv['movieId'] == movieId_j]

                for index3, user in users_rated_both_i_and_j.iterrows():
                    userId = int(user["userId"])
                    # print(userId)

                    rating_of_movie_i_by_this_user = int( ratings_of_movie_i[ratings_of_movie_i['userId'] == userId] ["rating"] )
                    rating_of_movie_j_by_this_user = int( ratings_of_movie_j[ratings_of_movie_j['userId'] == userId] ["rating"] )

                    Rui_minus_mean_Ri = float(rating_of_movie_i_by_this_user - movieId_i_rating_mean)
                    Ruj_minus_mean_Rj = float(rating_of_movie_j_by_this_user - movieId_j_rating_mean)
                    numerator += float((Rui_minus_mean_Ri) * (Ruj_minus_mean_Rj))

                    denominator_left += float(Rui_minus_mean_Ri * Rui_minus_mean_Ri)
                    denominator_right += float(Ruj_minus_mean_Rj * Ruj_minus_mean_Rj)

                    #print(rating_of_movie_i_by_this_user)
                # end for

                if numerator == 0:
                    similarity = 0
                else:
                    sqrt_denominator_left = float(np.sqrt(denominator_left))
                    sqrt_denominator_right = float(np.sqrt(denominator_right))

                    denominator = float(sqrt_denominator_left * sqrt_denominator_right)
                    if(denominator == 0):
                        similarity = 0
                    else:
                        similarity = numerator / denominator
                table_similarities.loc[table_similarities.shape[0]+1] = [movieId_i, movieId_j, similarity]
    #end for
#end for
table_similarities.to_csv(r'C:\4710project\item-based-collaborative-filtering\data\table_similarities.csv',
                                      index=None, header=True)
