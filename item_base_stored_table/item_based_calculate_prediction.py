import pandas as pd
import numpy as np
import math

similarity_csv = pd.read_csv('../data/similarities.csv')
exchange_i_and_j = similarity_csv[['movieId_j', 'movieId_i', 'similarity']]
exchange_i_and_j.rename(columns={'movieId_j': 'movieId_i', 'movieId_i': 'movieId_j'},
                        inplace=True)
concated_by_the_two_tables = pd.concat([similarity_csv, exchange_i_and_j], sort=False)

concated_by_the_two_tables.sort_values(['movieId_i', 'similarity'], ascending=False, inplace=True)

grouped = concated_by_the_two_tables.groupby(['movieId_i']).head(10)

grouped.to_csv(r'C:\4710project\item-based-collaborative-filtering\data\Top_10.csv',
                                      index=None, header=True)


