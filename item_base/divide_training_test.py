import pandas as pd
Trainings = pd.read_csv('data/ratings.csv', header = 0, skiprows =  lambda i: i%60 == 0 and i > 0)
Test = pd.read_csv('data/ratings.csv', header = 0, skiprows =  lambda i: i%60 != 0)
Trainings.to_csv('data/training_ratings.csv')
Test.to_csv('data/test_ratings.csv')
