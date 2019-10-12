import pandas as pd
Trainings = pd.read_csv('ratings.csv', header = 0, skiprows =  lambda i: i%5 == 0 and i > 0)
Test = pd.read_csv('ratings.csv', header = 0, skiprows =  lambda i: i%5 != 0)
Trainings.to_csv('training_ratings.csv')
Test.to_csv('test_ratings.csv')
