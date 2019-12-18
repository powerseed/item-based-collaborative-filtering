the model we proposed are:
1. ICRS.py: 
	combination of landmark, aggreate strategy, rough set

2. SwCRS.py:
	combination of sliding window, aggreate strategy, rough set 

3.TFCRS.py: 
	combination of time fading, aggreate strategy, rough set

4. IVRS.py: 
	combination of landmark, batch basis strategy, rough set

5. SwVRS.py: 
	combination of sliding window, batch basis strategy, rough set

6. TFVRS.py: 
	combination of time fading, batch basis strategy, rough set

7. IVFRS.py: 
	combination of landmark, batch basis, fuzzy rough set

8. SwVFRS.py: 
	combination of sliding window, batch basis, fuzzy rough set

9. TFVFRS.py: 
	combination of time fading, batch basis, fuzzy rough set

other file:
RS.py: this file was the taditional rough set system that model 1-6 based on

FRS.py: this file was the fuzzy rough set that 7 - 8 based on

IRS.py: this was our implementation of rough set system with incremental attribute reduction, we use it as counterpart in experiment

LEM2.py: this is an implementation of the LEM2 algorithm

clustering.py: this is an implementation of using K-means for discretization

The folder 'visualization' contain the result of test on different hyperparameter. The npy files in that folder
are the saved versions of a numpy matrix, which contain the test result.

The folder 'test' contain some testing code we use to verify our models, you might need to move them to current diretory for working. Simply run the test code and the result will shows.

The folder R implementation contain the source code of generating hyperplane.

To follow the convention of models in python(like skilearn), our proposed models offer predict() and fit() function.
