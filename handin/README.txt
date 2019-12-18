the model we proposed are:
1. ICRS.py: 
	combination of landmark, aggregate strategy, rough set

2. SwCRS.py:
	combination of sliding window, aggregate strategy, rough set 

3.TFCRS.py: 
	combination of time fading, aggregate strategy, rough set

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
RS.py: this file was the traditional rough set system that model 1-6 based on

FRS.py: this file was the fuzzy rough set that 7 - 8 based on

IRS.py: this was our implementation of rough set system with incremental attribute reduction, we use it as counterpart in experiment

LEM2.py: this is an implementation of the LEM2 algorithm

clustering.py: this is an implementation of using K-means for discretization

The folder 'visualization' contain the result of test on different hyper-parameter. The npy files in that folder
are the saved versions of a numpy matrix, which contain the test result.

The folder 'test' contain some testing code we use to verify our models, you might need to move them to current directory for working. Simply run the test code and the results will show.
Experiment 5.2.1:
testFRS_experiment5_2_1.py
testRs_experiment5_2_1.py

Experiment 5.2.2:
stream_test_rough_set_experiment5_2_2.py
stream_test_fuzzy_rough_set_experiment5_2_2.py

Experiment 5.2.4:
noise_test_experiment5_2_4.py

Experiment 5.2.5:
concept_drift_test_experiment5_2_5.py

Experiment 5.2.6:
batch_size_test_experiment5_2_6.py
fading_factor_test_experiment5_2_6.py
window_size_test_experiment5_2_6.py


The folder R implementation contain the source code of generating hyperplane.

To follow the convention of models in python(like ski-learn), our proposed models offer predict() and fit() function.
