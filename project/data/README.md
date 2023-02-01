# Data folder
This folder contains all the data from the TEBD simulations I have done.

---

**/trained**
: This folder contains the models trained on **data_training.hdf5**.
<br/>

**/trained_unc**
: This folder contains the models trained on **data_training_unc.hdf5**
<br/>


**/.trained_model_old**
: This contains the models trained on the rescaled data, but this data was generated with a cutoff of 1e-8, 
: which is too small and hence might bias the these models might get better predictions just becasue
: are trained on better data.
<br/>
    
**/.trained_not_rescaled**
: This folder contains the models trained on data not rescaled, indeed they suck.
<br/>

---

Legend:
: W &rarr; weakly interacting   V = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
: S &rarr; strongly interacting V = [1, 2, 3, 4, 5, 6, 7, 8, 9]
: 
<br/>

- **data_training**
This file contains the TEBD data (cutoff 1e-5) needed to train the models.
It contains the following data:

	| L  | V   | T  | num_traj |
	|:--:|:---:|:--:|:--------:|
	| 20 | S   |  5 |   30     |
	| 40 | WS  |  4 |   30     |
	| 60 | WS  |  4 |   30     |


- **data_test**
This file contains the TEBD data (cutoff 1e-5) needed to test the models.
(for W,L=20, not V=0.5, I used 10^-6 actually)
Note all initials conditions are the same.
It contains the following data:

	| L  | V   | T  | num_traj |
	|:--:|:---:|:--:|:--------:|
	| 20 |  W  | 20 |    1     |
	| 20 |  S  | 10 |    1     |
	| 40 |  W  | 20 |    1     |
	| 40 |  S  |  8 |    1     |
	| 60 |  W  | 20 |    1     |
	| 60 |  W  |  8 |    1     |

---

- **data_unc_training**
This file contains the TEBD data (cutoff 1e-5) needed to train the models.
Here I used uncorrelated initial consitions.
It contains the following data:

	| L  | V   | T  | num_traj |
	|:--:|:---:|:--:|:--------:|
	| 20 | W   | 10 |   30     |
	| 20 | S   |  5 |   30     |
	| 40 | W   | 10 |   30     |
	| 40 | S   |  5 |   30     |
	| 60 | W   | 10 |   30     |
	| 60 | S   |  5 |   30     |
	| 80 | W   | 10 |   30     |


- **data_unc_test**
This file contains the TEBD data (cutoff 1e-5) needed to test the models.
Here I used uncorrelated initial consitions.
Note all initials conditions are the same.
It contains the following data:

	| L  | V   | T  | num_traj |
	|:--:|:---:|:--:|:--------:|
	| 20 | W   | 20 |    1     |
	| 20 | S   | 10 |    1     |
	| 40 | W   | 20 |    1     |
	| 40 | S   |  8 |    1     |
	| 60 | W   | 20 |    1     |
	| 60 | S   |  8 |    1     |
	| 80 | W   | 20 |    1     |

---

- **data_tebd_old**
This is the data used to generate model_old, so the one generated with 1e-8

----

