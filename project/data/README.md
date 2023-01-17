# Data folder
This folder contains all the data from simulations and training.
    
**data_tebd_old**
: This is the data used to generate model_old, so the one generated with 1e-8

----
    
**data_training_W**
: This file contains the file needed to train the model for the weakly regime.
: Generated 30 trajectories using cutoff 1e-5
: - for L = 20 up to T = 10  **lost data**
: - for L = 40, 60 up to T = 4

**data_test_W**
: This file contains data needed to test the model for further times, up to T=20
: Generated 1 trajectory, up to T=10 using cutoff 1e-5  (N=20, not V=0.5, I used 10^-6 actually)
: - for L = 20,40,60 up to T = 20
    
**data_training_S**
: This file contains the file needed to train the model for the strong regime.
: Generated 30 trajectories, up to T=5 and using cutoff 1e-5
: - for L = 20 up to T = 5
: - for L = 40, 60 up to T = 4 

**data_test_S**
: This file contains data needed to test the model for further times, up to T=10
: Generated using cutoff 1e-5
: - for L = 20 up to  = 10
: - for L = 40, 60 up to  = 8

----

**data_unc_training_W**
: This file contains the file needed to train the model for the weakly regime.
: With uncorrelated initial condition.
: Generated 30 trajectories using cutoff 1e-5
: - for L = 20, 40, 60 up to T = 4


    
### Trained_S
- This folder contains the models trained on 'data_training_S'
    
### Trained_W
- This folder contains the models trained on 'data_training_W'

### Trained_model_old
- This contains the models trained on the rescaled data, but this data was generated with a cutoff of 1e-8, which is too small
    
### Trained_not_rescaled
- This folder contains the models trained on data not rescaled, indeed they suck.
