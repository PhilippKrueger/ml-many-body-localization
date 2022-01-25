# Machine Learning of Many Body Localization

Documentation is found in report.pdf

**Instructions to replicate results**

1. Clone the project.
2. Install requirements with ``pip install -r requirements.txt``.
3. Run ``main.py`` and set your preferred parameters for each step.
``Ns``: system sizes, 
``n_max`` or ``ns``: maximum block size/list of block sizes, 
``Ws``: list of tested disorder strengths, 
``repetitions``: amount of samples that is multiplied by the factor of different used Ws, number of used Eigenvalues, system sizes and block sizes.
    - The model will generate training data into the folder ``lanczos/training_sets``
    - The testing set will be saved to ``lanczos/testing_sets``
    - The trained models will be saved to ``lanczos/models``
    - The averaged prediction data will be saved to ``lanczos/avg_prediction_sets``
4. Look for plots in the ``/results`` folder.
    - ``results/accuracy_loss_epochs`` contains summary plots of all losses and 
    accuracies and also individual performances over epochs.
    - ``results/accuracy_loss_epochs`` contains vizualizations of the ground state density matrices
    for each block and system size.
    - ``results/Wc`` contains the heat map plots that show the phase transitions.
    
**Original Task**:

Use exact diagonalization to obtain all eigenstates of the the Heisenberg model with a
random field, 
<img src="https://render.githubusercontent.com/render/math?math=H=J\sum_i \vec{S}_i\cdot\vec{S}_{i%2B1}-\sum_ih_iS_i^z">
, where the values of the field <img src="https://render.githubusercontent.com/render/math?math=h_i \in \left[-W, W\right]">
 are
chosen from a uniform random distribution with a “disorder strength” W (Use moderate
system sizes L = 10, 12). The exciting property of this model is that it is believed to
undergo a phase transition from an extended phase (small W) to a localized phase (large
W). We will use ML to detect this transition: Pick a number of eigenstates that are
near energy <img src="https://render.githubusercontent.com/render/math?math=E = 0">
 and obtain the reduced density matrices 
 <img src="https://render.githubusercontent.com/render/math?math=\rho^A">
 , where A is a region
of n consecutive spins (a few hundred to thousands eigenstates for different disorder
realizations). Now use the density matrices for 
<img src="https://render.githubusercontent.com/render/math?math=W = 0.5J">
 and 
 <img src="https://render.githubusercontent.com/render/math?math=W = 8.0J">
 to train a
neural network (just interpret the entries of 
<img src="https://render.githubusercontent.com/render/math?math=\rho^A">
 as an image with 
 <img src="https://render.githubusercontent.com/render/math?math=2^n \times 2^n">
  pixel). Then
use this network and study the output of the neural network for different W. How does
the results depend on system size L and block size n? At which 
<img src="https://render.githubusercontent.com/render/math?math=W_c ">
do you expect the
transition to occur?
