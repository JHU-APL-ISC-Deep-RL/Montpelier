# Montpelier: PyTorch, MPI-based Multi-Task and Meta-Learning

Note: this repository has mostly been tested; however further modifications to the code may occur in the coming months.

----------------------
----------------------
Building & Installing
----------------------
 
 To install this code and get it running, first set up a virtual environment. We developed and tested this using 
 conda and python 3.6 but it is likely that one could replicate the results using a virtualenv.  Create the environment
 and then do the following:
 
1. Clone this repository

2.  This repository uses PyTorch.  To ensure that you get the correct version, go to their 
[installation page](https://pytorch.org/get-started/locally/) and install the correct version.

3. This repository uses MPI for parallelization.  We found that the conda-forge version works well; install it via
`conda install -c conda-forge mpi4py`.  Keeping the door open for Atari-type experiments, you may also want to install
   OpenCV via `conda install -c conda-forge opencv`.

4. We use [Meta-World](https://github.com/rlworkgroup/metaworld) for evaluating our ideas.  Go to the link, clone the
repository, and follow the instructions to install.  Note that Meta-World uses [MuJoCo](http://www.mujoco.org).

5.)  Install uaml.  From this home folder, type `pip install -e .` .

Training and Testing
----------------------
TODO