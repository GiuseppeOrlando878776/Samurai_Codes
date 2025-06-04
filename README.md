# TwoPhase_TwoScale
This is the code for the two-phase two-scale with mass transfer and interface regularization including capillarity effects. Two subfolders have been included :'TwoScale_Capillarity_NoMassTransfer' which is a minimal version where only large-scale variables are considered and 'TwoScale_Capillarity' with the whole model. An input parameter file is present in each subfolder. In order to compile the desired program it is sufficient to move into the corresponding subfolder and run source configure.sh

# Conda environment
mamba install cxx-compiler cmake make samurai libboost-mpi libboost-devel libboost-headers highfive=2.9 'hdf5=*=mpi*' nlohmann_json 

##License##
This project is licensed under the ##BSD## license.
