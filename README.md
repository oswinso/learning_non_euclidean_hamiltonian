# Data-driven discovery of non-Newtonian astronomy via learning non-Euclidean Hamiltonian

First, generate a dataset by compiling the GRIT simulator in `external/GRIT`.
The binary should be located in `external/GRIT/cmake-build-release/src/simulate`.
This path is hardcoded in `generate_grit_data.py`.

Next, run `generate_grit_data.py` to generate a dataset.

Finally, run `train_planets.py` to train.
