# Resonance counting

In this repository, we make available the code used to generate the data presented in the paper "Investigating finite-size effects in random matrices by counting resonances" by Anton Kutlin and Carlo Vanoni

## resonance_counting_SciPost.py
It contains the Python code used to generate the numerical results presented in Section 6 of the paper.  
It computes eigenfunction entropies via exact diagonalization and the self-consistent resonance condition presented in the paper, and save them to file. Different Rosenzweig-Porter models are available, namely Gaussian RP, Bernoulli RP, and Log-normal RP (the same presented in the paper).

## rc.nb
Contains Mathematica code with analytical calculations for the Gaussian RP model, including both the self-consistent resonance counting and the approach based on Phys. Rev. E 98, 032139 (2018), with the ready-to-plot figures used in the paper.
