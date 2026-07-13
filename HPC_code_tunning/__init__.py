"""HPC_code_tunning — GPU hyperparameter/feature-selection tuning for the TDEW model.

Tunes the local day-of-year half-window ``h`` and does multitask backward-stepwise
feature selection (LOYOCV cosine-skill objective) per SENAMHI climatic zone, then trains
the full grid per zone with the selected recipe. See ``../.claude`` plan and README.md.
"""
