#!/usr/bin/env python

import sys
from team_code import training_code

if __name__ == "__main__":
    # Check if the minimum required arguments are provided
    if len(sys.argv) < 3:
        raise Exception(
            "Include the data and model folders as arguments, e.g., python train_model.py data model."
        )

    # Parse command-line arguments
    data_directory = sys.argv[1]
    model_directory = sys.argv[2]

    # Set default values for optional parameters
    seed = 2021
    alpha = 0.01
    kl_layers = 2
    splitfile = "split.json"
    lead = None

    # Override default values if additional arguments are provided
    if len(sys.argv) >= 4:
        seed = int(sys.argv[3])
    if len(sys.argv) >= 5:
        alpha = float(sys.argv[4])
    if len(sys.argv) >= 6:
        kl_layers = int(sys.argv[5])
    if len(sys.argv) >= 7:
        splitfile = sys.argv[6]
    if len(sys.argv) >= 8:
        lead = int(sys.argv[7])

    # Call the training function with parsed arguments
    training_code(
        data_directory=data_directory,
        model_directory=model_directory,
        seed=seed,
        alpha=alpha,
        kl_layers=kl_layers,
        splitfile=splitfile,
        lead=lead,
    )
