#!/bin/bash

# Define the path to your IPython notebook
NOTEBOOK_PATH="notebooks/Synthetic_Data_Creation.ipynb"

# Run the notebook using nbconvert
jupyter nbconvert --to notebook --execute "$NOTEBOOK_PATH" --output "$NOTEBOOK_PATH-executed"
