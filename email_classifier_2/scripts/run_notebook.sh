#!/bin/bash

# Define the path to your IPython notebook
NOTEBOOK_PATH="notebooks/email_eda.ipynb"

# Run the notebook using nbconvert
jupyter nbconvert --to notebook --execute "$NOTEBOOK_PATH" --output "$NOTEBOOK_PATH-executed"
