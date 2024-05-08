Spam Classifier
==============================

Design and implement a simple email spam classifier on real estate dataset.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Data analysis
For the data processing, and analysis.

```commandline
cd notebooks
jupyter notebook
```
The above command will open jupyter notebook. Select the spam_classifier.ipynb notebook and execute the cells to view the data analysis plots.

## Data information

```commandline
spam class - ~800 samples
ham class - ~200 samples
```

## How to train?
1. Install the requirements
    ```commandline
   pip install -r requirements.txt
   ```
2. Train the model
    ```
   cd src/models
   python train_model.py --data_file ../../data/preprocessed_email_data.csv
   ```
3. Start the Mlflow ui in a new terminal tab
    ```
   mlflow ui
   ```

## Sample reports and metrics
   ![](reports\figures\LogisticRegression_5_confusion_matrix.png)

