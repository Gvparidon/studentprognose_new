# individual.py

# --- Standard library ---
import os
import sys
import math
import json
import logging
import warnings
from datetime import date
from pathlib import Path

# --- Third-party libraries ---
import numpy as np
from numpy import linalg as LA
import pandas as pd
import yaml
import joblib
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from dotenv import load_dotenv

# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.load_data import (
    load_individual,
    load_distances,
    load_latest,
)
from scripts.helper import *
from cli import parse_args


# --- Warnings and logging setup ---
warnings.simplefilter("ignore", ConvergenceWarning)
logger = logging.getLogger(__name__)

# --- Environment setup ---
load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH")

# --- Constant variable names ---
GROUP_COLS = [
    "Collegejaar", "Croho groepeernaam", "Faculteit",
    "Examentype", "Herkomst"
]

CATEGORICAL_COLS = [
    "Opleiding",
    "Type vooropleiding",
    "Nationaliteit",
    "EER",
    "Geslacht"
]

NUMERIC_COLS = [
    "Sleutel_count",
    "is_numerus_fixus",
    "Afstand",
    "Deadlineweek"
]

WEEK_COL = ["Weeknummer"]

TARGET_COL = ['Inschrijfstatus']    


# --- Main individual class ---

class Individual():
    def __init__(self, data_individual, data_distances, data_latest, configuration):
        self.data_individual = data_individual
        self.data_distances = data_distances
        self.data_latest = data_latest
        self.configuration = configuration
        self.skip_years = 0
        self.pred_len = None

        # Cached xgboost models
        self.xgboost_models = {}

        # Backup data
        self.data_individual_backup = self.data_individual.copy()

        # Store variables
        self.preprocessed = False

    
    ### --- Helpers --- ###


    ### Main logic ###  
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the input data for further analysis.
        """

        def to_weeknummer(datum):
            try:
                day, month, year = map(int, datum.split("-"))
                return date(year, month, day).isocalendar()[1]
            except (AttributeError, ValueError):
                return np.nan

        def get_herkomst(nat, eer):
            if nat == "Nederlandse":
                return "NL"
            elif eer == "J":
                return "EER"
            return "Niet-EER"

        def get_deadlineweek(row):
            return row["Weeknummer"] == 17 and (
                row["Croho groepeernaam"] not in list(self.configuration["numerus_fixus"].keys())
                or row["Examentype"] != "Bachelor"
            )

        # 1. --- Load and clean base dataset ---
        df = self.data_individual.copy()
        df = df.drop(columns=["Aantal studenten"])

        # 2. --- Filter out specific English programme in 2021 ---
        mask = (
            (df["Croho groepeernaam"] == "B English Language and Culture")
            & (df["Collegejaar"] == 2021)
            & (df["Examentype"] != "Propedeuse Bachelor")
        )
        df = df[~mask]

        # 3. --- Add count of entries per key ---
        df["Sleutel_count"] = df.groupby(["Collegejaar", "Sleutel"])["Sleutel"].transform(
            "count"
        )

        # 4. --- Convert dates to week numbers ---
        df["Datum intrekking vooraanmelding"] = df["Datum intrekking vooraanmelding"].apply(
            to_weeknummer
        )
        df["Weeknummer"] = df["Datum Verzoek Inschr"].apply(to_weeknummer)

        # 5. --- Derive origin from nationality and EER flag ---
        df["Herkomst"] = df.apply(lambda x: get_herkomst(x["Nationaliteit"], x["EER"]), axis=1)

        # 6. --- Keep only entries with September or October intake ---
        df = df[
            df["Ingangsdatum"].str.contains("01-09-")
            | df["Ingangsdatum"].str.contains("01-10-")
        ]

        # 7. --- Update RU faculty name ---
        df["Faculteit"] = df["Faculteit"].replace(self.configuration["faculty"])

        # 8. --- Add numerus fixus flag ---
        df["is_numerus_fixus"] = (
            df["Croho groepeernaam"].isin(list(self.configuration["numerus_fixus"].keys()))
            & (df["Examentype"] == "Bachelor")
        ).astype(int)

        # 9. --- Normalize exam type names ---
        df["Examentype"] = df["Examentype"].replace("Propedeuse Bachelor", "Bachelor")

        # 10. --- Filter on valid enrollment status and exam types ---
        df = df[df["Inschrijfstatus"].notna()]
        df = df[df["Examentype"].isin(["Bachelor", "Master", "Pre-master"])]

        # 11. --- Collapse rare nationalities into 'Overig' ---
        counts = df["Nationaliteit"].value_counts()
        rare_values = counts[counts < 100].index
        df["Nationaliteit"] = df["Nationaliteit"].replace(rare_values, "Overig")

        # 12. --- Add distances if available ---
        if self.data_distances is not None:
            afstand_lookup = self.data_distances.set_index("Geverifieerd adres plaats")["Afstand"]
            df["Afstand"] = df["Geverifieerd adres plaats"].map(afstand_lookup)
        else:
            df["Afstand"] = np.nan

        # 13. --- Determine deadline week flag ---
        df["Deadlineweek"] = df.apply(get_deadlineweek, axis=1)

        # 14. --- Drop unneeded columns ---
        df = df.drop(columns=["Sleutel"])

        # 15. --- Special handling for pre-master entries ---
        premaster_mask = df["Examentype"] == "Pre-master"
        df.loc[
            premaster_mask, ["Is eerstejaars croho opleiding", "Is hogerejaars", "BBC ontvangen"]
        ] = [1, 0, 0]

        # 16. --- Final filtering on enrollment status ---
        df = df[
            (df["Is eerstejaars croho opleiding"] == 1)
            & (df["Is hogerejaars"] == 0)
            & (df["BBC ontvangen"] == 0)
        ]

        # 17. --- Final cleanup ---
        df = df[GROUP_COLS + CATEGORICAL_COLS + NUMERIC_COLS + WEEK_COL + TARGET_COL + ["Datum intrekking vooraanmelding"]]

        # 18. --- Store results ---
        self.data_individual_backup = self.data_individual.copy()
        self.data_individual = df
        self.preprocessed = True

        return df
    

    def predict_preapplicant_probabilities(self, predict_year: int, predict_week: int):
        """
        Predict the chance that someone actually enrolls for each individual.
        """
        
        # --- Data copy ---
        if not self.preprocessed:
            self.preprocess()

        df = self.data_individual.copy()

        # --- Data transformation ---

        # Get weeks to predict only
        weeks_to_predict = get_weeks_list(predict_week)
        df = df[df["Weeknummer"].isin(weeks_to_predict)].copy()

        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Train/test split
        test_mask = df["Collegejaar"] == predict_year
        train_mask = (df["Collegejaar"] < predict_year) & (
            df["Collegejaar"] >= self.configuration["start_year"]
        )

        train = df[train_mask]
        test = df[test_mask]

        # Filter out canceled registrations
        if predict_week <= 38:
            cancellation_filter = train["Datum intrekking vooraanmelding"].isna() | (
                (train["Datum intrekking vooraanmelding"] >= predict_week)
                & (train["Datum intrekking vooraanmelding"] < 39)
            )
        else:
            cancellation_filter = (
                train["Datum intrekking vooraanmelding"].isna()
                | (train["Datum intrekking vooraanmelding"] > predict_week)
                | (train["Datum intrekking vooraanmelding"] < 39)
            )
        train = train[cancellation_filter]

        # Target Variable Transformation 
        status_map = {
            "Ingeschreven": 1,
            "Uitgeschreven": 1,
            "Geannuleerd": 0,
            "Verzoek tot inschrijving": 0,
            "Studie gestaakt": 0,
            "Aanmelding vervolgen": 0,
        }
        train.loc[:, TARGET_COL[0]] = train[TARGET_COL[0]].map(status_map)

        # Feature & Preprocessing Setup
        X_train = train.drop(columns=[TARGET_COL[0]])
        y_train = train[TARGET_COL[0]]
        X_test = test.drop(columns=[TARGET_COL[0]])
        y_test = test[TARGET_COL[0]]

        # --- Model Training ---
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough",  NUMERIC_COLS),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    CATEGORICAL_COLS + GROUP_COLS + WEEK_COL,
                ),
            ],
            remainder="drop",
        )

        # Model Training and Prediction
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)


        model = XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=0)
        model.fit(X_train_transformed, y_train)

        probabilities = model.predict_proba(X_test_transformed)[:, 1]

        # Post-processing 
        is_cancelled_in_period = (test[TARGET_COL[0]] == "Geannuleerd") & (
            test["Datum intrekking vooraanmelding"].isin(weeks_to_predict)
        )

        final_predictions = np.where(is_cancelled_in_period.values, 0, probabilities)

        # impute the predictions back into the dataset

        import sklearn.metrics as metrics
        y_test = y_test.map(status_map)
        print('MAE:', metrics.mean_absolute_error(y_test, final_predictions))

        return final_predictions




        
    
        


if __name__ == "__main__":

    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # Load data
    individual_data = load_individual()
    distances = load_distances()
    latest_data = load_latest()

    # Initialize model
    individual_model = Individual(individual_data, distances, latest_data, configuration)

    individual_model.predict_preapplicant_probabilities(2024,10)