# ratio.py

# cumulative.py

# --- Standard library ---
import os
import sys
import math
import json
import logging
import warnings
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
from xgboost import XGBRegressor
from dotenv import load_dotenv

# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.load_data import (
    load_cumulative,
    load_student_numbers_first_years,
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

NUMERIC_COLS = [
    "Ongewogen vooraanmelders", "Gewogen vooraanmelders",
    "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"
]

WEEK_COL = ["Weeknummer"]

TARGET_COL = ['Aantal_studenten']    

RENAME_MAP = {
    "Type hoger onderwijs": "Examentype",
    "Groepeernaam Croho": "Croho groepeernaam",
}

# --- Main ratio class ---

class Ratio():
    def __init__(self, data_cumulative, data_studentcount, data_latest, configuration):
        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.data_latest = data_latest
        self.configuration = configuration
        self.skip_years = 0
        self.pred_len = None

        # Backup data
        self.data_cumulative_backup = self.data_cumulative.copy()

    # --------------------------------------------------
    # -- Preprocessing --
    # --------------------------------------------------
    def preprocess(self) -> pd.DataFrame:
        """
        Cleans, filters, aggregates, and merges cumulative pre-application data.
        This is the same function as in cumulative.py
        """
        
        # --- Data copy ---
        df = self.data_cumulative.copy()

        # 1. Rename columns
        df = df.rename(columns=RENAME_MAP)

        # 2. Convert numeric columns to float64
        for col in NUMERIC_COLS:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = pd.to_numeric(
                    df[col], 
                    decimal=',', 
                    thousands='.', 
                    errors='coerce'
                )

        df[NUMERIC_COLS] = df[NUMERIC_COLS].astype('float64')

        # 3. Filter for first-year and pre-master students
        mask = (df["Hogerejaars"] == "Nee") | (df["Examentype"] == "Pre-master")
        df = df[mask]

        # 4. Group and aggregate data
        processed_df = df.groupby(GROUP_COLS + WEEK_COL, as_index=False)[NUMERIC_COLS].sum()

        # 5. Merge with student count data (if it exists)
        if self.data_studentcount is not None:
            processed_df = processed_df.merge(
                self.data_studentcount,
                on=[col for col in GROUP_COLS if col != "Faculteit"],
                how="left",
            )
        
        # 6. Create the 'ts' (time series) target column by adding 'Gewogen vooraanmelders' and 'Inschrijvingen'
        processed_df["ts"] = (
            processed_df["Gewogen vooraanmelders"] + processed_df["Inschrijvingen"]
        )

        # 7. Standardize faculty codes
        faculty_transformation = self.configuration["faculty"]
        processed_df["Faculteit"] = processed_df["Faculteit"].replace(faculty_transformation)

        # 8. Set week 39 and week 40 to 0
        processed_df.loc[processed_df["Weeknummer"] == 39, "ts"] = 0
        processed_df.loc[processed_df["Weeknummer"] == 40, "ts"] = 0
        
        # 9. Final sorting, ordering, and duplicate removal
        
        # Determine the final column order, keeping original columns first
        final_cols_order = GROUP_COLS + WEEK_COL + NUMERIC_COLS
        existing_cols = set(final_cols_order)
        # Add any new columns from the merge and the 'ts' column
        new_cols = [col for col in processed_df.columns if col not in existing_cols]
        final_cols_order.extend(new_cols)

        processed_df = (
            processed_df.sort_values(by=GROUP_COLS + WEEK_COL, ignore_index=True)
            .drop_duplicates()
            [final_cols_order]  # Enforce consistent column order
        )

        # --- Update Instance Attributes ---
        self.data_cumulative_backup = self.data_cumulative.copy()
        self.data_cumulative = processed_df

        return self.data_cumulative

    # --------------------------------------------------
    # -- Predicting with the ratio method --
    # --------------------------------------------------

    ### --- Helpers --- ###
    def _filter_data(self, df: pd.DataFrame, predict_year: int, predict_week: int, programme: str, examentype: str, herkomst: str) -> pd.DataFrame:
        df = df[df["Collegejaar"] >= self.configuration["start_year"]]
        
        filtered = df[
            (df["Herkomst"] == herkomst)
            & (df["Collegejaar"] <= predict_year)
            & (df['Weeknummer'] == predict_week)
            & (df["Croho groepeernaam"] == programme)
            & (df["Examentype"] == examentype)
        ]
        return filtered
    
    def _get_ratio(self, df: pd.DataFrame, predict_year: int, training_years: int = 3) -> float:
        for years in range(training_years, 0, -1):  
            subset = df[df["Collegejaar"] >= predict_year - years].copy()
            ts_sum = subset["ts"].fillna(0).sum()
            student_sum = subset["Aantal_studenten"].fillna(0).sum()
        
        if student_sum > 0:  
            return ts_sum / student_sum 
        
        return 0.5 # some default value
    
    ### --- Main logic --- ###
    def predict_with_ratio(self,
        programme: str,
        herkomst: str,
        examentype: str,
        predict_year: int,
        predict_week: int
    ) -> int:
        """
        Predict the inflow of students based on the ratio of pre-applicants.
        """
        
        # --- Data copy ---
        df = self.data_cumulative.copy()

        # --- Filter data ---
        df = self._filter_data(df, predict_year, predict_week, programme, examentype, herkomst)

        # --- Get ratio ---
        ratio = self._get_ratio(df, predict_year)
        # --- Predict ---
        prediction = df.loc[df["Collegejaar"] == predict_year, "ts"] / ratio      

        return round(prediction)

        


if __name__ == "__main__":
    # Load configuration
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # Load data
    cumulative_data = load_cumulative()
    student_counts = load_student_numbers_first_years()
    latest_data = load_latest()

    # Initialize model
    ratio_model = Ratio(cumulative_data, student_counts, latest_data, configuration)

    ratio_model.preprocess()

    ratio_model.predict_with_ratio(
        programme="B Sociologie",
        herkomst="NL",
        examentype="Bachelor",
        predict_year=2024,
        predict_week=10,
    )
    

    