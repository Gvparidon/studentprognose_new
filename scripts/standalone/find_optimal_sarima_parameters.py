# find_optimal_sarima_parameters.py

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


# --- Warnings and logging setup ---
warnings.simplefilter("ignore", ConvergenceWarning)
logger = logging.getLogger(__name__)

# --- Environment setup ---
load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH")


class SarimaParameterFinder:
    """
    A class to find the optimal SARIMA parameters for time series data
    grouped by different program categories.
    """

    def __init__(self, configuration, cumulative_model = None, individual_model = None):
        """
        Initializes the SarimaParameterFinder.
        """
        self.configuration = configuration
        self.cumulative_model = cumulative_model
        self.individual_model = individual_model
        self.results_df = None

    def _create_time_series(self, opleiding, herkomst, examentype):
        """
        Filters the data for a specific combination and creates a time series.

        Args:
            opleiding (str): The program name ('Croho groepeernaam').
            herkomst (str): The origin of the student ('Herkomst').
            examentype (str): The exam type ('Examentype').

        Returns:
            np.array: A NumPy array representing the time series.
        """
        # Filter data for the specific combination
        filtered_data = self.data[
            (self.data["Croho groepeernaam"] == opleiding)
            & (self.data["Herkomst"] == herkomst)
            & (self.data["Examentype"] == examentype)
        ].copy()

        # Ensure week numbers are treated as ordered categories
        week_order = get_all_weeks_ordered()
        filtered_data["Weeknummer"] = filtered_data["Weeknummer"].astype(str)
        filtered_data["Weeknummer"] = pd.Categorical(
            filtered_data["Weeknummer"], categories=week_order, ordered=True
        )

        # Sort values to ensure correct time series sequence
        sorted_data = filtered_data.sort_values(by=["Collegejaar", "Weeknummer"]).reset_index(
            drop=True
        )

        # Extract the time series and remove any NaN values
        ts = np.array(sorted_data["Gewogen vooraanmelders"])
        ts = ts[~np.isnan(ts)]

        return ts

    @staticmethod
    def _find_optimal_parameters(ts_data):
        """
        Performs a grid search to find the best SARIMA parameters (p,d,q)(P,D,Q)s.

        Args:
            ts_data (np.array): The time series data.

        Returns:
            tuple: A tuple containing the optimal parameters (p, d, q, P, D, Q).
        """
        # Define parameter ranges for the grid search
        p = d = q = range(0, 3)
        P = D = Q = range(0, 2)
        s = 52  # Seasonal cycle length (52 weeks in a year)

        # Generate all possible parameter combinations
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(P, D, Q))]

        best_aic = np.inf
        best_params = None
        best_seasonal_params = None

        # Grid search for the best parameters based on AIC
        for param in pdq:
            for seasonal_param in seasonal_pdq:
                try:
                    model = sm.tsa.statespace.SARIMAX(
                        ts_data,
                        order=param,
                        seasonal_order=seasonal_param,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    results = model.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = param
                        best_seasonal_params = seasonal_param
                except Exception:
                    continue

        if not best_params:
            return 0, 0, 0, 0, 0, 0

        p, d, q = best_params
        P, D, Q, s = best_seasonal_params

        return p, d, q, P, D, Q

    def _run_single_combination(self, row):
        """
        Processes a single row (combination) to find its optimal SARIMA parameters.
        This function is designed to be used with pandas .apply().
        """
        opleiding = row["Croho groepeernaam"]
        herkomst = row["Herkomst"]
        examentype = row["Examentype"]

        print(f"Running for: {opleiding} | {herkomst} | {examentype}")

        # Step 1: Create the time series for the combination
        ts = self._create_time_series(opleiding, herkomst, examentype)

        if len(ts) < 1:
            print(" -> Skipping, not enough data.")
            return pd.Series({"p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0})

        # Step 2: Find the optimal parameters
        p, d, q, P, D, Q = self._find_optimal_parameters(ts)

        return pd.Series({"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q})

    def run_optimization(self):
        """
        Runs the entire optimization process for all unique program combinations.
        """
        print("\nStarting SARIMA parameter optimization for all combinations...")

        # Create a DataFrame with unique combinations of interest
        required_cols = ["Croho groepeernaam", "Herkomst", "Examentype"]
        combinations_df = self.data[required_cols].drop_duplicates().reset_index(drop=True)

        # Apply the processing function to each row (combination)
        # This will calculate the parameters for each unique combination
        param_results = combinations_df.apply(self._run_single_combination, axis=1)

        # Combine the original combinations with their new parameters
        self.results_df = pd.concat([combinations_df, param_results], axis=1)

        print("\nOptimization complete.")
        return self.results_df

    def save_results(self):
        """
        Saves the resulting DataFrame with SARIMA parameters to an Excel file.
        """
        if self.results_df is not None:
            output_path = self.config["paths"]["path_sarima_paramters"]
            print(f"Saving results to {output_path}...")
            self.results_df.to_excel(output_path, index=False)
            print("Results saved successfully.")
        else:
            print("No results to save. Please run the optimization first.")


if __name__ == "__main__":
    # Define the path to your configuration file
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f) 

    # Step 1: Create an instance of the finder
    finder = SarimaParameterFinder(configuration)

    # Step 2: Run the optimization for all program combinations
    finder.run_optimization()

    # Step 3: Save the final results to an Excel file
    finder.save_results()
