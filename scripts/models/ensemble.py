# ensemble.py

# --- Standard library ---
import os
import sys
import logging
import time
import warnings
from pathlib import Path

# --- Third-party libraries ---
import numpy as np
import pandas as pd
import yaml
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from dotenv import load_dotenv

# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.load_data import (
    load_latest,
)
from cli import parse_args
from scripts.standalone.evaluate_results import ModelEvaluator

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

PREDICTION_COLS = ['SARIMA_cumulative', 'SARIMA_individual', 'Prognose_ratio']

WEEK_COL = ["Weeknummer"]

TARGET_COL = ['Aantal_studenten']  

# --- Main ratio class ---

class Ensemble():
    def __init__(self, data_latest, configuration):
        self.data_latest = data_latest
        self.configuration = configuration

        # Cached models
        self.lr_models = {}
        self.xgb_models = {}

    # --------------------------------------------------
    # -- Predicting using the ensemble method --
    # --------------------------------------------------

    ### --- Helpers --- ###
    def _filter_data(self, df: pd.DataFrame, predict_year: int, predict_week: int, programme: str, examentype: str, herkomst: str) -> pd.DataFrame:
        df = df[df["Collegejaar"] >= self.configuration["ensemble_start_year"]]
        
        filtered = df[
            (df["Herkomst"] == herkomst)
            & (df["Collegejaar"] <= predict_year)
            & (df['Weeknummer'] == predict_week)
            & (df["Examentype"] == examentype)
        ]
        return filtered

    def _clean_data(self, df: pd.DataFrame, predict_year: int) -> pd.DataFrame:
        # Remove rows where all columns are 0
        df = df.drop(df.index[df[PREDICTION_COLS].sum(axis=1) == 0])

        # Remove rows where target is NaN, but only if it's NOT the predict_year
        df = df[~(df['Collegejaar'] != predict_year) | df[TARGET_COL[0]].notna()]

        # Replace Inf/-Inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill NaN with column mean
        for col in PREDICTION_COLS:
            df[col] = df[col].fillna(df[col].mean())

        return df

    def _build_models(self, X_train, y_train, model_key):

        lr_model = None
        xgb_model = None

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", PREDICTION_COLS),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                GROUP_COLS),
            ],
            remainder="drop",
        )

        # --- Build models ---
        if model_key in self.lr_models:
            lr_model = self.lr_models[model_key]
        else:
            # Linear Regression pipeline
            lr_model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression())
            ])
            lr_model.fit(X_train, y_train)
            self.lr_models[model_key] = lr_model
        
        if model_key in self.xgb_models:
            xgb_model = self.xgb_models[model_key]
        else:
            # XGBRegressor pipeline
            xgb_model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", XGBRegressor(learning_rate=0.1, random_state=0, n_jobs=-1))
            ])
            xgb_model.fit(X_train, y_train)
            self.xgb_models[model_key] = xgb_model

        return lr_model, xgb_model
    
    
    ### --- Main logic --- ###
    def predict_using_ensemble_method(self,
        programme: str,
        herkomst: str,
        examentype: str,
        predict_year: int,
        predict_week: int,
        verbose: bool
    ) -> int:
        """
        Predict the inflow of students based on the ratio of pre-applicants.
        """
        
        # --- Data copy ---
        df = self.data_latest.copy()

        # --- Filter data ---
        df = self._filter_data(df, predict_year, predict_week, programme, examentype, herkomst)

        # --- Select only the relevant columns ---
        df = df[GROUP_COLS + PREDICTION_COLS + TARGET_COL]

        # --- Clean to make sure for no corrupt values or empty values ---
        df = self._clean_data(df, predict_year)

        # --- Make a train and test split ---
        train = df[df["Collegejaar"] < predict_year]
        test = df[df["Collegejaar"] == predict_year]
        test = test[test["Croho groepeernaam"] == programme]

        x_train = train[PREDICTION_COLS + GROUP_COLS]
        y_train = train[TARGET_COL]

        x_test = test[PREDICTION_COLS + GROUP_COLS]

        # --- Make predictions --
        model_key = f"{herkomst}_{examentype}"
        try:
            lr_model, xgb_model = self._build_models(x_train, y_train, model_key)
            # Fit LR
            lr_preds = lr_model.predict(x_test)

            # Fit XGB
            xgb_preds = xgb_model.predict(x_test)

            prediction = round((lr_preds[0,0] + xgb_preds[0]) / 2)

            # Make sure prediction is not negative
            prediction = max(0, prediction)
        except Exception:
            prediction = 0
        
        if verbose:
            print(
                f"Ensemble prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {prediction}"
            )

        # --- Return the prediction --- 
        return prediction


    # --------------------------------------------------
    # -- Full prediction loop --
    # --------------------------------------------------

    def run_full_prediction_loop(self, predict_year: int, predict_week: int, write_file: bool, verbose: bool, args = None):

        """
        Run the full prediction loop for all years and weeks.
        """
        logger.info('Running Ensemble prediction loop')

        # --- Apply filtering from configuration ---
        filtering = self.configuration["filtering"]

        # --- Filter data ---
        mask = np.ones(len(self.data_latest), dtype=bool) 

        # --- Apply conditional filters from configuration ---
        if filtering["programme"]:
            mask &= self.data_latest["Croho groepeernaam"].isin(filtering["programme"])
        if filtering["herkomst"]:
            mask &= self.data_latest["Herkomst"].isin(filtering["herkomst"])
        if filtering["examentype"]:
            mask &= self.data_latest["Examentype"].isin(filtering["examentype"])
        
        # --- Apply year and week filters ---
        mask &= self.data_latest["Collegejaar"] == predict_year
        mask &= self.data_latest["Weeknummer"] == predict_week

        # --- Apply mask ---
        prediction_df = self.data_latest.loc[mask, GROUP_COLS + WEEK_COL].copy()

        # --- Prediction ---
        prediction_df["Ensemble_prediction"] = prediction_df.apply(
            lambda row: self.predict_using_ensemble_method(
                programme=row["Croho groepeernaam"],
                herkomst=row["Herkomst"],
                examentype=row["Examentype"],
                predict_year=predict_year,
                predict_week=predict_week,
                verbose=verbose
            ),
            axis=1,
            result_type="expand"
        )

        # --- Map ratio predictions back into latest data ---
        prediction_map = prediction_df.set_index(GROUP_COLS + WEEK_COL)["Ensemble_prediction"].to_dict()
        self.data_latest["Ensemble_prediction"] = [
            prediction_map.get(tuple(row[col] for col in GROUP_COLS + WEEK_COL), row["Ensemble_prediction"] )
            for _, row in self.data_latest.iterrows()
        ]

        # --- Write the file ---
        if write_file:
            output_path = self.configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
            self.data_latest.to_excel(output_path, index=False, engine="xlsxwriter")

        # --- Evaluate predictions (if required) ---
        if args.evaluate:
            evaluator = ModelEvaluator(
                self.data_latest,
                actual_col="Aantal_studenten",
                pred_col="Ensemble_prediction",
                baseline_col="Prognose_ratio",
                configuration=self.configuration,
                args=args
            )

            evaluator.print_evaluation_summary(print_programmes=args.print_programmes)

        logger.info('Ensemble prediction done')

# --- Main function ---
def main():
    # --- Parse arguments ---
    args = parse_args()

    # --- Load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # --- Load data ---
    latest_data = load_latest()

    # --- Initialize model ---
    ensemble_model = Ensemble(latest_data, configuration)

    # --- Run prediction loop ---
    for year in args.years:
        for week in args.weeks:
            ensemble_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                verbose=args.verbose,
                args=args
            )


if __name__ == "__main__":
    main()
    

    