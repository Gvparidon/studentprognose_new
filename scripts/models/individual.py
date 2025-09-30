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

        # Store processing variables
        self.preprocessed = False
        self.predicted = False

    
        ### --- Helpers --- ###
    
    def _get_transformed_data(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Drops duplicates, filters data from start_year onwards, and transforms from long to wide format.

        Args:
            data (pd.DataFrame): Input DataFrame.
            column (str): Column to pivot.

        Returns:
            pd.DataFrame: Transformed DataFrame with weeks as columns.
        """
        # Drop duplicates and filter years
        df = data.drop_duplicates()
        df = df[df["Collegejaar"] >= self.configuration["start_year"]]

        # Keep relevant columns
        df = df.loc[:, GROUP_COLS + [column, "Weeknummer"]].drop_duplicates()

        df[column] = pd.to_numeric(df[column], errors="coerce")

        # Pivot to wide format
        df = df.pivot_table(
            index=GROUP_COLS,
            columns="Weeknummer",
            values=column,
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        # Flatten column names and reorder based on valid weeks
        df.columns = df.columns.map(str)
        valid_weeks = get_all_weeks_valid(df.columns)
        df = df[GROUP_COLS + valid_weeks]

        # Reshape back to long format
        df = df.melt(
            id_vars=GROUP_COLS,
            value_vars=[w for w in valid_weeks if w in df.columns],
            var_name="Weeknummer",
            value_name=column,
            )

        df[TARGET_COL[0]] = df.groupby(GROUP_COLS)[column].cumsum()

        return df


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
    

    def predict_preapplicant_probabilities(self, predict_year: int, predict_week: int) -> pd.DataFrame:
        """
        Predict the probability that a pre-applicant will enroll for each individual. Returns the updated dataset.
        """

        # --- Preprocess if not done ---
        if not self.preprocessed:
            self.preprocess()

        df = self.data_individual.copy()

        # --- Filter by weeks to predict ---
        weeks_to_predict = get_weeks_list(predict_week)
        df = df[df["Weeknummer"].isin(weeks_to_predict)].copy()

        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # --- Train/Test Split ---
        train_mask = (df["Collegejaar"] < predict_year) & (df["Collegejaar"] >= self.configuration["start_year"])
        test_mask = df["Collegejaar"] == predict_year

        train = df[train_mask].copy()
        test = df[test_mask].copy()

        # --- Filter out cancelled registrations ---
        if predict_week <= 38:
            cancellation_filter = train["Datum intrekking vooraanmelding"].isna() | (
                (train["Datum intrekking vooraanmelding"] >= predict_week) & (train["Datum intrekking vooraanmelding"] < 39)
            )
        else:
            cancellation_filter = (
                train["Datum intrekking vooraanmelding"].isna()
                | (train["Datum intrekking vooraanmelding"] > predict_week)
                | (train["Datum intrekking vooraanmelding"] < 39)
            )
        train = train[cancellation_filter]

        # --- Target Mapping ---
        status_map = {
            "Ingeschreven": 1,
            "Uitgeschreven": 1,
            "Geannuleerd": 0,
            "Verzoek tot inschrijving": 0,
            "Studie gestaakt": 0,
            "Aanmelding vervolgen": 0,
        }
        train[TARGET_COL[0]] = train[TARGET_COL[0]].map(status_map)
        test[TARGET_COL[0]] = test[TARGET_COL[0]].map(status_map)

        # --- Features and Labels ---
        X_train = train.drop(columns=[TARGET_COL[0]])
        y_train = train[TARGET_COL[0]]
        X_test = test.drop(columns=[TARGET_COL[0]])

        # --- Preprocessing Pipeline ---
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", NUMERIC_COLS),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_COLS + GROUP_COLS + WEEK_COL),
            ],
            remainder="drop",
        )

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # --- Model Training ---
        model = XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=0)
        model.fit(X_train_transformed, y_train)

        # --- Predictions ---
        probabilities = model.predict_proba(X_test_transformed)[:, 1]

        # --- Vectorized Post-processing ---
        cancelled_flag = (test[TARGET_COL[0]] == 0)  # already mapped
        week_mask = np.isin(test["Datum intrekking vooraanmelding"].to_numpy(), weeks_to_predict)
        final_mask = cancelled_flag.to_numpy() & week_mask
        final_predictions = np.where(final_mask, 0, probabilities)

        # --- Assign predictions back safely ---
        self.data_individual.loc[:, TARGET_COL[0]] = self.data_individual[TARGET_COL[0]].map(status_map)
        self.data_individual.loc[test.index, TARGET_COL[0]] = final_predictions

        self.predicted = True

        return self.data_individual


    def predict_inflow_with_sarima(self,
        programme: str,
        herkomst: str,
        examentype: str,
        predict_year: int,
        predict_week: int,
        refit: bool = False
    ) -> list[float]:
        """
        Predicts the number of students using SARIMA.
        """

        # --------------------------------------------------
        # -- Helper functions --
        # --------------------------------------------------
        def _filter_data(data: pd.DataFrame) -> pd.DataFrame:
            data = self._get_transformed_data(data, TARGET_COL[0])
            filtered = data[
                (data["Herkomst"] == herkomst)
                & (data["Collegejaar"] <= predict_year)
                & (data["Croho groepeernaam"] == programme)
                & (data["Examentype"] == examentype)
            ]
            return filtered

        def _create_time_series(data: pd.DataFrame, pred_len: int) -> np.ndarray:
            ts_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
            return ts_data[:-pred_len]

        def _fit_sarima(ts_data: np.ndarray, model_name: str, seasonal_order=(1, 1, 1, 52)):
            model_path = os.path.join(configuration["other_paths"]["cumulative_sarima_models"].replace("${root_path}", ROOT_PATH), f"{model_name}.json")

            sarimax_args = dict(
                order=(1, 1, 1) if len(ts_data) < 52 else (1, 0, 1),
                seasonal_order=(0, 0, 0, 0) if len(ts_data) < 52 else seasonal_order,
                trend="c" if len(ts_data) < 52 else None,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            model = sm.tsa.SARIMAX(ts_data, **sarimax_args)

            if os.path.exists(model_path) and not refit:
                with open(model_path, "r") as f:
                    model_data = json.load(f)
                loaded_params = model_data["model_params"]
                trained_year = model_data["trained_year"]

                if predict_year > trained_year:
                    fitted_model = model.fit(disp=False)
                else:
                    param_array = [loaded_params[name] for name in model.param_names]
                    fitted_model = model.fit(start_params=param_array, disp=False)
            else:
                fitted_model = model.fit(disp=False)

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "w") as f:
                json.dump(
                    {"trained_year": predict_year, "model_params": dict(zip(fitted_model.param_names, fitted_model.params))},
                    f, indent=4
                )

            return fitted_model

        # --------------------------------------------------
        # -- Main logic --
        # --------------------------------------------------
        if not self.predicted:
            self.predict_preapplicant_probabilities(predict_year, predict_week)

        pred_len = get_pred_len(predict_week)

        data = _filter_data(self.data_individual.copy())
        ts_data = _create_time_series(data, pred_len)
        model_name = f"{programme}{herkomst}{examentype}"
        print(ts_data)
        return 0
        results = _fit_sarima(ts_data, model_name)
        forecast = results.forecast(steps=pred_len).tolist()

        def filter_data(df, programme, herkomst, examentype, year, max_year):
            """Filter dataset by program, origin, exam type, and year."""
            df = df[df["Herkomst"] == herkomst]
            df = df[df["Croho groepeernaam"] == programme]
            df = df[df["Examentype"] == examentype]
            if year != max_year:
                df = df[df["Collegejaar"] <= year]
            return df

        def is_deadline_week(week, croho, ex_type):
            """Returns 1 if it's a deadline week for a given bachelor program."""
            if ex_type == "Bachelor":
                if week in [16, 17] and croho not in self.numerus_fixus_list:
                    return 1
                elif week in [1, 2] and croho in self.numerus_fixus_list:
                    return 1
            return 0

        # Filter data
        data = filter_data(data, programme, herkomst, examentype, self.predict_year, self.max_year)
        if data_exog is not None:
            data_exog = filter_data(
                data_exog, programme, herkomst, examentype, self.predict_year, self.max_year
            )
            data_exog["Deadline"] = data_exog.apply(
                lambda x: is_deadline_week(
                    x["Weeknummer"], x["Croho groepeernaam"], x["Examentype"]
                ),
                axis=1,
            )
            data_exog = transform_data(data_exog, "Deadline")

        # Shortcut for week 38 (no prediction needed)
        if self.predict_week == 38:
            ts_values = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
            return ts_values[-1] if len(ts_values) else np.nan

        # Determine prediction length
        predict_week = int(self.predict_week)
        pred_len = (38 + 52 - predict_week) if predict_week > 38 else (38 - predict_week)

        def extract_time_series(df, pred_len):
            """Extracts the time series for training."""
            ts = df.loc[:, get_all_weeks_valid(df.columns)].values.flatten()
            return ts[:-pred_len] if len(ts) > pred_len else np.array([])

        def extract_exogenous(df, pred_len):
            """Extracts training and test exogenous series."""
            exg = df.loc[:, get_all_weeks_valid(df.columns)].values.flatten()
            return exg[:-pred_len], exg[-pred_len:]

        ts_data = extract_time_series(data, pred_len)
        if ts_data.size == 0:
            return np.nan

        # Prepare exogenous if available
        if data_exog is not None:
            exog_train, exog_test = extract_exogenous(data_exog, pred_len)
        else:
            exog_train = exog_test = None

        # Choose SARIMA model based on program type and deadline proximity
        try:
            deadline_weeks = [17, 18, 19, 20, 21]
            is_bachelor_near_deadline = (
                programme.startswith("B") and predict_week in deadline_weeks
            )

            model = sm.tsa.statespace.SARIMAX(
                ts_data,
                order=(1, 0, 1) if is_bachelor_near_deadline else (1, 1, 1),
                seasonal_order=(1, 1, 1 if is_bachelor_near_deadline else 0, 52),
                exog=exog_train,
            )
            results = model.fit(disp=0)

            forecast = results.forecast(
                steps=pred_len, exog=exog_test if exog_test is not None else None
            )
            return forecast[-1]

        except (LA.LinAlgError, IndexError, ValueError) as e:
            print(f"Model error on: {programme}, {herkomst}\n{e}")
            return np.nan

        except KeyError as e:
            print(f"Key error on: {programme}, {herkomst}\n{e}")
            return np.nan

        




        
    
        


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

    individual_model.predict_inflow_with_sarima(programme="B Sociologie", herkomst="NL", examentype="Bachelor", predict_year=2024, predict_week=10)