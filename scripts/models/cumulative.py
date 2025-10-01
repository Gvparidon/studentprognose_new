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

# --- Main cumulative class ---

class Cumulative():
    def __init__(self, data_cumulative, data_studentcount, data_latest, configuration):
        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.data_latest = data_latest
        self.configuration = configuration
        self.skip_years = 0
        self.pred_len = None

        # Cached xgboost models
        self.xgboost_models = {}

        # Backup data
        self.data_cumulative_backup = self.data_cumulative.copy()

    ### --- Helpers --- ###
    
    def _get_transformed_data(self, data: pd.DataFrame, column: str = "ts") -> pd.DataFrame:
        """
        Drops duplicates, filters data from start_year onwards, and transforms from long to wide format.

        Args:
            data (pd.DataFrame): Input DataFrame.
            column (str): Column to pivot (default: "ts").

        Returns:
            pd.DataFrame: Transformed DataFrame with weeks as columns.
        """
        # Drop duplicates and filter years
        df = data.drop_duplicates()
        df = df[df["Collegejaar"] >= self.configuration["start_year"]]

        # Keep relevant columns
        df = df.loc[:, GROUP_COLS + TARGET_COL + [column, "Weeknummer"]].drop_duplicates()

        # Pivot to wide format
        df_wide = df.pivot_table(
            index=GROUP_COLS + TARGET_COL,
            columns="Weeknummer",
            values=column,
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        # Flatten column names and reorder based on valid weeks
        df_wide.columns = df_wide.columns.map(str)
        valid_weeks = get_all_weeks_valid(df_wide.columns)
        df_wide = df_wide[GROUP_COLS + TARGET_COL + valid_weeks]

        return df_wide


    ### Main logic ###    

    def preprocess(self) -> pd.DataFrame:
        """
        Cleans, filters, aggregates, and merges cumulative pre-application data.
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


    def predict_preapplicants_with_sarima(
        self,
        programme: str,
        herkomst: str,
        examentype: str,
        predict_year: int,
        predict_week: int,
        return_values: str = "weighted + applications",
        refit: bool = False
    ) -> list[float]:
        """
        Predict pre-registrations with SARIMA per programme/origin/week.

        Parameters:
            return_values: str
                "weighted + applications" or "weighted only"
            refit: bool
                Whether to refit the SARIMA model or not
        """

        # --------------------------------------------------
        # -- Helper functions --
        # --------------------------------------------------
        def _filter_data(data: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
            if weighted:
                data = self._get_transformed_data(data, "Gewogen vooraanmelders")
            else:
                data = self._get_transformed_data(data)
            filtered = data[
                (data["Herkomst"] == herkomst)
                & (data["Collegejaar"] <= predict_year)
                & (data["Croho groepeernaam"] == programme)
                & (data["Examentype"] == examentype)
            ]
            # Ensure week column exists
            if "39" not in filtered.columns:
                filtered["39"] = 0
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
        self.data_cumulative = self.data_cumulative.astype({"Weeknummer": "int32", "Collegejaar": "int32"})
        pred_len = get_pred_len(predict_week)

        try:
            if return_values == "weighted + applications":
                data = _filter_data(self.data_cumulative.copy())
                ts_data = _create_time_series(data, pred_len)
                model_name = f"{programme}{herkomst}{examentype}"
                results = _fit_sarima(ts_data, model_name)
                return results.forecast(steps=pred_len).tolist()

            elif return_values == "weighted only" and is_current_week(predict_year, predict_week):
                data_weighted = _filter_data(self.data_cumulative.copy(), weighted=True)
                ts_data_weighted = _create_time_series(data_weighted, pred_len)
                model_name = f"{programme}{herkomst}{examentype}_weighted"
                weighted_results = _fit_sarima(ts_data_weighted, model_name)
                return weighted_results.forecast(steps=pred_len).tolist()

            return []

        except (LA.LinAlgError, IndexError, ValueError) as error:
            logger.error("SARIMA error on: %s, %s, %s. Error: %s", programme, herkomst, examentype, error)
            return []


    def predict_students_with_preapplicants(
        self,
        programme: str,
        herkomst: str,
        examentype: str,
        predict_year: int,
        predict_week: int,
        refit_sarima: bool = False
    ) -> int:
        """
        Predict the inflow of students based on the number of pre-applicants.
        """
        # --------------------------------------------------
        # -- Helper --
        # --------------------------------------------------
        def _build_model(X_train, y_train, model_key):
            """
            Trains a model if not already in cache, otherwise returns the cached model.
            """
            if model_key in self.xgboost_models:
                return self.xgboost_models[model_key]

            # Define preprocessing and pipeline (as you did before)
            numeric_cols = ["Collegejaar"] + [str(x) for x in get_weeks_list(38)]
            categorical_cols = ["Examentype", "Faculteit", "Croho groepeernaam", "Herkomst"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ]
            )

            # Define pipeline with preprocessing + model
            model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", XGBRegressor(learning_rate=0.25, random_state=0, n_jobs=-1)),
                ]
            )

            # Fit and store the model
            model.fit(X_train, y_train)
            self.xgboost_models[model_key] = model
            return model

        # --------------------------------------------------
        # -- Data processing --
        # --------------------------------------------------

        if self.data_studentcount is None:
            raise FileNotFoundError("Student count is required")

        # Predict pre-applicants with SARIMA
        predicted_preapplicants = self.predict_preapplicants_with_sarima(
            programme, herkomst, examentype, predict_year, predict_week, return_values="weighted + applications", refit=refit_sarima
        )

        # Transform data
        data = self._get_transformed_data(self.data_cumulative, "ts")

        # Add predictions into the dataset
        prediction_weeks = get_prediction_weeks_list(predict_week)
        for week, value in zip(prediction_weeks, predicted_preapplicants):
            data.loc[data["Collegejaar"] == predict_year, str(week)] = value

        # Train/test split
        train = data[data["Collegejaar"] < predict_year].drop_duplicates().reset_index(drop=True)
        test = data[
            (data["Collegejaar"] == predict_year)
            & (data["Herkomst"] == herkomst)
            & (data["Croho groepeernaam"] == programme)
            & (data["Examentype"] == examentype)
        ].drop(columns=["Aantal_studenten"])

        # Modify train so it only consist of the examtype (and NF seperate)
        numerus_fixus = self.configuration["numerus_fixus"]
        if programme in list(numerus_fixus.keys()) and examentype == "Bachelor":
            train = train[train["Croho groepeernaam"] == programme]
        else:
            train = train[
                (train["Examentype"] == examentype) &
                (train["Herkomst"] == herkomst) &
                (
                    (~train["Croho groepeernaam"].isin(numerus_fixus.keys()))
                )
            ]

        # Separate features/target
        X_train = train.drop(columns=["Aantal_studenten"])
        y_train = train["Aantal_studenten"]

        # --------------------------------------------------
        # -- XGBOOST prediction --
        # --------------------------------------------------
        model_key = f"{examentype}_{herkomst}"
        if programme in list(self.configuration["numerus_fixus"].keys()):
             model_key = programme 

        # Get the appropriate model (either from cache or by training it now)
        model = _build_model(X_train, y_train, model_key)
        prediction = model.predict(test).round().astype(int)

        print(
            f"Cumulative prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {prediction[0]}"
        )

        return int(prediction[0]) if len(prediction) else 0

    
    def run_full_prediction_loop(self, predict_year, predict_week, skip_years=0):
        """
        Run the full prediction loop for all years and weeks.
        """

        # 1. Preprocess data
        self.preprocess()

        # 2. Apply filtering from configuration
        filtering = self.configuration["filtering"]

        # Filter data 
        mask = np.ones(len(self.data_cumulative), dtype=bool) 

        # Apply conditional filters from configuration
        if filtering["programme"]:
            mask &= self.data_cumulative["Croho groepeernaam"].isin(filtering["programme"])
        if filtering["herkomst"]:
            mask &= self.data_cumulative["Herkomst"].isin(filtering["herkomst"])
        if filtering["examentype"]:
            mask &= self.data_cumulative["Examentype"].isin(filtering["examentype"])
        
        # Apply year and week filters
        mask &= self.data_cumulative["Collegejaar"] == predict_year
        mask &= self.data_cumulative["Weeknummer"] == predict_week

        # Apply mask
        prediction_df = self.data_cumulative.loc[mask, GROUP_COLS + WEEK_COL].copy()

        # 3. Parallel prediction

        # Split the DataFrame into smaller chunks for parallel processing
        nr_CPU_cores = os.cpu_count() or 1
        chunk_size = math.ceil(len(prediction_df) / nr_CPU_cores)

        chunks = [
            prediction_df.iloc[i : i + chunk_size] for i in range(0, len(prediction_df), chunk_size)
        ]

        # Helper functions to apply to each row tuple
        def predict_students_row(row_tuple):
            return self.predict_students_with_preapplicants(
                programme=row_tuple._1,       # Croho groepeernaam
                herkomst=row_tuple.Herkomst,
                examentype=row_tuple.Examentype,
                predict_year=row_tuple.Collegejaar,
                predict_week=row_tuple.Weeknummer,
            )

        def predict_preapplicants_row(row_tuple):
            return self.predict_preapplicants_with_sarima(
                programme=row_tuple._1,       # Croho groepeernaam
                herkomst=row_tuple.Herkomst,
                examentype=row_tuple.Examentype,
                predict_year=row_tuple.Collegejaar,
                predict_week=row_tuple.Weeknummer,
                return_values="weighted only"
            )

        def impute_predicted_preapplicants(data, row_tuple, predicted_preapplicants):
            # Build mapping (full group_cols tuple -> value)
            prediction_weeks = get_prediction_weeks_list(row_tuple.Weeknummer)
            pred_map = pd.Series(
                data=predicted_preapplicants,
                index=pd.MultiIndex.from_tuples([
                    (
                        row_tuple.Collegejaar,
                        row_tuple._1, # Croho groepeernaam
                        row_tuple.Faculteit,
                        row_tuple.Examentype,
                        row_tuple.Herkomst,
                        week,
                    )
                    for week in prediction_weeks
                ])
            )

            # Apply mapping row-wise
            data = data.copy()
            for idx, row in data.iterrows():
                key = tuple(row[col] for col in GROUP_COLS + WEEK_COL)
                if key in pred_map.index:
                    data.at[idx, "Voorspelde gewogen vooraanmelders"] = pred_map[key]

            return data


        logger.info("Start parallel predicting...")

        # --- Predict student inflow --- 
        predictions = joblib.Parallel(n_jobs=nr_CPU_cores)(
            joblib.delayed(predict_students_row)(row)
            for chunk in chunks
            for row in chunk.itertuples(index=False)
        )

        prediction_df["SARIMA_cumulative"] = predictions

        # Map SARIMA predictions back into latest data
        sarima_map = prediction_df.set_index(GROUP_COLS + WEEK_COL)["SARIMA_cumulative"].to_dict()
        self.data_latest["SARIMA_cumulative"] = [
            sarima_map.get(tuple(row[col] for col in GROUP_COLS + WEEK_COL), row["SARIMA_cumulative"])
            for _, row in self.data_latest.iterrows()
        ]

        # --- Predict preapplicants ---
        preapplicants = joblib.Parallel(n_jobs=nr_CPU_cores)(
            joblib.delayed(predict_preapplicants_row)(row)
            for chunk in chunks
            for row in chunk.itertuples(index=False)
        )
        
        # Apply imputation
        for row, preapplicant in zip(prediction_df.itertuples(index=False), preapplicants):
            if preapplicant:  # skip empty 
                self.data_latest = impute_predicted_preapplicants(self.data_latest, row, preapplicant)



        # 5. Write the file
        output_path = self.configuration["paths"]['input']["path_latest"].replace("${root_path}", ROOT_PATH)
        #self.data_latest.to_excel(output_path, index=False)

        logger.info(f"Total file updated: {output_path}")



def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # Load data
    cumulative_data = load_cumulative()
    student_counts = load_student_numbers_first_years()
    latest_data = load_latest()

    # Initialize model
    cumulative_model = Cumulative(cumulative_data, student_counts, latest_data, configuration)

    for year in args.years:
        for week in args.weeks:
            cumulative_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                skip_years=args.skip_years
            )


if __name__ == "__main__":
    main()

    ''' 
    # Load configuration
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # Load data
    cumulative_data = load_cumulative()
    student_counts = load_student_numbers_first_years()
    latest_data = load_latest()

    # Initialize model
    cumulative_model = Cumulative(cumulative_data, student_counts, latest_data, configuration)

    cumulative_model.preprocess()

    cumulative_model.predict_students_with_preapplicants(
        programme="B Sociologie",
        herkomst="NL",
        examentype="Bachelor",
        predict_year=2024,
        predict_week=10,
    )
    '''
