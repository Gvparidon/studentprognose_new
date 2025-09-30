# --- Standard library ---
import os
import sys
import math
import json
import logging
import warnings
from pathlib import Path
from typing import Tuple, List

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

# --- Path setup ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

# --- Project modules ---
from scripts.load_data import (
    load_cumulative,
    load_student_numbers_first_years,
    load_latest,
)
from scripts.helper import *
from scripts.transform_data import *
from cli import parse_args


# --- Warnings and logging setup ---
warnings.simplefilter("ignore", ConvergenceWarning)
logger = logging.getLogger(__name__)

# --- Environment setup ---
load_dotenv()

# --- Load configuration ---
ROOT_PATH = os.getenv("ROOT_PATH")


class Cumulative():
    def __init__(self, data_cumulative, data_studentcount, data_latest, configuration):
    
        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.data_latest = data_latest
        self.configuration = configuration
        self.skip_years = 0
        self.pred_len = None

    
    def preprocess(self) -> pd.DataFrame:
        """
        Cleans, filters, aggregates, and merges cumulative pre-application data.

        The method updates `self.data_cumulative` with the processed DataFrame and
        creates a backup in `self.data_cumulative_backup`.

        Returns:
            pd.DataFrame: The fully preprocessed and merged DataFrame.
        """
        # --- Configuration ---
        numeric_cols = [
            "Ongewogen vooraanmelders", "Gewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"
        ]
        rename_map = {
            "Type hoger onderwijs": "Examentype",
            "Groepeernaam Croho": "Croho groepeernaam",
        }
        group_cols = [
            "Collegejaar", "Croho groepeernaam", "Faculteit",
            "Examentype", "Herkomst", "Weeknummer"
        ]
        sort_cols = [
            "Collegejaar", "Croho groepeernaam", "Examentype",
            "Herkomst", "Weeknummer"
        ]
        merge_on_cols = ["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"]

        # --- Initial Processing ---
        df = self.data_cumulative.copy()

        # 1. Rename columns
        df.rename(columns=rename_map, inplace=True)

        # 2. Convert numeric columns to float64
        for col in numeric_cols:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = pd.to_numeric(
                    df[col]
                    .str.replace(".", "", regex=False)  # For thousand separators like '1.000'
                    .str.replace(",", ".", regex=False), # For decimal commas like '12,34'
                    errors='coerce'
                )

        df[numeric_cols] = df[numeric_cols].astype('float64')

        # 3. Filter for first-year and pre-master students
        mask = (df["Hogerejaars"] == "Nee") | (df["Examentype"] == "Pre-master")
        df = df[mask]

        # 4. Group and aggregate data
        processed_df = df.groupby(group_cols, as_index=False)[numeric_cols].sum()

        # 5. Merge with student count data (if it exists)
        if self.data_studentcount is not None:
            processed_df = processed_df.merge(
                self.data_studentcount,
                on=merge_on_cols,
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
        final_cols_order = group_cols + numeric_cols
        existing_cols = set(final_cols_order)
        # Add any new columns from the merge and the 'ts' column
        new_cols = [col for col in processed_df.columns if col not in existing_cols]
        final_cols_order.extend(new_cols)

        processed_df = (
            processed_df.sort_values(by=sort_cols, ignore_index=True)
            .drop_duplicates()
            [final_cols_order]  # Enforce consistent column order
        )

        # --- Update Instance Attributes ---
        self.data_cumulative_backup = self.data_cumulative.copy()
        self.data_cumulative = processed_df

        return self.data_cumulative


    def get_transformed_data(self, data, column="ts"):
        data = data.drop_duplicates()
        data = data[data["Collegejaar"] >= 2016]

        # Makes a certain pivot_wider where it transforms the data from long to wide
        data = transform_data(data, column)
        return data

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
                data = self.get_transformed_data(data, "Gewogen vooraanmelders")
            else:
                data = self.get_transformed_data(data)
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
                print(weighted_results.forecast(steps=pred_len).tolist())
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
        if self.data_studentcount is None:
            raise FileNotFoundError("Student count is required")

        # Predict pre-applicants with SARIMA
        predicted_preapplicants = self.predict_preapplicants_with_sarima(
            programme, herkomst, examentype, predict_year, predict_week, return_values="weighted + applications", refit=refit_sarima
        )

        # Transform data
        data = transform_data(self.data_cumulative, "ts")

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

        # Define preprocessing
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

        # Fit and predict
        model.fit(X_train, y_train)
        prediction = model.predict(test).round().astype(int)

        print(
            f"Prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {prediction[0]}"
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

        # Define columns to keep
        group_cols = [
            "Collegejaar",
            "Croho groepeernaam",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Weeknummer"
        ]

        # Apply mask
        prediction_df = self.data_cumulative.loc[mask, group_cols].copy()

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
            group_cols = [
                "Collegejaar",
                "Croho groepeernaam",
                "Faculteit",
                "Examentype",
                "Herkomst",
                "Weeknummer",
            ]
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
                key = tuple(row[col] for col in group_cols)
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
        sarima_map = prediction_df.set_index(group_cols)["SARIMA_cumulative"].to_dict()
        self.data_latest["SARIMA_cumulative"] = [
            sarima_map.get(tuple(row[col] for col in group_cols), 0)
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



    def _predict_with_xgboost_extra_year(self, train, test, data_to_predict, replace_mask):
        """
        Determines the right train and testdata to pass on to the XGBoost model.

        Args:
            train (pd.DataFrame): training data that still has to be filtered.
            test (pd.DataFrame): test data that still has to be filtered.
            data_to_predict (pd.DataFrame): Dataframe with in every row a different item that
            has to be predicted.
            replace_mask (pd.DataFrame): Dataframe with booleans indicating which rows in the
            data_to_predict are the rows we are going to predict.

        Returns:
            (pd.DataFrame): data_to_predict dataframe with the final XGBoost (SARIMA_cumulative)
            prediction added.
        """

        columns_to_match = [
            "Collegejaar",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Croho groepeernaam",
        ]

        # Predict for only the next academic year
        train = train[(train["Collegejaar"] < self.predict_year)]
        test = test[(test["Collegejaar"] == self.predict_year)]

        # Actual test data (test_merged) is obtained by filtering data_to_predict on the
        # 'Weeknummer' and the replace_mask, merged with the testdata based on 5 columns.
        test_merged = data_to_predict[
            (data_to_predict["Weeknummer"] == self.predict_week) & replace_mask
        ][columns_to_match].merge(test, on=columns_to_match)

        if not test_merged.empty:
            predictions = self._predict_with_xgboost(train, test_merged)

            # This mask indicates which items in data_to_predict are just predicted.
            mask = (
                data_to_predict[columns_to_match]
                .apply(tuple, axis=1)
                .isin(test_merged[columns_to_match].apply(tuple, axis=1))
            )

            # Apply the masks
            full_mask = replace_mask & (data_to_predict["Weeknummer"] == self.predict_week) & mask

            # Fill in the predictions in the dataframe
            data_to_predict.loc[full_mask, "SARIMA_cumulative"] = predictions[: full_mask.sum()]

        return data_to_predict


    def predict_nr_of_students_skipyear(self, year_pred, week_pred, data_to_predict):
        """
        Predicts the number of students for a skip year scenario.

        Args:
            year_pred (int): The year to be predicted.
            week_pred (int): The week to be predicted.
            data_to_predict (pd.DataFrame): DataFrame with pre-application data to predict.

        Returns:
            pd.DataFrame: DataFrame with predictions for the skip year scenario.
        """

        # Create skip year data
        data_cumulative_skip_years, data_student_numbers_skip_years = self.create_skip_year_data(
            year_pred, week_pred, data_to_predict
        )

        # Create a backup of the original data
        data_cumulative_backup_backup = self.data_cumulative_backup.copy()
        data_student_numbers_backup = self.data_studentcount.copy()

        # Update the dataholder with the skip year data
        self.data_cumulative_backup = data_cumulative_skip_years.copy()
        self.data_studentcount = data_student_numbers_skip_years.copy()

        # Predict for the next year
        data_to_predict_skip_year = self.predict_nr_of_students(year_pred + 1, 39, 1)

        # Restore the original data after prediction
        self.data_cumulative = data_cumulative_backup_backup.copy()
        self.data_studentcount = data_student_numbers_backup.copy()

        # Transform and merge the skip year predictions
        data_to_predict_skip_year = data_to_predict_skip_year[
            data_to_predict_skip_year["Weeknummer"] == 39
        ]
        data_to_predict_skip_year.rename(
            columns={"SARIMA_cumulative": "Skip_year_prediction_cumulative"}, inplace=True
        )
        columns_to_match = [
            "Collegejaar",
            "Croho groepeernaam",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Weeknummer",
        ]
        data_to_predict_skip_year = data_to_predict_skip_year[
            columns_to_match + ["Skip_year_prediction_cumulative"]
        ]
        data_to_predict_skip_year["Collegejaar"] = year_pred
        data_to_predict_skip_year["Weeknummer"] = week_pred

        data_to_predict = data_to_predict.merge(
            data_to_predict_skip_year,
            on=columns_to_match,
            how="left",
        )

        return data_to_predict

    def create_skip_year_data(self, year_pred, week_pred, data_to_predict):
        # 1. Load all three workbooks
        # data_to_predict = data_to_predict.copy(deep=True)
        data_cumulative = self.data_cumulative_original.copy(deep=True)
        data_student_numbers = self.data_studentcount.copy(deep=True)

        # 3. Find all unique study/examtype/herkomst combinations in the prediction set
        key_cols = ["Croho groepeernaam", "Examentype", "Herkomst", "Faculteit"]

        combos = data_to_predict[key_cols].drop_duplicates()

        # 5. Build list of “remaining weeks” for that same Collegejaar:
        #    weeks week_pred+1 … 52, then 1 … 38
        if week_pred == 38:
            weeks_after = []
        elif week_pred < 38:
            weeks_after = list(range(week_pred + 1, 39))
        else:
            weeks_after = list(range(week_pred + 1, 53)) + list(range(1, 39))

        # 4. Trim data_cumulative to only data up through that point
        data_cumulative_trimmed = data_cumulative.loc[
            ~(
                (data_cumulative["Collegejaar"] == year_pred)
                & (data_cumulative["Weeknummer"].isin(weeks_after))
            )
            & ~(data_cumulative["Collegejaar"] > year_pred),
            :,
        ].copy()

        # 6. For each combo & each remaining week, grab the predicted “Voorspelde vooraanmelders en inschrijvingen”
        #    from data_to_predict and append a row to data_cumulative_trimmed.
        #    The last 3 columns of data_cumulative get NaN, and we map:
        #      Gewogen vooraanmelders ← Voorspelde vooraanmelders en inschrijvingen
        #    (adjust names if your columns differ)
        last3 = data_cumulative_trimmed.columns[-3:].tolist()

        rows_to_append = []
        for _, r in combos.iterrows():
            studie, exam, herkomst, faculty = (
                r["Croho groepeernaam"],
                r["Examentype"],
                r["Herkomst"],
                r["Faculteit"],
            )
            for wk in weeks_after:
                # look up that week in data_to_predict
                m = (
                    (data_to_predict["Collegejaar"] == year_pred)
                    & (data_to_predict["Weeknummer"] == wk)
                    & (data_to_predict["Croho groepeernaam"] == studie)
                    & (data_to_predict["Examentype"] == exam)
                    & (data_to_predict["Herkomst"] == herkomst)
                )
                if m.any():
                    pred_val = data_to_predict.loc[
                        m, "Voorspelde vooraanmelders en inschrijvingen"
                    ].iloc[0]
                else:
                    pred_val = np.nan

                base = {
                    "Collegejaar": year_pred,
                    "Croho groepeernaam": studie,
                    "Faculteit": faculty,
                    "Examentype": exam,
                    "Herkomst": herkomst,
                    "Weeknummer": wk,
                    "Gewogen vooraanmelders": pred_val,
                }
                # fill last 3 columns with NaN
                for c in last3:
                    base[c] = np.nan

                rows_to_append.append(base)

        df_append = pd.DataFrame(rows_to_append)
        data_cumulative_updated = pd.concat(
            [data_cumulative_trimmed, df_append], ignore_index=True
        )

        # 7. Finally, append week 39 of Collegejaar+1, with last 4 columns = 0.
        #    (last four columns could be e.g. Gewogen…, plus your 3 others)
        last4 = data_cumulative_updated.columns[-4:].tolist()
        rows_final = []
        for _, r in combos.iterrows():
            studie, exam, herkomst, faculty = (
                r["Croho groepeernaam"],
                r["Examentype"],
                r["Herkomst"],
                r["Faculteit"],
            )

            rows_final.append(
                {
                    "Collegejaar": year_pred + 1,
                    "Croho groepeernaam": studie,
                    "Faculteit": faculty,
                    "Examentype": exam,
                    "Herkomst": herkomst,
                    "Weeknummer": 39,
                    **{c: 0 for c in last4},
                }
            )
        df_final = pd.DataFrame(rows_final)
        data_cumulative_final = pd.concat([data_cumulative_updated, df_final], ignore_index=True)

        # 8. Process data_student_numbers: drop Collegejaar >= year_pred
        data_student_numbers_trim = data_student_numbers.loc[
            data_student_numbers["Collegejaar"] < year_pred, :
        ].copy()

        # 9. For each combo, append a row for Collegejaar=year_pred with Aantal_studenten = SARIMA_cumulative
        rows_stu = []
        for _, r in combos.iterrows():
            studie, exam, herkomst = r["Croho groepeernaam"], r["Examentype"], r["Herkomst"]
            m = (
                (data_to_predict["Collegejaar"] == year_pred)
                & (data_to_predict["Croho groepeernaam"] == studie)
                & (data_to_predict["Examentype"] == exam)
                & (data_to_predict["Herkomst"] == herkomst)
            )
            if m.any():
                cum_val = data_to_predict.loc[m, "SARIMA_cumulative"].iloc[0]
            else:
                cum_val = np.nan

            rows_stu.append(
                {
                    "Collegejaar": year_pred,
                    "Croho groepeernaam": studie,
                    "Examentype": exam,
                    "Herkomst": herkomst,
                    "Aantal_studenten": cum_val,
                }
            )

        data_student_numbers_new = pd.DataFrame(rows_stu)
        data_student_numbers_final = pd.concat(
            [data_student_numbers_trim, data_student_numbers_new], ignore_index=True
        )

        return data_cumulative_final, data_student_numbers_final

        # 10. Save out your results
        # data_cumulative_final.to_excel('data_cumulative_appended.xlsx', index=False)
        # data_student_numbers_final.to_excel('data_student_numbers_appended.xlsx', index=False)

        # print("Done. Outputs saved to:")
        # print("  • data_cumulative_appended.xlsx")
        # print("  • data_student_numbers_appended.xlsx")


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
    
