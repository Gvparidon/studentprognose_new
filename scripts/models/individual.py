# individual.py

# --- Standard library ---
import os
import sys
import math
import json
import time
import logging
import warnings
from datetime import date
from pathlib import Path

# --- Third-party libraries ---
import numpy as np
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
from scripts.helper import get_all_weeks_valid, get_weeks_list, get_pred_len
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
        self.pred_len = None

        # Cached xgboost models
        self.xgboost_models = {}

        # Backup data
        self.data_individual_backup = self.data_individual.copy()

        # Store processing variables
        self.preprocessed = False
        self.predicted = False

    # --------------------------------------------------
    # -- General helper functions --
    # --------------------------------------------------
    
    def _get_transformed_data(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Drops duplicates, filters data from start_year onwards, and transforms from long to wide format.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column to pivot.

        Returns:
            pd.DataFrame: Transformed DataFrame with weeks as columns.
        """
        # Drop duplicates and filter years
        df = df[df["Collegejaar"] >= self.configuration["start_year"]]

        # Keep relevant columns
        df = df.loc[:, GROUP_COLS + [column, "Weeknummer"]]

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


    # --------------------------------------------------
    # -- Preprocessing --
    # --------------------------------------------------

    ### --- Helpers --- ###
    def to_weeknummer(self, datum):
        try:
            day, month, year = map(int, datum.split("-"))
            return date(year, month, day).isocalendar()[1]
        except (AttributeError, ValueError):
            return np.nan

    def get_herkomst(self, nat, eer):
        if nat == "Nederlandse":
            return "NL"
        elif eer == "J":
            return "EER"
        return "Niet-EER"

    def get_deadlineweek(self, row):
        return row["Weeknummer"] == 17 and (
            row["Croho groepeernaam"] not in list(self.configuration["numerus_fixus"].keys())
            or row["Examentype"] != "Bachelor"
        )

    # --- Main logic ---
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the input data for further analysis.
        """

        # --- Load and clean base dataset ---
        df = self.data_individual.copy()
        df = df.drop(columns=["Aantal studenten"])

        # --- Filter out specific English programme in 2021 ---
        mask = (
            (df["Croho groepeernaam"] == "B English Language and Culture")
            & (df["Collegejaar"] == 2021)
            & (df["Examentype"] != "Propedeuse Bachelor")
        )
        df = df[~mask]

        # --- Add count of entries per key ---
        df["Sleutel_count"] = df.groupby(["Collegejaar", "Sleutel"])["Sleutel"].transform(
            "count"
        )

        # --- Convert dates to week numbers ---
        df["Datum intrekking vooraanmelding"] = df["Datum intrekking vooraanmelding"].apply(
            self.to_weeknummer
        )
        df["Weeknummer"] = df["Datum Verzoek Inschr"].apply(self.to_weeknummer)

        # --- Derive origin from nationality and EER flag ---
        df["Herkomst"] = df.apply(lambda x: self.get_herkomst(x["Nationaliteit"], x["EER"]), axis=1)

        # --- Keep only entries with September or October intake ---
        df = df[
            df["Ingangsdatum"].str.contains("01-09-")
            | df["Ingangsdatum"].str.contains("01-10-")
        ]

        # --- Update RU faculty name ---
        df["Faculteit"] = df["Faculteit"].replace(self.configuration["faculty"])

        # --- Add numerus fixus flag ---
        df["is_numerus_fixus"] = (
            df["Croho groepeernaam"].isin(list(self.configuration["numerus_fixus"].keys()))
            & (df["Examentype"] == "Bachelor")
        ).astype(int)

        # --- Normalize exam type names ---
        df["Examentype"] = df["Examentype"].replace("Propedeuse Bachelor", "Bachelor")

        # --- Filter on valid enrollment status and exam types ---
        df = df[df["Inschrijfstatus"].notna()]
        df = df[df["Examentype"].isin(["Bachelor", "Master", "Pre-master"])]

        # --- Collapse rare nationalities into 'Overig' ---
        counts = df["Nationaliteit"].value_counts()
        rare_values = counts[counts < 100].index
        df["Nationaliteit"] = df["Nationaliteit"].replace(rare_values, "Overig")

        # --- Add distances if available ---
        if self.data_distances is not None:
            afstand_lookup = self.data_distances.set_index("Geverifieerd adres plaats")["Afstand"]
            df["Afstand"] = df["Geverifieerd adres plaats"].map(afstand_lookup)
        else:
            df["Afstand"] = np.nan

        # --- Determine deadline week flag ---
        df["Deadlineweek"] = df.apply(self.get_deadlineweek, axis=1)

        # --- Drop unneeded columns ---
        df = df.drop(columns=["Sleutel"])

        # --- Special handling for pre-master entries ---
        premaster_mask = df["Examentype"] == "Pre-master"
        df.loc[
            premaster_mask, ["Is eerstejaars croho opleiding", "Is hogerejaars", "BBC ontvangen"]
        ] = [1, 0, 0]

        # --- Final filtering on enrollment status ---
        df = df[
            (df["Is eerstejaars croho opleiding"] == 1)
            & (df["Is hogerejaars"] == 0)
            & (df["BBC ontvangen"] == 0)
        ]

        # --- Final cleanup ---
        df = df[GROUP_COLS + CATEGORICAL_COLS + NUMERIC_COLS + WEEK_COL + TARGET_COL + ["Datum intrekking vooraanmelding"]]

        # --- Store results ---
        self.data_individual_backup = self.data_individual.copy()
        self.data_individual = df
        self.preprocessed = True

        return df

    # --------------------------------------------------
    # -- Prediction of pre-applicant probabilities (chance that someone will enroll) --
    # --------------------------------------------------
    
    ### --- Main logic --- ###
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

        # --- Preprocessing + Model Pipeline ---
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", NUMERIC_COLS),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_COLS + GROUP_COLS + WEEK_COL),
            ],
            remainder="drop",
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=0))
        ])

        # --- Fit Pipeline ---
        pipeline.fit(X_train, y_train)

        # --- Predictions ---
        probabilities = pipeline.predict_proba(X_test)[:, 1]

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

    # --------------------------------------------------
    # -- Prediction of inflow (using SARIMA to extent the current inflow to week 38) --
    # --------------------------------------------------

    ### --- Helpers --- ###
    def _filter_data(self, data: pd.DataFrame, herkomst: str, predict_year: int, programme: str, examentype: str) -> pd.DataFrame:
        data = self._get_transformed_data(data, TARGET_COL[0])
        filtered = data[
            (data["Herkomst"] == herkomst)
            & (data["Collegejaar"] <= predict_year)
            & (data["Croho groepeernaam"] == programme)
            & (data["Examentype"] == examentype)
        ]
        return filtered

    def _create_exog_variables(self, df: pd.DataFrame):
        df = df.melt(
        id_vars=GROUP_COLS,
        value_vars=[w for w in get_all_weeks_valid(df.columns) if w in df.columns],
        var_name="Weeknummer",
        value_name=TARGET_COL[0],
        )


        # Deadline week
        def set_deadline(row):
            if row["Examentype"] == "Bachelor":
                if row["Weeknummer"] in ['16', '17'] and row["Croho groepeernaam"] not in list(self.configuration["numerus_fixus"].keys()):
                    return 1
                elif row["Weeknummer"] in ['1', '2'] and row["Croho groepeernaam"] in list(self.configuration["numerus_fixus"].keys()):
                    return 1
            return 0
        
        df["Deadline"] = df.apply(set_deadline, axis=1)

        return df

    def _create_time_series(self, data: pd.DataFrame, pred_len: int, target = TARGET_COL[0]) -> np.ndarray:
        data = data.pivot_table(
            index=GROUP_COLS,
            columns="Weeknummer",
            values=target,
            aggfunc="sum",
            fill_value=0
        ).reset_index()
        ts_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
        return ts_data[:-pred_len]


    def _fit_sarima(self, ts_data: np.ndarray, model_name: str, predict_year: int, refit: bool = False):
        model_path = os.path.join(self.configuration["other_paths"]["individual_sarima_models"].replace("${root_path}", ROOT_PATH), f"{model_name}.json")

        sarimax_args = dict(
            order=(1, 1, 1) if len(ts_data) < 52 else (1, 0, 1),
            seasonal_order=(0, 0, 0, 0) if len(ts_data) < 52 else (1, 1, 1, 52),
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


    ### --- Main logic --- ###
    def predict_inflow_with_sarima(self,
        programme: str,
        herkomst: str,
        examentype: str,
        predict_year: int,
        predict_week: int,
        refit: bool = False,
        verbose: bool = False
    ) -> list[float]:
        """
        Predicts the number of students using SARIMA.
        """

        # --- Check if preapplicant probabilities are predicted ---
        if not self.predicted:
            self.predict_preapplicant_probabilities(predict_year, predict_week)

        # --- Get the prediction length ---
        pred_len = get_pred_len(predict_week)

        # --- Filter data based on the parameters given ---
        data = self._filter_data(self.data_individual.copy(), herkomst, predict_year, programme, examentype)

        # --- Create time series data ---
        ts_data = self._create_time_series(data, pred_len)

        # --- Create exog variables ---
        #exog_train, exog_test = _create_exog_variables(data)

        # --- Shortcut for week 38 (no prediction needed) ---
        if predict_week == 38:
            return ts_data[-1] if len(ts_data) else np.nan

        # --- Fit SARIMA model ---  
        model_name = f"{programme}{herkomst}{examentype}"
        try:
            results = self._fit_sarima(ts_data, model_name, predict_year, refit)
            forecast = results.forecast(steps=pred_len).tolist()

            # --- Return prediction ---
            prediction = round(forecast[-1])
        except ValueError:
            prediction = 0

        if verbose:
            print(
                f"Individual prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {prediction}"
            )
 

        return prediction


    # --------------------------------------------------
    # -- Full prediction loop --
    # --------------------------------------------------

    ### --- Helpers --- ###
    def predict_students_row(self, row_tuple, verbose):
        return self.predict_inflow_with_sarima(
            programme=row_tuple._1,       # Croho groepeernaam
            herkomst=row_tuple.Herkomst,
            examentype=row_tuple.Examentype,
            predict_year=row_tuple.Collegejaar,
            predict_week=row_tuple.Weeknummer,
            verbose=verbose
        )

    ### --- Main logic --- ###
    def run_full_prediction_loop(self, predict_year: int, predict_week: int, write_file: bool, verbose: bool, args = None):
        """
        Run the full prediction loop for all years and weeks.
        """
        logger.info("Running individual prediction loop")

        # --- Preprocess if not done ---
        if not self.preprocessed:
            self.preprocess()

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
        prediction_df = self.data_latest.loc[mask, GROUP_COLS + WEEK_COL].copy()

        mask &= self.data_latest["Weeknummer"] == predict_week
        
        # --- Apply mask ---
        prediction_df = self.data_latest.loc[mask, GROUP_COLS + WEEK_COL].copy()

        # --- Make sure the rows are unique ---
        prediction_df = prediction_df.drop_duplicates()

        # --- Parallel prediction ---
        nr_CPU_cores = os.cpu_count() or 1
        chunk_size = math.ceil(len(prediction_df) / nr_CPU_cores)

        chunks = [
            prediction_df.iloc[i : i + chunk_size] for i in range(0, len(prediction_df), chunk_size)
        ]

        # --- Predict student inflow --- 
        predictions = joblib.Parallel(n_jobs=nr_CPU_cores)(
            joblib.delayed(self.predict_students_row)(row, verbose)
            for chunk in chunks
            for row in chunk.itertuples(index=False)
        )

        prediction_df["SARIMA_individual"] = predictions

        # --- Map SARIMA predictions back into latest data ---
        sarima_map = prediction_df.set_index(GROUP_COLS + WEEK_COL)["SARIMA_individual"].to_dict()
        self.data_latest["SARIMA_individual"] = [
            sarima_map.get(tuple(row[col] for col in GROUP_COLS + WEEK_COL), row["SARIMA_individual"])
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
                pred_col="SARIMA_individual",
                baseline_col="Prognose_ratio",
                configuration=self.configuration,
                args=args
            )

            evaluator.print_evaluation_summary(print_programmes=False)

        logger.info("Individual prediction done")



# --- Main function ---
def main():
    # --- Parse arguments ---
    args = parse_args()

    # --- Load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # --- Load data ---
    individual_data = load_individual()
    distances = load_distances()
    latest_data = load_latest()

    # --- Initialize model ---
    individual_model = Individual(individual_data, distances, latest_data, configuration)

    # --- Run prediction loop ---
    for year in args.years:
        for week in args.weeks:
            individual_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                verbose=args.verbose,
                args=args
            )



if __name__ == "__main__":

    main()

    '''
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # Load data
    individual_data = load_individual()
    distances = load_distances()
    latest_data = load_latest()

    # Initialize model
    individual_model = Individual(individual_data, distances, latest_data, configuration)

    individual_model.predict_inflow_with_sarima(programme="B Bedrijfskunde", herkomst="NL", examentype="Bachelor", predict_year=2024, predict_week=20)
    '''