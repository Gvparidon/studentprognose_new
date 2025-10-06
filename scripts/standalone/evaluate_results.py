# evaluate_results.py

# --- imports ---
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.load_data import load_latest
from cli import parse_args

# --- Constant variable names ---
GROUP_COLS = [
    "Croho groepeernaam", "Faculteit",
    "Examentype", "Herkomst", "Weeknummer"
]

YEAR_COL = ['Collegejaar']

# --- Helpers ---
def _prepare_evaluation_data(df, actual_col, pred_col, configuration, args):
    """
    Prepares the data for evaluation.
    """

     # --- Get predict year and week ---
    predict_year = args.years[0]
    predict_week = args.weeks[0]

    # --- Create the evaluation df ---
    evaluation_df = df.copy()

    # --- Apply filtering from configuration ---
    filtering = configuration["filtering"]

    # --- Filter data ---
    mask = np.ones(len(evaluation_df), dtype=bool) 

    # --- Apply conditional filters from configuration ---
    if filtering["programme"]:
        mask &= evaluation_df["Croho groepeernaam"].isin(filtering["programme"])
    if filtering["herkomst"]:
        mask &= evaluation_df["Herkomst"].isin(filtering["herkomst"])
    if filtering["examentype"]:
        mask &= evaluation_df["Examentype"].isin(filtering["examentype"])

    # --- Apply week filter (if applicable) ---
    if not predict_week == 999:
        mask &= evaluation_df["Weeknummer"] == predict_week
    
    # --- Filter out rows without prediction (or 0 when actual is > 5) ---
    mask &= (evaluation_df[pred_col] > 0) | (evaluation_df[actual_col] <= 5)

    # --- Apply mask ---
    evaluation_df = evaluation_df.loc[mask, YEAR_COL + GROUP_COLS + [actual_col, pred_col]].copy()

    # --- Filter out rows where target is NA ---
    evaluation_df = evaluation_df.dropna(subset=[actual_col])

    return evaluation_df


# --- Evaluation functions ---
def volatility_weighted_scaled_mae(df, actual_col, pred_col, configuration, args, eps=1e-6):
    """
    Compute the Volatility-Weighted Scaled MAE (VWS-MAE)
    to evaluate inflow prediction models across multiple programme-origin pairs.

    Returns
    -------
    float
        The overall Volatility-Weighted Scaled MAE (VWS-MAE).
    """

    # --- Prepare evaluation data ---
    evaluation_df = _prepare_evaluation_data(df, actual_col, pred_col, configuration, args)

    # Add an absolute error column for efficient aggregation
    evaluation_df['abs_error'] = (evaluation_df[pred_col] - evaluation_df[actual_col]).abs()

    # --- Compute per-(programme, origin) stats using the more efficient .agg ---
    stats = evaluation_df.groupby(GROUP_COLS).agg(
        mae=('abs_error', 'mean'),
        mean_actual=(actual_col, 'mean'),
        std_actual=(actual_col, lambda x: np.std(x, ddof=0))
    ).reset_index()

    # Compute scaled MAE per group
    stats["scaled_mae"] = stats["mae"] / (stats["mean_actual"] + eps)

    # Compute volatility weights (using std as volatility measure)
    stats["weight"] = stats["std_actual"]
    total_volatility = stats["weight"].sum()

    # Handle case where all stds are zero (fully stable inflow)
    if total_volatility == 0:
        vws_mae = stats["scaled_mae"].mean()
    else:
        stats["weight_norm"] = stats["weight"] / total_volatility

        print(stats)

        # Weighted average of scaled MAE
        vws_mae = np.sum(stats["weight_norm"] * stats["scaled_mae"])

    return vws_mae



# --- main ---
def main():

    # --- Parse arguments ---
    args = parse_args()
    
    # --- Load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # --- Load data ---
    latest_data = load_latest()

    # --- Evaluate results ---
    vws_mae_result = volatility_weighted_scaled_mae(
        latest_data, "Aantal_studenten", "SARIMA_cumulative", configuration, args
    )

    # The main function is responsible for output
    print(f"Volatility-Weighted Scaled MAE: {vws_mae_result:.4f}")

        

if __name__ == "__main__":
    main()

