# main.py

# --- Standard library ---
import sys
import logging
from pathlib import Path
import time

# --- Third-party libraries ---
import yaml
from dotenv import load_dotenv

# --- Warnings and logging setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Project modules ---
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.load_data import load_data
from scripts.models.individual import Individual
from scripts.models.cumulative import Cumulative
from scripts.models.ratio import Ratio

from cli import parse_args


# --- Main pipeline ---
def pipeline(configuration, args):
    """
    Main pipeline for running the prediction loop for all models.
    """
    start_time = time.time()

    # --- load data ---
    data = load_data()

    cumulative_data = data["cumulative"]
    individual_data = data["individual"]
    distances = data["distances"]
    latest_data = data["latest"]
    lookup_higher_years = data["lookup_higher_years"]
    weighted_ensemble = data["weighted_ensemble"]
    student_counts = data["student_numbers_first_years"]
    logger.info("Data loaded.")

    # --- Initialize models ---
    cumulative_model = Cumulative(cumulative_data, student_counts, latest_data, configuration)
    individual_model = Individual(individual_data, distances, latest_data, configuration)
    ratio_model = Ratio(cumulative_data, student_counts, latest_data, configuration)
    logger.info("Models initialized.")

    # --- Run prediction loop for all models---
    for year in args.years:
        for week in args.weeks:
            logger.info(f"Running prediction loop for year: {year}, week: {week}")

            # --- Run cumulative prediction loop ---
            cumulative_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file
            )
            individual_model.data_latest = cumulative_model.data_latest.copy()
            ratio_model.data_latest = cumulative_model.data_latest.copy()

            # --- Run individual prediction loop ---
            individual_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file
            )
            ratio_model.data_latest = individual_model.data_latest.copy()

            # --- Run ratio prediction loop ---
            ratio_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file
            )
            
            latest_data = ratio_model.data_latest.copy()
            
    logger.info("Prediction loop completed.")
    

    # --- Write the file ---
    output_path = configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
    latest_data.to_excel(output_path, index=False, engine="xlsxwriter")

    logger.info(f"Output written to: {output_path}")
    
    end_time = time.time()
    logger.info(f"Total time: {(end_time - start_time) / 60:.2f} minutes")

def main():
    # --- Parse arguments ---
    args = parse_args()

    # --- Load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # --- Run pipeline ---
    pipeline(configuration, args)


if __name__ == "__main__":
    main()