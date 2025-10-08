import argparse
import datetime

def parse_args():
    """
    Parses command-line arguments using argparse for a more robust and readable CLI.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Student Prognosis Model CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Argument Definitions ---
    parser.add_argument(
        '-w', '--weeks',
        nargs='+',
        type=int,
        help='One or more week numbers to process (e.g., 5 6 7).'
    )
    parser.add_argument(
        '-y', '--years',
        nargs='+',
        type=int,
        help='One or more academic years to process (e.g., 2023 2024).'
    )
    parser.add_argument(
        '-d', '--data-option',
        type=str,
        default='both',
        choices=['individual', 'cumulative', 'both'],
        help='The dataset option to use for the prediction.'
    )
    parser.add_argument(
        '-wf', '--write-file',
        action='store_true',
        help='Write predictions to the total file.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output.'
    )
    parser.add_argument(
        '-ev', '--evaluate',
        action='store_true',
        help='Evaluate the predictions.'
    )

    parser.add_argument(
        '-rf', '--refit',
        action='store_true',
        help='Refit the models.'
    )

    args = parser.parse_args()

    # --- Post-processing and Default Value Logic ---

    # Set default year if not provided
    if not args.years:
        # If weeks are also not specified, and it's after week 39,
        # default to predicting for the *next* academic year.
        if not args.weeks and datetime.date.today().isocalendar()[1] >= 39:
            args.years = [datetime.date.today().year + 1]
        else:
            args.years = [datetime.date.today().year]


    # Set default week if not provided
    if not args.weeks:
        current_week = datetime.date.today().isocalendar()[1]
        # Week 53 is an edge case, default to 52.
        args.weeks = [min(current_week, 52)]


    return args