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


class ModelEvaluator:
    def __init__(self, df, actual_col, pred_col, configuration, args, baseline_col=None, eps=1e-6, alpha=0.5):
        """
        Class to evaluate a model with optional baseline comparison.
        """
        self.df = df
        self.actual_col = actual_col
        self.pred_col = pred_col
        self.baseline_col = baseline_col
        self.configuration = configuration
        self.args = args
        self.eps = eps
        self.alpha = alpha
        self.stats = None

        self._prepare_data()
        self._compute_group_stats()

    # -------------------------
    # Data preparation
    # -------------------------
    def _prepare_data(self):
        """Filter and prepare the evaluation DataFrame."""
        predict_year = self.args.years[0]
        predict_week = self.args.weeks[0]

        evaluation_df = self.df.copy()
        evaluation_df = evaluation_df[evaluation_df["Collegejaar"] >= 2021]

        filtering = self.configuration["filtering"]
        mask = np.ones(len(evaluation_df), dtype=bool)

        if filtering.get("programme"):
            mask &= evaluation_df["Croho groepeernaam"].isin(filtering["programme"])
        if filtering.get("herkomst"):
            mask &= evaluation_df["Herkomst"].isin(filtering["herkomst"])
        if filtering.get("examentype"):
            mask &= evaluation_df["Examentype"].isin(filtering["examentype"])
        if predict_week != 999:
            mask &= evaluation_df["Weeknummer"] == predict_week

        # Filter out rows without prediction (or 0 when actual <= 5)
        mask &= (evaluation_df[self.pred_col] > 0) | (evaluation_df[self.actual_col] <= 5)
        columns = YEAR_COL + GROUP_COLS + [self.actual_col, self.pred_col]
        if self.baseline_col:
            columns.append(self.baseline_col)
        evaluation_df = evaluation_df.loc[mask, columns].copy()
        evaluation_df = evaluation_df.dropna(subset=[self.actual_col])

        # Compute errors
        evaluation_df['abs_error'] = (evaluation_df[self.pred_col] - evaluation_df[self.actual_col]).abs()
        evaluation_df['mape_component'] = evaluation_df['abs_error'] / (evaluation_df[self.actual_col] + self.eps)

        if self.baseline_col:
            evaluation_df['baseline_abs_error'] = (evaluation_df[self.baseline_col] - evaluation_df[self.actual_col]).abs()
            evaluation_df['baseline_mape_component'] = evaluation_df['baseline_abs_error'] / (evaluation_df[self.actual_col] + self.eps)

        self.evaluation_df = evaluation_df

    # -------------------------
    # Compute group-level stats
    # -------------------------
    def _compute_group_stats(self):
        """Aggregate metrics per (programme, origin) group and compute all weighted components."""
        stats = self.evaluation_df.groupby(GROUP_COLS).agg(
            mae=('abs_error', 'mean'),
            mape=('mape_component', 'mean'),
            mean_actual=(self.actual_col, 'mean'),
            std_actual=(self.actual_col, lambda x: np.std(x, ddof=0)),
            sum_actual=(self.actual_col, 'sum')
        ).reset_index()

        # Scaled MAPE per group
        stats["scaled_mape"] = stats["mae"] / (stats["mean_actual"] + self.eps)
        # Coefficient of variation (relative volatility)
        stats["cv"] = stats["std_actual"] / (stats["mean_actual"] + self.eps)
        # Smoothed CV weight
        stats["weight"] = stats["cv"] ** self.alpha

        # Volatility-weighted MAPE component
        total_volatility = stats["weight"].sum()
        if total_volatility == 0:
            stats['vws_mape_component'] = stats['scaled_mape'] / len(stats)
        else:
            stats['vws_mape_component'] = stats['scaled_mape'] * (stats["weight"] / total_volatility)

        # Volume-weighted MAPE component
        total_volume = stats["sum_actual"].sum()
        if total_volume == 0:
            stats['volume_mape_component'] = stats['scaled_mape'] / len(stats)
        else:
            stats['volume_mape_component'] = stats['scaled_mape'] * (stats["sum_actual"] / total_volume)

        # Baseline metrics
        if self.baseline_col:
            baseline_stats = self.evaluation_df.groupby(GROUP_COLS).agg(
                baseline_mae=('baseline_abs_error', 'mean'),
                baseline_mape=('baseline_mape_component', 'mean')
            ).reset_index()
            baseline_stats['baseline_scaled_mape'] = baseline_stats['baseline_mae'] / (stats['mean_actual'] + self.eps)

            stats = pd.merge(stats, baseline_stats, on=GROUP_COLS, how='left')

            # Baseline weighted components
            if total_volatility == 0:
                stats['vws_mape_component_baseline'] = stats['baseline_scaled_mape'] / len(stats)
            else:
                stats['vws_mape_component_baseline'] = stats['baseline_scaled_mape'] * (stats["weight"] / total_volatility)

            if total_volume == 0:
                stats['volume_mape_component_baseline'] = stats['baseline_scaled_mape'] / len(stats)
            else:
                stats['volume_mape_component_baseline'] = stats['baseline_scaled_mape'] * (stats["sum_actual"] / total_volume)

        self.stats = stats

    # -------------------------
    # Metric computations
    # -------------------------
    def compute_mae(self, baseline=False):
        if baseline and self.baseline_col:
            return self.evaluation_df['baseline_abs_error'].mean()
        return self.evaluation_df['abs_error'].mean()

    def compute_mape(self, baseline=False):
        if baseline and self.baseline_col:
            return self.evaluation_df['baseline_mape_component'].mean()
        return self.evaluation_df['mape_component'].mean()

    def compute_unweighted_mape(self, baseline=False):
        col = 'baseline_scaled_mape' if baseline else 'scaled_mape'
        return self.stats[col].mean()

    def compute_volatility_weighted_mape(self, baseline=False):
        col = 'vws_mape_component_baseline' if baseline else 'vws_mape_component'
        return self.stats[col].sum()

    def compute_volume_weighted_mape(self, baseline=False):
        col = 'volume_mape_component_baseline' if baseline else 'volume_mape_component'
        return self.stats[col].sum()

    # -------------------------
    # Print evaluation summary
    # -------------------------
    def detailed_baseline_comparison(self):
        """
        Returns a DataFrame showing, per group, whether the model outperformed the baseline
        along with actual, model prediction, and baseline prediction.
        """
        if not self.baseline_col:
            raise ValueError("Baseline column not provided.")

        # Create a comparison DataFrame
        comparison_df = self.evaluation_df.copy()
        comparison_df['model_better'] = (
            comparison_df['abs_error'] < comparison_df['baseline_abs_error']
        )

        # Select relevant columns
        display_cols = GROUP_COLS + [
            self.actual_col,
            self.pred_col,
            self.baseline_col,
            'abs_error',
            'baseline_abs_error',
            'model_better'
        ]
        comparison_df = comparison_df[display_cols]

        # Sort by model_better and group size
        comparison_df = comparison_df.sort_values(
            by=['model_better', self.actual_col],
            ascending=[False, False]
        )

        # Split into wins and losses
        wins = comparison_df[comparison_df['model_better']]
        losses = comparison_df[~comparison_df['model_better']]

        print(f"\nModel won in {len(wins)} rows, lost in {len(losses)} rows\n")

        return wins.reset_index(drop=True), losses.reset_index(drop=True)




    def print_evaluation_summary(self):
        """Print all metrics and comparison against baseline."""
        def format_metric(metric_func):
            if self.baseline_col:
                return f"{metric_func():.4f} ({metric_func(baseline=True):.4f})"
            else:
                return f"{metric_func():.4f}"

        print("\n------- Model Evaluation -------")
        print(f"MAE:                              {format_metric(self.compute_mae)}")
        print(f"MAPE:                             {format_metric(self.compute_mape)}")
        print(f"Unweighted Scaled MAPE:           {format_metric(self.compute_unweighted_mape)}")
        print(f"Volatility-Weighted Scaled MAPE:  {format_metric(self.compute_volatility_weighted_mape)}")
        print(f"Volume-Weighted Scaled MAPE:      {format_metric(self.compute_volume_weighted_mape)}")
        print("-------------------------------\n")

        # Compare model vs baseline per group
        if self.baseline_col:
            metrics_to_compare = [
                ('MAE', 'mae', 'baseline_mae'),
                ('MAPE', 'mape', 'baseline_mape'),
                ('Scaled MAPE', 'scaled_mape', 'baseline_scaled_mape'),
                ('Volatility-Weighted Scaled MAPE', 'vws_mape_component', 'vws_mape_component_baseline'),
                ('Volume-Weighted Scaled MAPE', 'volume_mape_component', 'volume_mape_component_baseline')
            ]

            print("Model vs Baseline per Group Comparison:")
            for name, model_col, baseline_col in metrics_to_compare:
                if model_col in self.stats and baseline_col in self.stats:
                    better_count = (self.stats[model_col] < self.stats[baseline_col]).sum()
                    total_groups = len(self.stats)
                    print(f"{name}: Model outperformed baseline in {better_count}/{total_groups} groups "
                          f"({better_count / total_groups:.1%})")

        wins, losses = self.detailed_baseline_comparison()

        
        # Inspect first few wins
        print("Top 10 wins:")
        print(wins.head(10))

        # Inspect first few losses
        print("Top 10 losses:")
        print(losses.head(10))



# -------------------------
# Main execution
# -------------------------
def main():
    args = parse_args()

    # Load configuration
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)

    # Load data
    latest_data = load_latest()

    # Create evaluator
    evaluator = ModelEvaluator(
        latest_data,
        actual_col="Aantal_studenten",
        pred_col="SARIMA_cumulative",
        baseline_col="Prognose_ratio",
        configuration=configuration,
        args=args
    )

    # Print summary
    evaluator.print_evaluation_summary()


if __name__ == "__main__":
    main()
