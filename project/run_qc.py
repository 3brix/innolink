import pandas as pd

from config.datasets import cfg
from config.paths import RAW_DATA_DIR, QC_DIR, ensure_directory
from analysis.distributions.qc import check_basic_integrity, metric_description, missing_metrics_by_sample, nonfinite_report


input_path = RAW_DATA_DIR / cfg.name / "merged.csv"
df = pd.read_csv(input_path)

report = check_basic_integrity(df)
qc_df = pd.DataFrame(report.items(), columns=["check", "value"])

output_dir = QC_DIR / cfg.name
ensure_directory(output_dir)

output_path = output_dir / "qc_report.csv"
qc_df.to_csv(output_path, index=False)

# Per-metric summary statistics (describe() + missing) — a QC reference
# artifact, previously produced by run_distributions.py.
summary_path = output_dir / "metric_summary.csv"
metric_description(df).to_csv(summary_path, index=False)

# Per-sample missingness: which sample is missing which metrics. Complements
# the per-metric ``missing_metrics`` totals in qc_report.csv.
missing_by_sample_path = output_dir / "missing_by_sample.csv"
missing_metrics_by_sample(df).to_csv(missing_by_sample_path, index=False)

# Per-metric non-finite breakdown (NaN / +inf / -inf). Infinities are invalid
# and otherwise invisible in the missing counts.
nonfinite_path = output_dir / "nonfinite_report.csv"
nonfinite_report(df).to_csv(nonfinite_path, index=False)

print(f"QC report saved: {output_path}")
print(f"Metric summary saved: {summary_path}")
print(f"Missing-by-sample saved: {missing_by_sample_path}")
print(f"Non-finite report saved: {nonfinite_path}")