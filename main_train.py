import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from ninflam.cli import load_config
from ninflam.pipeline.train import run_training_pipeline


if __name__ == "__main__":
    cfg = load_config()
    run_training_pipeline(cfg)

    # Print summary metrics table at the bottom
    import json
    import pandas as pd
    from ninflam.pipeline.train import _resolve_paths

    try:
        paths = _resolve_paths(cfg)
        metrics_dir = Path(paths["metrics_dir"])
        cv_dir = Path(paths["cv_dir"])

        results_list = []

        for model_name in cfg.training.models:
            metric_file = metrics_dir / f"{model_name}.json"
            cv_file = cv_dir / f"{model_name}_cv_summary.csv"

            row = {
                "Model": model_name,
                "cv_acc_mean": 0.0,
                "cv_acc_std": 0.0,
                "cv_bal_acc": 0.0,
                "cv_prec": 0.0,
                "cv_rec": 0.0,
                "cv_f1": 0.0,
                "test_acc": 0.0,
                "test_bal_acc": 0.0,
                "test_prec": 0.0,
                "test_rec": 0.0,
                "test_f1": 0.0
            }

            if cv_file.exists():
                try:
                    df_cv = pd.read_csv(cv_file)
                    row["cv_acc_mean"] = df_cv["accuracy_mean"].iloc[0] * 100
                    row["cv_acc_std"] = df_cv["accuracy_std"].iloc[0] * 100
                    row["cv_bal_acc"] = df_cv.get("balanced_accuracy_mean", pd.Series([0.0])).iloc[0] * 100
                    row["cv_prec"] = df_cv.get("precision_macro_mean", pd.Series([0.0])).iloc[0] * 100
                    row["cv_rec"] = df_cv.get("recall_macro_mean", pd.Series([0.0])).iloc[0] * 100
                    row["cv_f1"] = df_cv["f1_macro_mean"].iloc[0] * 100
                except Exception:
                    pass

            if metric_file.exists():
                try:
                    with open(metric_file, "r") as f:
                        m_data = json.load(f)
                    ext = m_data.get("external", {})
                    row["test_acc"] = ext.get("accuracy", 0) * 100
                    row["test_bal_acc"] = ext.get("balanced_accuracy", 0) * 100
                    row["test_prec"] = ext.get("precision_macro", 0) * 100
                    row["test_rec"] = ext.get("recall_macro", 0) * 100
                    row["test_f1"] = ext.get("f1_macro", 0) * 100
                except Exception:
                    pass

            results_list.append(row)

        print("\n" + "=" * 132)
        print("                                     ALL MODEL EVALUATION & BENCHMARK METRICS")
        print("=" * 132)
        print(f"{'Model Name':<20} | {'10-Fold CV Acc':<18} | {'CV Bal Acc':<12} | {'CV Prec':<10} | {'CV Rec':<10} | {'CV F1':<10} | {'Test Acc':<10} | {'Test F1':<10}")
        print("-" * 132)

        best_cv_acc_model = max(results_list, key=lambda x: x["cv_acc_mean"])
        best_cv_f1_model = max(results_list, key=lambda x: x["cv_f1"])
        best_test_acc_model = max(results_list, key=lambda x: x["test_acc"])

        for r in results_list:
            tag = ""
            if r["Model"] == best_cv_acc_model["Model"]:
                tag = " (Best CV Acc)"
            elif r["Model"] == best_cv_f1_model["Model"]:
                tag = " (Best CV F1)"

            cv_acc_fmt = f"{r['cv_acc_mean']:.2f}% ± {r['cv_acc_std']:.2f}%"
            print(f"{r['Model'] + tag:<20} | {cv_acc_fmt:<18} | {r['cv_bal_acc']:>10.2f}% | {r['cv_prec']:>8.2f}% | {r['cv_rec']:>8.2f}% | {r['cv_f1']:>8.2f}% | {r['test_acc']:>8.2f}% | {r['test_f1']:>8.2f}%")

        print("=" * 132)
        print(f"👉 BEST OVERALL MODEL (by 10-Fold CV Accuracy): {best_cv_acc_model['Model']} ({best_cv_acc_model['cv_acc_mean']:.2f}% ± {best_cv_acc_model['cv_acc_std']:.2f}%)")
        print(f"👉 BEST MODEL (by 10-Fold CV Macro F1):        {best_cv_f1_model['Model']} ({best_cv_f1_model['cv_f1']:.2f}%)")
        print(f"👉 BEST MODEL (by Test Split Accuracy):         {best_test_acc_model['Model']} ({best_test_acc_model['test_acc']:.2f}%)\n")

    except Exception as e:
        print(f"\n[INFO] Training complete. (Could not format console table: {e})")


