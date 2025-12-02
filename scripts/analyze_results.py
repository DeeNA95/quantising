import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")  # Use non-interactive backend


def analyze_logic():
    print("Fetching runs from WandB...")
    try:
        api = wandb.Api()
        # Project path
        entity = "dadadee02-n-a"
        project = "mixed-precision-quant-2"
        runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"Error connecting to WandB: {e}")
        print("Ensure you are logged in: wandb login")
        return

    data = []
    print(f"Found {len(runs)} runs. Processing...")

    for run in runs:
        if run.state != "finished":
            continue

        # Extract config ID
        run_name = run.name
        config_id = "Unknown"
        if run_name.startswith("C-1"):
            config_id = "C-1"
        elif run_name.startswith("C-2"):
            config_id = "C-2"
        elif run_name.startswith("E-A"):
            config_id = "E-A"
        elif run_name.startswith("E-B"):
            config_id = "E-B"
        elif run_name.startswith("E-C"):
            config_id = "E-C"

        # Metrics
        vram = run.summary.get("vram_gb", 0)
        latency = run.summary.get("latency", 0)
        hellaswag = run.summary.get("hellaswag_acc", 0)
        mmlu = run.summary.get("mmlu_avg", 0)

        # Calculate Efficiency Score
        accuracy = hellaswag
        if mmlu > 0:
            accuracy = (hellaswag + mmlu) / 2

        efficiency_score = 0
        if vram > 0:
            efficiency_score = (accuracy / vram) * latency

        data.append(
            {
                "Config": config_id,
                "Name": run_name,
                "VRAM (GB)": round(vram, 2),
                "Latency (tok/s)": round(latency, 2),
                "HellaSwag": round(hellaswag, 4),
                "MMLU": round(mmlu, 4),
                "Avg Acc": round(accuracy, 4),
                "Efficiency Score": round(efficiency_score, 2),
            }
        )

    if not data:
        print("No finished runs found.")
        return

    df = pd.DataFrame(data)
    df.sort_values(by="Config", inplace=True)

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))
    print("=" * 80)

    print("\n🏆 WINNERS (excluding C-1 Baseline):")

    # Filter out C-1
    quantized_df = df[df["Config"] != "C-1"]

    if not quantized_df.empty:
        # Best VRAM
        best_vram = quantized_df.loc[quantized_df["VRAM (GB)"].idxmin()]
        print(f"📉 Lowest VRAM: {best_vram['Config']} ({best_vram['VRAM (GB)']} GB)")

        # Best Accuracy
        best_acc = quantized_df.loc[quantized_df["Avg Acc"].idxmax()]
        print(f"🎯 Best Accuracy: {best_acc['Config']} ({best_acc['Avg Acc']})")

        # Best Efficiency
        best_eff = quantized_df.loc[quantized_df["Efficiency Score"].idxmax()]
        print(
            f"⚡ Best Efficiency: {best_eff['Config']} (Score: {best_eff['Efficiency Score']})"
        )


    output_dir = "linkedin_visuals"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save table as image
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axis("tight")
    ax.axis("off")

    table_data = df[
        [
            "Config",
            "Name",
            "VRAM (GB)",
            "Latency (tok/s)",
            "HellaSwag",
            "MMLU",
            "Avg Acc",
            "Efficiency Score",
        ]
    ].values
    col_labels = [
        "Config",
        "Name",
        "VRAM (GB)",
        "Latency (tok/s)",
        "HellaSwag",
        "MMLU",
        "Avg Acc",
        "Efficiency Score",
    ]

    table = ax.table(
        cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)


    col_widths = [0.08, 0.24, 0.10, 0.12, 0.11, 0.09, 0.10, 0.14]
    for i, width in enumerate(col_widths):
        for j in range(len(table_data) + 1):
            table[(j, i)].set_width(width)

    # Style the header row
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E7E6E6")
            else:
                table[(i, j)].set_facecolor("#FFFFFF")

    plt.title(
        "Experiment Results: Mixed-Precision Quantization",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    table_path = os.path.join(output_dir, "results_table.png")
    plt.savefig(table_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    print(f"✅ Table saved to: {table_path}")

    metrics = ["VRAM (GB)", "Latency (tok/s)", "Avg Acc", "Efficiency Score"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))
        names = df["Name"].values
        values = df[metric].values

        bars = ax.bar(
            names, values, color=["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"]
        )

        y_min = values.min()
        y_max = values.max()
        y_range = y_max - y_min

        if y_range > 0:
            ax.set_ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}" if metric != "Avg Acc" else f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )

        # Styling
        ax.set_xlabel("Configuration", fontsize=12, weight="bold")
        ax.set_ylabel(metric, fontsize=12, weight="bold")
        ax.set_title(f"{metric} by Configuration", fontsize=14, weight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()

        # Save
        metric_filename = (
            metric.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        graph_path = os.path.join(output_dir, f"{metric_filename}_comparison.png")
        plt.savefig(graph_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close()
        print(f"✅ Bar graph saved to: {graph_path}")

    print(f"\n✨ All visualizations saved to '{output_dir}/' directory")


if __name__ == "__main__":
    analyze_logic()
