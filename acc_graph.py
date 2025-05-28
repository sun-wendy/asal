import matplotlib.pyplot as plt

# Stats
# Balanced accuracy here, not the same as accuracy
acc_stats = {
    "dt1": [1.0, 1.0],
    "dt4": [0.99997, 0.99994],
    "dt6": [0.71075],
    "dt8": [0.73616, 0.73685],
    "dt16": [0.71807, 0.71999],
    "dt32": [],
}

precision_stats = {
    "dt1": [1.0, 1.0],
    "dt4": [0.99975, 0.99965],
    "dt6": [0.48326],
    "dt8": [0.37183, 0.37266],
    "dt16": [0.34379, 0.32199],
    "dt32": [],
}

recall_stats = {
    "dt1": [1.0, 1.0],
    "dt4": [0.99995, 0.99991],
    "dt6": [0.45125],
    "dt8": [0.5238, 0.5258],
    "dt16": [0.47888, 0.48806],
    "dt32": [],
}

auroc_stats = {
    "dt1": [1.0, 1.0],
    "dt4": [1.0, 1.0],
    "dt6": [0.95351],
    "dt8": [0.94255, 0.94212],
    "dt16": [0.93534, 0.93308],
    "dt32": [],
}


def prepare_data(stats_dict):
    """
    Given a dict mapping 'dtX' to lists of values,
    returns sorted (dt_values, average_values) tuples,
    skipping empty lists.
    """
    dt_vals = []
    avg_vals = []
    for key, vals in stats_dict.items():
        if not vals:
            continue
        dt = int(key.lstrip("dt"))
        avg = sum(vals) / len(vals)
        dt_vals.append(dt)
        avg_vals.append(avg)
    dt_vals, avg_vals = zip(*sorted(zip(dt_vals, avg_vals)))
    return dt_vals, avg_vals


if __name__ == "__main__":
    metrics = {
        "Balanced Accuracy": acc_stats,
        "Precision": precision_stats,
        "Recall": recall_stats,
        "AUROC": auroc_stats,
    }

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (metric_name, stats) in zip(axes, metrics.items()):
        dt_vals, avg_vals = prepare_data(stats)
        ax.plot(dt_vals, avg_vals, marker='o', linestyle='-')
        ax.set_title(metric_name)
        ax.set_xlabel('dt')
        ax.set_ylabel(f'Average {metric_name}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("acc_graph.png")
