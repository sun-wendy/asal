import matplotlib.pyplot as plt

# Stats
# Balanced accuracy here, not the same as accuracy
mse_stats = {
    "dt1": [0.003523],
    "dt2": [0.003403],
    "dt4": [0.003513],
    "dt5": [0.003559],
    "dt6": [0.003400],
    "dt8": [0.003533],
    "dt16": [0.003479],
}

r2_stats = {
    "dt1": [0.027299],
    "dt2": [0.026465],
    "dt4": [0.025700],
    "dt5": [0.024685],
    "dt6": [0.027853],
    "dt8": [0.023196],
    "dt16": [0.022535],
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
        "MSE": mse_stats,
        "R^2": r2_stats,
    }

    # Create 2x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes = axes.flatten()

    for ax, (metric_name, stats) in zip(axes, metrics.items()):
        dt_vals, avg_vals = prepare_data(stats)
        ax.plot(dt_vals, avg_vals, marker='o', linestyle='-')
        ax.set_title(metric_name)
        ax.set_xlabel('dt')
        ax.set_ylabel(f'{metric_name}')
        ax.grid(True, alpha=0.3)

    fig.suptitle("Linear Probe for Shannon Entropy 16 Steps Ahead", fontsize=16)
    plt.tight_layout()
    plt.savefig("probe_graph.png")
