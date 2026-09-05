"""Training plots shared by both entry points."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_output_graphs(
    output_dir, num_episodes, num_players, bot_types, size, output_lists
):
    sizes = {0: "Small", 1: "Medium", 2: "Large", "classic": "Classic"}
    bot_types = [bot or "Player" for bot in bot_types]
    bots = ", ".join(sorted(set(bot_types)))
    output_path = Path(f"{output_dir}/outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    names = [
        "Average Reward",
        "Cumulative Reward",
        "Loss",
        "Illegal Move Ratio",
        "Number of Actions",
        "Number of Turns",
        "Episode Time",
        "Action Selection STD",
        "Skip Action Ratio",
        "Percentage Map Owned on End",
    ]

    # Determine the layout of the subplots
    num_graphs = len(output_lists)
    num_cols = 5
    num_rows = (
        num_graphs + num_cols - 1
    ) // num_cols  # Ensure enough rows for all graphs
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(36, 6 * num_rows)
    )  # Adjust size accordingly
    axes = axes.flatten()  # Flatten the 2D array of axes to easily iterate over it

    for ax, output, name in zip(axes, output_lists, names):
        # Calculate the rolling average with a window of 30
        data_series = pd.Series(output)
        rolling_avg = data_series.rolling(window=30).mean()

        # Plot the original data and rolling average on each subplot
        ax.plot(range(1, num_episodes + 1), output, label="Original")
        ax.plot(
            range(1, num_episodes + 1),
            rolling_avg,
            label="Rolling Average",
            color="orange",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel(name)
        ax.set_title(f"{name} - {sizes[size]} Board; {num_players} Players; {bots}")
        ax.legend()

    # Turn off any extra empty subplots
    for i in range(num_graphs, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    # Save the entire figure with all subplots
    plt.savefig(
        output_path
        / f"combined_{sizes[size].lower()}_{num_players}_{'_'.join(sorted(set(bot_types))).lower()}.png"
    )
    plt.close(fig)

    # in addition to the large graph which is easier viewing for monitoring,
    # create all the smaller graphs which can be better for reporting
    def create_graph(output, name):
        data_series = pd.Series(output)
        rolling_avg = data_series.rolling(window=30).mean()

        # plt.plot(range(1, num_episodes+1), output)
        plt.plot(range(1, num_episodes + 1), output, label="Original")
        plt.plot(
            range(1, num_episodes + 1),
            rolling_avg,
            label="Rolling Average",
            color="orange",
        )
        plt.xlabel("Episode")
        plt.ylabel(name)
        plt.title(f"{name} - {sizes[size]} Board; {num_players} Players; {bots} Bots")
        plt.savefig(
            output_path
            / f"{sizes[size].lower()}_{num_players}_{'_'.join(sorted(set(bot_types))).lower()}_{name.lower().replace(' ', '_')}.png"
        )
        plt.close()

    for out, name in zip(output_lists, names):
        create_graph(out, name)
