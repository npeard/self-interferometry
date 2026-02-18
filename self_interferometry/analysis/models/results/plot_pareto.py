#!/usr/bin/env python3
"""Plot Pareto Frontier - Visualize trade-offs between model size and performance.

This script reads test results from test_results.json and plots the Pareto frontier
showing the relationship between model parameters and loss metrics (displacement and velocity).
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: Path) -> list[dict[str, Any]]:
    """Load and flatten test results from JSON file.

    Args:
        results_path: Path to test_results.json

    Returns:
        List of result dictionaries
    """
    with open(results_path) as f:
        data = json.load(f)

    # Flatten the nested structure (experiment_id -> list of results)
    results = []
    for experiment_id, experiment_results in data.items():
        for result in experiment_results:
            results.append(result)

    return results


def sort_by_params(
    params: np.ndarray, loss: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sort data points by parameter count.

    Args:
        params: Array of parameter counts
        loss: Array of loss values

    Returns:
        Tuple of (params, loss) arrays sorted by increasing params
    """
    # Sort by params
    sorted_indices = np.argsort(params)
    params_sorted = params[sorted_indices]
    loss_sorted = loss[sorted_indices]

    return params_sorted, loss_sorted


def plot_pareto_frontiers(
    results: list[dict[str, Any]], output_path: Path | None = None
) -> None:
    """Create Pareto frontier plots for displacement and velocity loss vs parameters.

    Args:
        results: List of result dictionaries from test_results.json
        output_path: Optional path to save the figure
    """
    # Group results by model_type and dataset_file
    groups = {}
    for result in results:
        model_type = result['model_type']
        dataset_file = result['dataset_file']
        key = (model_type, dataset_file)

        if key not in groups:
            groups[key] = {'params': [], 'displacement_loss': [], 'velocity_loss': []}

        groups[key]['params'].append(result['total_params'])
        groups[key]['displacement_loss'].append(result['displacement_loss'])
        groups[key]['velocity_loss'].append(result['velocity_loss'])

    # Convert lists to numpy arrays
    for key in groups:
        groups[key]['params'] = np.array(groups[key]['params'])
        groups[key]['displacement_loss'] = np.array(groups[key]['displacement_loss'])
        groups[key]['velocity_loss'] = np.array(groups[key]['velocity_loss'])

    # Define colors for each model type
    model_types = sorted(set(model_type for model_type, _ in groups.keys()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))
    model_color_map = {model_type: colors[i] for i, model_type in enumerate(model_types)}

    # Define line styles for each dataset
    dataset_style_map = {
        'free-space-synchro_10k.h5': '-',  # Solid line
        'mmfiber-synchro_10k.h5': '--',  # Dashed line
    }

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Track legend entries to avoid duplicates
    legend_handles = []
    legend_labels = []
    added_combinations = set()

    # Plot displacement loss
    for (model_type, dataset_file), data in groups.items():
        params_sorted, loss_sorted = sort_by_params(
            data['params'], data['displacement_loss']
        )

        color = model_color_map[model_type]
        linestyle = dataset_style_map.get(dataset_file, '-')

        # Plot scatter points
        ax1.scatter(
            data['params'], data['displacement_loss'], color=color, alpha=0.3, s=50
        )

        # Plot connecting line
        line = ax1.plot(
            params_sorted, loss_sorted, color=color, linestyle=linestyle, linewidth=2
        )[0]

        # Add to legend only once per combination
        combo_key = (model_type, dataset_file)
        if combo_key not in added_combinations:
            dataset_name = dataset_file.replace('-synchro_10k.h5', '').replace('-', ' ')
            label = f'{model_type} - {dataset_name}'
            legend_handles.append(line)
            legend_labels.append(label)
            added_combinations.add(combo_key)

    ax1.set_xlabel('Total Parameters', fontsize=12)
    ax1.set_ylabel('Displacement Loss', fontsize=12)
    ax1.set_title('Displacement Loss vs Model Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot velocity loss
    for (model_type, dataset_file), data in groups.items():
        params_sorted, loss_sorted = sort_by_params(
            data['params'], data['velocity_loss']
        )

        color = model_color_map[model_type]
        linestyle = dataset_style_map.get(dataset_file, '-')

        # Plot scatter points
        ax2.scatter(data['params'], data['velocity_loss'], color=color, alpha=0.3, s=50)

        # Plot connecting line
        ax2.plot(
            params_sorted, loss_sorted, color=color, linestyle=linestyle, linewidth=2
        )

    ax2.set_xlabel('Total Parameters', fontsize=12)
    ax2.set_ylabel('Velocity Loss', fontsize=12)
    ax2.set_title('Velocity Loss vs Model Size', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Add single shared legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(legend_handles),
        fontsize=10,
    )

    plt.suptitle('Pareto Frontier: Model Performance vs Size', fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {output_path}')

    plt.show()


def main():
    """Main entry point."""
    # Get the path to test_results.json
    results_dir = Path(__file__).parent
    results_path = results_dir / 'test_results.json'

    if not results_path.exists():
        raise FileNotFoundError(f'Results file not found: {results_path}')

    # Load results
    print(f'Loading results from {results_path}')
    results = load_results(results_path)
    print(f'Loaded {len(results)} result entries')

    # Create plots
    output_path = results_dir / 'pareto_frontier.png'
    plot_pareto_frontiers(results, output_path=output_path)


if __name__ == '__main__':
    main()
