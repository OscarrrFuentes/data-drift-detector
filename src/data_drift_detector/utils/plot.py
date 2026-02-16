#!/usr/bin/env python3
"""
This script provides functions to plot 2D and 3D distributions of data,
as well as training and validation loss curves.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_drift_detector.utils.load_json import load_json


def parse_args() -> argparse.Namespace:
    """
    Parse the input arguments for plotting distributions and loss curves.
    
    returns: An argparse.Namespace of the parsed arguments.
    """

    parser = argparse.ArgumentParser(description='Plot distributions and loss curves.')
    parser.add_argument(
        '--data',
        type=str,
        help='Path to the data file (CSV format).',
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['2D_scatter', '2D_heatmap', '3D'],
        help='Type of distribution to plot.',
    )
    parser.add_argument(
        '--title',
        type=str,
        help='Title for the plot.',
    )
    parser.add_argument(
        '--loss',
        type=str,
        help='Path to the loss values file (CSV format).',
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the plots."
    )
    parser.add_argument(
        "--data_plot_name",
        type=str,
        default="plots/distribution_plot",
        help="Filename for the distribution plot.",
    )
    parser.add_argument(
        "--loss_plot_name",
        type=str,
        default="plots/loss_curve.png",
        help="Filename for the loss curve plot.",
    )
    parser.add_argument(
        "--distribution_kwargs",
        type=load_json,
        default={},
        help=("Path to a JSON file or a JSON string containing additional keyword "
        "arguments for plotting the input data distribution."),
    )
    parser.add_argument(
        "--train_kwargs",
        type=load_json,
        default={},
        help=("Path to a JSON file or a JSON string containing additional keyword "
        "arguments for plotting the training loss curve."),
    )
    parser.add_argument(
        "--val_kwargs",
        type=load_json,
        default={},
        help=("Path to a JSON file or a JSON string containing additional keyword "
        "arguments for plotting the validation loss curve."),
    )

    return parser.parse_args()


def plot_2d_scatter(
        data: np.ndarray,
        title: str,
        save: bool,
        filename: str,
        kwargs: dict,
    ) -> None:
    """
    Plot the 2D scatterplot distribution of the given data.
    
    np.ndarray data: 2D array of the input data
    str title: Title of the plot
    bool save: Whether to save the plot
    str filename: Filename to save the plot to
    dict kwargs: Additional keyword arguments for the scatter plot
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, **kwargs)
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid()
    fig.tight_layout()
    if save:
        plt.savefig(filename + "_2D_scatter.png")
    plt.show()
    plt.close()


def plot_2d_heatmap(
        data: np.ndarray,
        title: str,
        save: bool,
        filename: str,
        kwargs: dict,
    ) -> None:
    """
    Plot a 2D heatmap of the given data.
    
    np.ndarray data: 2D array of the input data
    str title: Title of the plot
    bool save: Whether to save the plot
    str filename: Filename to save the plot to
    dict kwargs: Additional keyword arguments for the heatmap
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = min(len(data) // 10, 100) # Adjust the number of bins based on the data size
    heatmap, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax.imshow(
        heatmap.T,
        cmap="jet",
        extent=extent,
        origin='lower',
        aspect='auto',
        **kwargs
    )
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save:
        plt.savefig(filename + "_2D_heatmap.png")
    plt.show()
    plt.close()


def plot_3d_distribution(
        data: np.ndarray,
        title: str,
        save: bool,
        filename: str,
        kwargs: dict,
    ) -> None:
    """
    Plots a 3D histogram of the given data.
    
    np.ndarray data: 2D array of the input data
    str title: Title of the plot
    bool save: Whether to save the plot
    str filename: Filename to save the plot to
    dict kwargs: Additional keyword arguments for the histogram plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    bins = min(len(data) // 20, 30) # Adjust the number of bins based on the data size
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like(xpos)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = hist.flatten()

    cmap = plt.get_cmap('jet')
    max_height = np.max(dz)
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average', **kwargs)
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Frequency')

    if save:
        plt.savefig(filename + "_3D.png")
    plt.show()
    plt.close()

def plot_loss_curve(
        loss_values: np.ndarray,
        title: str,
        save: bool,
        filename: str,
        train_kwargs: dict,
        val_kwargs: dict,
    ) -> None:
    """
    Plots the training and validation loss curves.
    
    np.ndarray loss_values: 2D array of loss values (epoch, training_loss, validation_loss)
    str title: Title of the plot
    bool save: Whether to save the plot
    str filename: Filename to save the plot to
    dict train_kwargs: Additional keyword arguments for the training loss line
    dict val_kwargs: Additional keyword arguments for the validation loss line
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        loss_values[:,0],
        loss_values[:,1],
        "-",
        label="Training Loss",
        **train_kwargs
    )
    ax.plot(
        loss_values[:,0],
        loss_values[:,2], 
        "-",
        label="Validation Loss",
        **val_kwargs
    )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()


def main():
    args = parse_args()

    if args.data is not None:
        data = np.loadtxt(args.data, delimiter=',')
        if args.type == '2D_scatter':
            plot_2d_scatter(
                data,
                args.title,
                args.save,
                args.data_plot_name,
                args.distribution_kwargs
            )
        elif args.type == '2D_heatmap':
            plot_2d_heatmap(
                data,
                args.title,
                args.save,
                args.data_plot_name,
                args.distribution_kwargs
            )
        elif args.type == '3D':
            plot_3d_distribution(
                data,
                args.title,
                args.save,
                args.data_plot_name,
                args.distribution_kwargs,
            )
        else:
            raise ValueError("Invalid plot type. Choose '2D' or '3D'.")

    if args.loss is not None:
        loss_values = np.loadtxt(args.loss, delimiter=',')
        plot_loss_curve(
            loss_values,
            args.title,
            args.save,
            args.loss_plot_name,
            args.train_kwargs,
            args.val_kwargs,
        )

if __name__ == "__main__":
    main()
