#!/usr/bin/env bash

# This script provides functions to plot 2D and 3D distributions of data,
# as well as training and validation loss curves.

import argparse
import numpy as np
import matplotlib.pyplot as plt


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
        choices=['2D', '3D'],
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
    parser.add_argument

    return parser.parse_args()


def plot_2d_distribution(
        data: np.ndarray,
        title: str,
        save: bool,
        filename: str,
    ) -> None:
    """
    Plot the 2D distribution of the given data.
    
    np.ndarray data: 2D array of the input data
    str title: Title of the plot
    bool save: Whether to save the plot
    str filename: Filename to save the plot to
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid()
    fig.tight_layout()
    if save:
        plt.savefig(filename + "_2D.png")
    plt.show()
    plt.close()


def plot_3d_distribution(
        data: np.ndarray,
        title: str,
        save: bool,
        filename: str,
    ) -> None:
    """
    Plots a 3D histogram of the given data.
    
    np.ndarray data: 2D array of the input data
    str title: Title of the plot
    bool save: Whether to save the plot
    str filename: Filename to save the plot to
    """
    fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')

    #make histogram stuff - set bins - I choose 20x20 because I have a lot of data
    bins = min(len(data) // 20, 30)
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)

    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()

    cmap = plt.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title(title)
    plt.xlabel("My X data source")
    plt.ylabel("My Y data source")
    if save:
        plt.savefig(filename + "_3D.png")
    plt.show()
    plt.close()

def plot_loss_curve(
        loss_values: np.ndarray,
        title: str,
        save: bool,
        filename: str,
    ) -> None:
    """
    Plots the training and validation loss curves.
    
    np.ndarray loss_values: 2D array of loss values (epoch, training_loss, validation_loss)
    str title: Title of the plot
    bool save: Whether to save the plot
    str filename: Filename to save the plot to
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(loss_values[:,0], loss_values[:,1], format="-", label="Training Loss")
    ax.plot(loss_values[:,0], loss_values[:,2], format="-", label="Validation Loss")
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
        if args.type == '2D':
            plot_2d_distribution(data, args.title, args.save, args.data_plot_name)
        elif args.type == '3D':
            plot_3d_distribution(data, args.title, args.save, args.data_plot_name)
        else:
            raise ValueError("Invalid plot type. Choose '2D' or '3D'.")

    if args.loss is not None:
        loss_values = np.loadtxt(args.loss, delimiter=',')
        plot_loss_curve(loss_values, args.title, args.save, args.loss_plot_name)

if __name__ == "__main__":
    main()
