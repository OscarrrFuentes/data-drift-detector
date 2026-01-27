#!/usr/bin/env python3
"""
Create a sample dataset for testing.
"""

import argparse
import textwrap
import numpy as np


ALLOWED_DISTRIBUTIONS = ["normal",
                         "uniform",
                         "exponential",
                         "multivariate_normal",
                         "poisson",
                         ]


def parse_args() -> argparse.Namespace:
    """
    Parse custom dataset arguments
    """
    parser = argparse.ArgumentParser(
        description="Create a sample dataset for testing.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of data points to generate (default: 1000)",
    )
    parser.add_argument(
        "--random",
        action = "store_true",
        help="If set, generate a random dataset (default: False)\n",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="testing/mock_dataset.txt",
        help="Output file name (default: testing/mock_dataset.txt)",
    )
    parser.add_argument(
        "--header",
        type=str,
        default="x,y",
        help="Header for the dataset file (default: 'x,y')",
    )
    parser.add_argument(
        "--dont_ignore",
        action="store_true",
        help="If set, do not include in .gitignore (default: False)\n",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default=None,
        help="Distribution to sample from (default: None, standard normal)",
    )
    parser.add_argument(
        "--print_distributions",
        action="store_true",
        help="Flag to print all acceptable distributions",
    )
    parser.add_argument(
        "--drift_data",
        type=dict,
        default=None,
        help=("Dictionary containing instructions to drift the mock dataset\n"\
              "Pass --drift_data_dict_keys to see full description of allowed values"
),
    )
    parser.add_argument(
        "--drift_data_dict_keys",
        action="store_true",
        help="Flag to print all of the currently accepted changes to the mock dataset"
    )
    return parser.parse_args()


def check_extension(name: str, edit_flag: bool = False) -> tuple[str, bool]:
    """
    Check if the file name has a .txt extension.

    str name: File name to check
    return: Corrected file name with .txt extension
    """
    if name.split(".")[-1] != "txt":
        edit_flag = True

        # There is no extension at all
        if len(name.split(".")) == 1:
            print("The file should be a .txt file. Adding .txt extension")
            name += ".txt"

        # There is one extension but it is not .txt    
        elif len(name.split(".")) == 2:
            print("The file should be a .txt file. Changing extension to .txt")
            name = "".join(name.split(".")[:-1]) + ".txt"
        
        # There are multiple extensions
        elif len(name.split(".")) > 2:
            print("The file should be a .txt file. Merging and changing extension to "\
                  ".txt")
            name = ".".join(name.split(".")) + ".txt"
    return (name, edit_flag)


def check_folder(name: str, edit_flag: bool = False) -> tuple[str, bool]:
    """
    Ensure the dataset is created in the testing/ folder.

    str name: File name to check
    return: Corrected file name in testing/ folder
    """
    if (len(name)<=6) or (name[0:7] != "testing"):
        edit_flag = True
        print("The dataset must be created in the testing/ folder,"\
              " changing folder to testing/")
        if "/" in name:
            name = "testing/" + name.split("/")[-1]
        else:
            name = "testing/" + name
    return (name, edit_flag)


def create_dataset(rng: np.random.Generator,
                   n: int,
                   distribution: str | None = None,
                   ) -> np.ndarray:
    """
    Create a 2D dataset with n data points sampled from distribution.

    np.random.Generator rng: Random number generator
    int n: Number of data points to generate
    str | None distribution: Distribution to sample from
    return: Generated dataset
    """

    # Generate 2D data based on specified distribution
    match distribution:
        case "uniform":
            data = rng.uniform(low=0.0, high=1.0, size=(n, 2))
        case "exponential":
            data = rng.exponential(scale=1.0, size=(n, 2))
        case "multivariate_normal":
            mean = [0, 0]
            cov = [[1, 0.9], [0.9, 1]]
            data = rng.multivariate_normal(mean, cov, size=n)
        case "poisson":
            data = rng.poisson(lam=10.0, size=(n, 2))
        case "normal" | None:
            data = rng.standard_normal(size=(n, 2))
        case _:
            print(f"Distribution \"{distribution}\" not allowed.\n"
              "Generating standard normal data instead.")
            data = rng.standard_normal(size=(n, 2))

    return data


def add_to_gitignore(name: str) -> None:
    """
    Add the created dataset to .gitignore if not already present.

    str name: File name to add to .gitignore
    """

    with open(".gitignore", "r", encoding="utf-8") as f:
        if name in f.read().splitlines():
            return

    with open(".gitignore", "a", encoding="utf-8") as f:
        f.write("\n"+name)
    return


def create_sample(n: int,
                  random: bool,
                  name: str,
                  header: str,
                  dont_ignore: bool,
                  distribution: str | None = None,
                  ) -> int:
    """
    Create a sample dataset with n data points and save to a file.

    int n: Number of data points to generate
    int random: Bool of whether the dataset is random
    str name: Output file name
    str header: Header for the dataset file
    """
    # Check file name
    (name, edit_flag) = check_extension(name)
    (name, edit_flag) = check_folder(name, edit_flag=edit_flag)
    if edit_flag:
        print(f"Saving file to {name}")

    # Set random seed
    if random:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(0)

    # Generate data from distribution
    data = create_dataset(rng, n, distribution)

    np.savetxt(name, data, delimiter=",", header=header)

    if dont_ignore:
        return 0

    # Add the created dataset to .gitignore if not already present
    add_to_gitignore(name)
    return 0


def print_info(args: argparse.Namespace) -> None:
    """
    Print dataset information
    
    argparse.Namespace args: Arguments passed to this program
    """
    if args.print_distributions:
        print("\nALLOWED DISTRIBUTIONS:\n")
        for d in ALLOWED_DISTRIBUTIONS:
            print(d)
        print("\n-----------------------\n")

    if args.drift_data_dict_keys:
        print("\n", textwrap.fill("ALLOWED drift_data DICT KEY:VALUE PAIRS"), "\n\n",
              textwrap.fill("\"loc\": tuple[float, float] | list[tuple[float, float]] - "
                            "(x, y) locations for the drift to happen.",
                            width=80,
                            subsequent_indent="   ",), "\n",
              textwrap.fill("\"stdev\": "
                            "tuple[float, float] | list[tuple[float, float]] - "
                            "Changes to (x, y) standard deviations at each loc. If "
                            "list[tuple[float, float]], "
                            "it must be the same length as Loc.",
                            width=80,
                            subsequent_indent="   ",), "\n",
              textwrap.fill("\"covariance_matrix\": "
                            "tuple[tuple[float, float], tuple[float, float]] - "
                            "MULTIVARIATE NORMAL DISTRIBUTION ONLY. "
                            "Change the covariance matrix of the multivariate normal.",
                            width=80,
                            subsequent_indent="   ",
                            ),
              "\n\n----------------------\n",
              )

    print("\nINPUTS:\n",
          "\nn:", args.n,
          "\nrandom:", args.random,
          "\nname:", args.name,
          "\nheader:", args.header,
          "\ndistribution:", (args.distribution if args.distribution else "normal"),
          "\n\n----------------------\n",
          )


if __name__ == "__main__":
    args = parse_args()
    print_info(args)
    create_sample(args.n,
                  args.random,
                  args.name,
                  args.header,
                  args.dont_ignore,
                  args.distribution)
