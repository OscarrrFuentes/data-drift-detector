#!/usr/bin/env python3
"""
Create a sample dataset for testing.
"""

import sys
import argparse
import textwrap
import logging
import numpy as np
from data_drift_detector.utils.load_json import load_json


ALLOWED_DISTRIBUTIONS = ["normal",
                         "uniform",
                         "exponential",
                         "multivariate_normal",
                         "poisson",
                         ]


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler(sys.stdout))


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
        "--seed",
        type=int,
        default=None,
        help=("Random seed set (default: None)"),
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
        choices=ALLOWED_DISTRIBUTIONS,
        help="Distribution to sample from (default: None, standard normal)",
    )
    parser.add_argument(
        "--print_distributions",
        action="store_true",
        help="Flag to print all acceptable distributions",
    )
    parser.add_argument(
        "--drift_data",
        type=load_json,
        default=None,
        help=("JSON filepath or string dictionary containing instructions to drift the"
                "mock dataset. Pass --drift_data_dict_keys to see full description of "
                "allowed values"
        ),
    )
    parser.add_argument(
        "--drift_data_dict_keys",
        action="store_true",
        help="Flag to print all of the currently accepted changes to the mock dataset"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Printing info, -vv for debug info"
    )
    parser.add_argument(
        "-vv",
        action="store_true"
    )
    return parser.parse_args()


def set_logger(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Set the verbosity of the logger
    
    argparse.Namespace args: Arguments passed into the program
    logging.Logger logger: Logger to set verbosity for
    """
    if args.vv:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)


def check_extension(name: str, logger: logging.Logger) -> tuple[str, bool]:
    """
    Check if the file name has a .txt extension.

    str name: File name to check
    logging.Logger logger: Logger to use for logging messages
    return: Corrected file name with .txt extension
    """
    edit_flag = False

    if name.split(".")[-1] != "txt":
        edit_flag = True

        # There is no extension at all
        if len(name.split(".")) == 1:
            logger.warning("The file should be a .txt file. Adding .txt extension")
            name += ".txt"

        # There is one extension but it is not .txt
        elif len(name.split(".")) == 2:
            logger.warning("The file should be a .txt file. Changing extension to .txt")
            name = "".join(name.split(".")[:-1]) + ".txt"

        # There are multiple extensions
        elif len(name.split(".")) > 2:
            logger.warning("The file should be a .txt file. Merging and changing "
                           "extension to .txt")
            name = ".".join(name.split(".")) + ".txt"
    return (name, edit_flag)


def check_folder(
        name: str,
        logger: logging.Logger,
        edit_flag: bool = False,
    ) -> tuple[str, bool]:
    """
    Ensure the dataset is created in the testing/ folder.

    str name: File name to check
    logging.Logger logger: Logger to use for logging messages
    bool edit_flag: Flag to indicate whether the file name was edited
    return: Corrected file name in testing/ folder
    """
    if (len(name)<=6) or (name[0:7] != "testing"):
        edit_flag = True
        logger.warning("The dataset must be created in the testing/ folder,"
                       " changing folder to testing/")
        if "/" in name:
            name = "testing/" + name.split("/")[-1]
        else:
            name = "testing/" + name
    return (name, edit_flag)


def multivariate_drift_data(
        rng: np.random.Generator,
        n: int,
        drift_data: dict,
        data_dict: dict,
        logger: logging.Logger,
    ) -> np.ndarray:
    """
    Create a drifted dataset for the multivariate normal distribution.

    np.random.Generator rng: Random number generator
    int n: Number of data points to generate
    dict | None drift_data: Parameters to drift the data by
    dict | None data_dict: Original dictionary of parameters of mock dataset
    logging.Logger logger: Logger to use for logging messages
    return: Generated dataset
    """

    # Determine the number of data drifts
    drift_lengths = set()
    for value in drift_data.values():
        if np.ndim(value) > 1:
            drift_lengths.add(len(value))
        else:
            drift_lengths.add(1)

    if len(drift_lengths) == 0:
        num_drifts = 1
    elif len(drift_lengths) == 1:
        num_drifts = drift_lengths.pop()
    else:
        raise ValueError("All drift_data dict key:values must have the same length.")

    # Checking n is divisible by number of drifts
    if n % num_drifts != 0:
        logger.warning("\n\nDATA DRIFT NUMBER WARNING:\n%s is not divisible by %s",
                       n,
                       num_drifts,
                       )
        n += num_drifts - (n % num_drifts)
        logger.warning("Rounding n up to %s", n)

    # Build data_dict with drift_data values
    tmp_data_dict = data_dict.copy()
    for key in tmp_data_dict.keys():
        # Check whether all of the keys in data_dict are in drift_data
        # otherwise fill with original value repeated num_drifts times
        try:
            # Check for valid covariance matrix
            if key == "cov":
                if np.any(np.linalg.eigvalsh(drift_data["cov"]) < 0):
                    raise ValueError("Covariance matrix eigenvalues must be positive")
                elif np.any(drift_data["cov"][:,0,1] != drift_data["cov"][:,1,0]):
                    raise ValueError("Covariance matrix must be symmetric")
            data_dict[key] = drift_data[key]

        except KeyError:
            data_dict[key] = [data_dict[key] for _ in range(num_drifts)]

    # Check whether all of the keys in drift_data are valid
    for key in drift_data.keys():
        if key not in data_dict.keys():
            raise KeyError(f"Key \"{key}\" in drift_data is not valid.")

    data = np.empty(shape=(0,drift_data["loc"][0].shape[0]))
    for loc_xy, cov in zip(data_dict["loc"], data_dict["cov"]):
        data = np.vstack((data, rng.multivariate_normal(loc_xy,
                                                        cov,
                                                        size=int(n/num_drifts)
                                                        )))

    return data


def create_dataset(
        rng: np.random.Generator,
        n: int,
        distribution: str | None,
        drift_data: dict | None,
        logger: logging.Logger,
    ) -> np.ndarray:
    """
    Create a 2D dataset with n data points sampled from distribution.

    np.random.Generator rng: Random number generator
    int n: Number of data points to generate
    str | None distribution: Distribution to sample from
    dict | None drift_data: Parameters to drift the data by
    logging.Logger logger: Logger to use for logging messages
    return: Generated dataset
    """

    data_dict = {"loc": [0, 0], "stdev": [1, 1], "cov": [[5, 0], [0, 5]]}

    # Generate 2D data based on specified distribution
    match distribution:
        case "uniform":
            data = rng.uniform(low=0.0, high=1.0, size=(n, 2))

        case "exponential":
            data = rng.exponential(scale=1.0, size=(n, 2))

        case "multivariate_normal":
            if drift_data:
                data = multivariate_drift_data(rng, n, drift_data, data_dict, logger)
            else:
                data = rng.multivariate_normal(data_dict["loc"],
                                               data_dict["cov"],
                                               size=n
                                               )

        case "poisson":
            data = rng.poisson(lam=10.0, size=(n, 2))

        case "normal" | None:
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


def create_sample(
        n: int,
        seed: int,
        name: str,
        header: str,
        dont_ignore: bool,
        distribution: str | None,
        drift_data: dict | None,
        logger: logging.Logger,
    ) -> None:
    """
    Create a sample dataset with n data points and save to a file.

    int n: Number of data points to generate
    int seed: Random seed for the random generator
    str name: Output file name
    str header: Header for the dataset file
    bool dont_ignore: If True, do not include the dataset file in .gitignore
    logging.Logger logger: Logger to use for logging messages
    str | None distribution: Distribution to sample from (default: standard normal)
    dict | None drift_data: Parameters to drift the data by (default: None)
    logging.Logger logger: Logger to use for logging messages
    """
    # Check file name
    logger.debug("Checking file name...")
    (name, edit_flag) = check_extension(name, logger)
    (name, edit_flag) = check_folder(name, logger, edit_flag=edit_flag)
    if edit_flag:
        logger.warning("Saving file to %s", name)
    logger.debug("\nChecking file name done")

    # Set random seed
    rng = np.random.default_rng(seed)

    # Generate data from distribution
    logger.debug("Generating dataset...")
    data = create_dataset(rng, n, distribution, drift_data, logger)
    logger.debug("\nGenerating dataset done")

    np.savetxt(name, data, delimiter=",", header=header)

    if not dont_ignore:
        # Add the created dataset to .gitignore if not already present
        add_to_gitignore(name)
    logger.debug("\n%s saved", name)


def print_info(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Print dataset information
    
    argparse.Namespace args: Arguments passed to this program
    logging.Logger logger: Logger to use for printing info
    """
    if args.print_distributions:
        print("\nALLOWED DISTRIBUTIONS:\n")
        for d in ALLOWED_DISTRIBUTIONS:
            print(d)
        print("\n-----------------------\n")
        sys.exit(0)

    if args.drift_data_dict_keys:
        logger.info("\n%s", textwrap.fill("ALLOWED drift_data DICT KEY:VALUE PAIRS"))
        logger.info("\n%s", textwrap.fill("\"loc\": tuple[float, float] | "
                            "list[tuple[float, float]] - (x, y) locations for the "
                            "drift to happen.",
                            width=80,
                            subsequent_indent="   ",), "\n",
              textwrap.fill("\"stdev\": "
                            "tuple[float, float] | list[tuple[float, float]] - "
                            "Changes to (x, y) standard deviations at each loc. If "
                            "list[tuple[float, float]], "
                            "it must be the same length as Loc.",
                            width=80,
                            subsequent_indent="   ",), "\n",
              textwrap.fill("\"cov\": "
                            "tuple[tuple[float, float], tuple[float, float]] | "
                            "list[tuple[tuple[float, float], tuple[float, float]]] - "
                            "MULTIVARIATE NORMAL DISTRIBUTION ONLY. "
                            "Change the covariance matrix of the multivariate normal.",
                            width=80,
                            subsequent_indent="   ",
                            ),
              "\n\n----------------------\n",
              )

    logger.info("\nINPUTS:\n\nn: %s\nseed: %s\nname: %s"
                "\nheader: %s\ndistribution: "
                "%s"
                "\ndrift data:{",
                args.n, (args.seed if args.seed else "random"), args.name, args.header,
                args.distribution if args.distribution else "normal")

    if args.drift_data:
        for key, value in args.drift_data.items():
            logger.info("%s: %s\n", key, value)
        logger.info(
            "}\n----------------------\n",
            )


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.drift_data is not None:
        for k, v in parsed_args.drift_data.items():
            parsed_args.drift_data[k] = np.array(v)
    set_logger(parsed_args, logger)
    print_info(parsed_args, logger)
    create_sample(
        parsed_args.n,
        parsed_args.seed,
        parsed_args.name,
        parsed_args.header,
        parsed_args.dont_ignore,
        parsed_args.distribution,
        parsed_args.drift_data,
        logger,
    )
