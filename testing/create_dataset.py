import numpy as np
import argparse

def parse_args():
    """
    Parse custom dataset arguments
    """
    parser = argparse.ArgumentParser(
        description="Create a sample dataset for testing."
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
        default=0,
        help="Random seed for reproducibility (default: 0)\n" \
            "If seed set to -1, then it is random",
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
    return parser.parse_args()

def create_sample(n, seed, name, header):
    """
    Create a sample dataset with n data points and save to a file.

    int n: Number of data points to generate
    int seed: Random seed
    str name: Output file name
    str header: Header for the dataset file
    """
    if name.split(".")[-1] != "txt":
        if len(name.split(".")) == 1:
            print("The file should be a .txt file. Adding .txt extension")
            name += ".txt"
        elif len(name.split(".")) == 2:
            print("The file should be a .txt file. Changing extension to .txt")
            name = "".join(name.split(".")[:-1]) + ".txt"
        elif len(name.split(".")) > 2:
            print("The file should be a .txt file. Merging and changing extension to "\
                  ".txt")
            name = ".".join(name.split(".")) + ".txt"

    if (len(name)<6) or (name[0:7] != "testing"):
        print("The dataset must be created in the testing/ folder")
        if "/" in name:
            name = "testing/" + name.split("/")[-1]
        else:
            name = "testing/" + name
        print("Saving data to:", name)

    if seed != -1:
        np.random.seed(seed)

    data = np.random.randn(n, 2)

    np.savetxt(name, data, delimiter=",", header=header)
    return 0

if __name__ == "__main__":
    args = parse_args()
    print("n:", args.n, "\nseed:", args.seed, "\nname:", args.name, "\nheader:", args.header)
    create_sample(args.n, args.seed, args.name, args.header)
