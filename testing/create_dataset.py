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
    return parser.parse_args()

def create_sample(n, random, name, header, dont_ignore):
    """
    Create a sample dataset with n data points and save to a file.

    int n: Number of data points to generate
    int random: Bool of whether the dataset is random
    str name: Output file name
    str header: Header for the dataset file
    """
    # Ensure the file has a .txt extension
    if name.split(".")[-1] != "txt":
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

    # Ensure the dataset is created in the testing/ folder
    if (len(name)<6) or (name[0:7] != "testing"):
        print("The dataset must be created in the testing/ folder")
        if "/" in name:
            name = "testing/" + name.split("/")[-1]
        else:
            name = "testing/" + name
        print("Saving data to:", name)

    # Set random seed
    if random:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(0)

    # Generate independent data from a standard normal distribution
    data = rng.standard_normal(size=(n, 2))

    np.savetxt(name, data, delimiter=",", header=header)

    if dont_ignore:
        return 0

    # Add the created dataset to .gitignore if not already present
    with open(".gitignore", "r", encoding="utf-8") as f:
        if name in f.read().splitlines():
            return 0
    with open(".gitignore", "a", encoding="utf-8") as f:
        f.write("\n"+name)
    return 0

if __name__ == "__main__":
    args = parse_args()
    print("n:", args.n, "\nrandom:", args.random, "\nname:", args.name, "\nheader:", args.header)
    create_sample(args.n, args.random, args.name, args.header, args.dont_ignore)
