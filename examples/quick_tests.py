import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--integer', type=int,
                        help='an integer for the accumulator')
    parser.add_argument("--int2", type=int,
                        help="another integer for the accumulator")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Integer 1: {args.integer}")
    print(f"Integer 2: {args.int2}")
    