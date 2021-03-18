"""Pretty print a json file

Author:
    Jeffrey Shen
"""
import argparse
import json

def main():
    parser = argparse.ArgumentParser("Train a model on SQuAD")

    parser.add_argument(
        "--in_file", type=str, help="Json file to pretty print."
    )

    parser.add_argument("--out_file", type=str, default="pretty.json", help="Json out file")
    args = parser.parse_args()
    with open(args.in_file) as in_file:
        x = json.load(in_file)
        with open(args.out_file, "w") as out_file:
            json.dump(x, out_file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
