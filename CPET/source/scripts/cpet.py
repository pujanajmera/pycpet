from CPET.source.CPET import CPET
import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="CPET: A tool for computing and analyzing electric fields in proteins"
    )
    parser.add_argument(
        "-o",
        type=str,
        help="Options for CPET",
        default="./options/options.json",
    )

    args = parser.parse_args()
    options = args.o
    # check if the options are valid
    if not os.path.exists(options):
        raise FileNotFoundError(f"Options File {options} not found!")
    else:
        with open(options, "r") as f:
            options = json.load(f)

    cpet = CPET(options)
    cpet.run()


main()
