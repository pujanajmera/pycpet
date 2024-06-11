from CPET.source.CPET import CPET
import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="CPET: A tool for computing and analyzing electric fields in proteins"
    )
    parser.add_argument(
        "-o",
        type=json.loads,
        help="Options for CPET",
        default=json.load(open("./options/options.json")),
    )
    args = parser.parse_args()
    options = args.o
    cpet = CPET(options)
    cpet.run()


main()
