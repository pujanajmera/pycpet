print("Importing...")
from CPET.source.CPET import CPET

print("Importing done!")
import json
import os
import argparse
import logging


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

    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    # Overrides to the options file

    parser.add_argument(
        "-i",
        type=str,
        help="Input path. Overrides the input path in the options file",
        default=None,
    )

    parser.add_argument(
        "-d",
        type=str,
        help="Output path. Overrides the output path in the options file",
        default=None,
    )

    parser.add_argument(
        "-m",
        type=str,
        help="pyCPET method. Overrides the method in the options file",
        default=None,
    )

    parser.add_argument(
        "--units",
        type=str,
        help="Units for the output files. Default is V/Angstrom. Overrides the units in the options file",
        default=None,
    )

    args = parser.parse_args()
    options = args.o
    logging.basicConfig(level=args.loglevel)

    # check if the options are valid
    if not os.path.exists(options):
        raise FileNotFoundError(f"Options File {options} not found!")
    else:
        with open(options, "r") as f:
            options = json.load(f)
        if args.i:
            options["inputpath"] = args.i
        if args.d:
            options["outputpath"] = args.d
        if args.m:
            options["CPET_method"] = args.m
        if args.units:
            options["units"] = args.units

    cpet = CPET(options)
    cpet.run()


if __name__ == "__main__":
    main()
