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

    #Have other arguments as overrides to the options file

    parser.add_argument(
        "-i",
        type=str,
        help="Input path. Overrides the input path in the options file",
        default=None
    )

    parser.add_argument(
        "-d",
        type=str,
        help="Output path. Overrides the output path in the options file",
        default=None
    )

    parser.add_argument(
        "-m",
        type=str,
        help="pyCPET method. Overrides the method in the options file",
        default=None
    )


    args = parser.parse_args()
    options = args.o

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

    cpet = CPET(options)
    cpet.run()


main()
