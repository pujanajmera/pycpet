from CPET.source.CPET import CPET
from CPET.utils import intro_citation

import json
import os
import argparse
import logging


def main():
    parser = argparse.ArgumentParser(
        description="CPET: A tool for computing and analyzing electric fields in proteins"
    )

    # log level flags
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,  # By default, show info and warnings
    )

    parser.add_argument(
        "-o",
        type=str,
        help="Options for CPET",
        default="./options/options.json",
    )

    # Overrides to the options file

    parser.add_argument(
        "-i",
        type=str,
        help="Input path. Overrides the input path in the options file",
        default=None,
    )

    parser.add_argument(
        "-p",
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

    """
    TO-DO: Configure units argument across codebase
    parser.add_argument(
        "--units",
        type=str,
        help="Units for the output files. Default is V/Angstrom. Overrides the units in the options file",
        default=None,
    )
    """

    args = parser.parse_args()

    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    log.info("... > Starting cpet.py")

    # check if the options are valid
    if not os.path.exists(args.o):
        raise FileNotFoundError(f"Options File {args.o} not found!")
    else:
        with open(args.o, "r") as f:
            options = json.load(f)
        if args.i:
            options["inputpath"] = args.i
        if args.p:
            options["outputpath"] = args.p
        if args.m:
            options["CPET_method"] = args.m
        """
        if args.units:
            options["units"] = args.units
        """

    cpet = CPET(options)
    cpet.run()


if __name__ == "__main__":
    main()
