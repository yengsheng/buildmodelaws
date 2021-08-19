from __future__ import absolute_import

import argparse
import sys

from pipelines._utils import get_pipeline_driver

def main():
    # Gets the pipeline definition JSON, and either prints to stdout or saves to a file

    parser = argparse.ArgumentParser("Gets the pipeline definition for the pipeline script.")

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-f",
        "--file-name",
        dest="file_name",
        type=str,
        default=None,
        help="The file to output the pipeline definition json to."
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments from pipeline generation",
    )
    args = parser.parse_args()

    if args.module_name is None:
        parser.print_help()
        sys.exit(2)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        content = pipeline.definition()
        if args.file_name:
            with open(args.file_name, "w") as f:
                f.write(content)
        else:
            print(content)
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()