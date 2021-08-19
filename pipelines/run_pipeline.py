from __future__ import absolute_import

import argparse
import json
import sys

from pipelines._utils import get_pipeline_driver, convert_struct, get_pipeline_custom_tags

def main():
    # Creates or updates and runs the pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for pipeline generation",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"},...]'""",
    )
    args = parser.parse_args()

    if args.module_name is None or args.role_arn is None:
        parser.print_help()
        sys.exit(2)

    tags = convert_struct(args.tags)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        print("###### Creating/updating a SageMaker Pipeline with the following definition:")
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        all_tags = get_pipeline_custom_tags(args.module_name, args.kwargs, tags)

        upsert_response = pipeline.upsert(
            role_arn=args.role_arn, description=args.description, tags=all_tags
        )
        print("\n##### Created/Updated SageMaker Pipeline: Response Received:")
        print(upsert_response)

        execution = pipeline.start()
        print(f"\n##### Execution started with PipelineExecutionArn: {execution.arn}")
        print("Waiting for execution to finish...")
        execution.wait()
        print("\n#####Execution completed. Execution step details:")
        print(execution.list_steps())
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()