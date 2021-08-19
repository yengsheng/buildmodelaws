from __future__ import absolute_import

import ast

def get_pipeline_driver(module_name, passed_args=None):
    # Args: module_name, passed_args
    # Pipeline modules must define get_pipeline() module-level method
    # Returns the Sagemaker Workflow pipeline

    _imports = __import__(module_name, fromlist=["get_pipeline"])
    kwargs = convert_struct(passed_args)
    return _imports.get_pipeline(**kwargs)

def convert_struct(str_struct=None):
    return ast.literal_eval(str_struct) if str_struct else {}

def get_pipeline_custom_tags(module_name, args, tags):
    # Gets custom tags for pipeline
    try:
        _imports = __import__(module_name, fromlist=["get_pipeline_custom_tags"])
        kwargs = convert_struct(args)
        return _imports.get_pipeline_custom_tags(tags, kwargs['region'], kwargs['sagemaker_project_arn'])
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return tags    
