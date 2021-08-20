import os
import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    # Get Sagemaker client, returning a sagemaker session instance.
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client

def get_session(region, default_bucket):
    # Gets sagemaker session based on region, returning a sagemaker.session.Session instance
    boto_session = boto3.Session(region_name = region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn
        )
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags {e}")
    return new_tags

def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AviationPackageGroup",
    pipeline_name="AviationPipeline",
    base_job_prefix="Aviation",
):
    #Gets a sagemaker ML Pipeline instance working on our aviation data, returns an instance of the pipeline
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value = 1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value = "ml.m5.xlarge")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value = "ml.m5.xlarge")
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value = "PendingManualApproval")
    input_data = ParameterString(name="InputDataUrl", default_value = f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/aviation_main.csv")

    sklearn_processor = SKLearnProcessor(
        framework_version = "0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-aviation-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_process = ProcessingStep(
        name="PreprocessAviationData",
        processor=sklearn_processor,
        output=[
            ProcessingOutput(output_name="train", source="opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR,"preprocess.py"),
        job_arguments=["--input-data", input_data]
    )

    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AviationTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/aviation-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=0.4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
    step_train = TrainingStep(
        name="TrainAviationModel",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-aviation-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    evaluation_report = PropertyFile(
        name="AviationEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateAviationModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    step_register = RegisterModel(
        name="RegisterAviationModel",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name="CheckMSEAviationEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
