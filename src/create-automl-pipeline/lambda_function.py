# Requirements:
# sagemaker
# boto3>=1.24.*
# botocore>=1.27.*

# For demo purposes, this script simplifies the IAM permissions configuration when [creating required IAM roles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-service.html) that can be assumed by the SageMaker and Lambda services. The following managed policies are sufficient to run this script but should be further scoped down to improve security (least privilege principle).
# - Lambda Execution Role, named LambdaExecutionRole:
#   - AmazonSageMakerFullAccess
#   - AmazonSQSFullAccess
# - SageMaker Execution Role:
#   - AmazonSageMakerFullAccess
#   - AWSLambda_FullAccess
#   - AmazonSQSFullAccess

# Move Amazon SageMaker Autopilot ML models from experimentation to production using Amazon SageMaker Pipelines

# [Amazon SageMaker Autopilot](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html) automatically builds, trains, and tunes the best custom machine learning (ML) models based on your data. It’s an automated machine learning (AutoML) solution that eliminates the heavy lifting of handwritten ML models that requires ML expertise. Data scientists need to only provide a tabular dataset and select the target column to predict, and Autopilot automatically infers the problem type, performs data preprocessing and feature engineering, selects the algorithms and training mode, and explores different configurations to find the best ML model. Then you can directly deploy the model to an [Amazon SageMaker](https://aws.amazon.com/sagemaker/) endpoint or iterate on the recommended solutions to further improve the model quality.

# Although Autopilot eliminates the heavy lifting of building ML models, MLOps engineers still have to create, automate, and manage end-to-end ML workflows. [SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) helps you automate the different steps of the ML lifecycle, including data preprocessing, training, tuning and evaluating ML models, and deploying them. 

# This script demonstrates how to leverage SageMaker Autopilot as part of a SageMaker Pipelines end-to-end AutoML training workflow. This script has successfully been run using SageMaker Studio with the Amazon Linux 2, Jupyter Lab 3 platform identifier. When running this script with older versions of SageMaker Studio or a SageMaker script Instance, the *boto3* and/or *sagemaker* packages might need to be upgraded.

## Imports

import json
import boto3
import os
import re
import pandas as pd
import time
from datetime import datetime
import sagemaker
from sagemaker import get_execution_role, MetricsSource, ModelMetrics, ModelPackage
from sagemaker.s3 import s3_path_join
from sagemaker.image_uris import retrieve
from sagemaker.transformer import Transformer
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.workflow.callback_step import CallbackStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.steps import ProcessingStep, TransformStep, TransformInput, CacheConfig
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.execution_variables import ExecutionVariables
import cfnresponse

def create_automl_pipeline(pipeline_name='TrainingPipeline', role='lambda-role-poc1', bucket='S3Bucket' ):
    #pipeline create step
    # Check that the pipeline name satisfies the SageMaker requirements
    # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreatePipeline.html#API_CreatePipeline_RequestSyntax
    pattern = re.compile("^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}")
    if pattern.match(pipeline_name) is None:
        raise ValueError("The pipeline name does not match the pattern '^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}$'")
    
    ## Initialization
    execution_role = get_execution_role()
    pipeline_session = PipelineSession()
    sagemaker_client = boto3.client("sagemaker")
    lambda_client = boto3.client("lambda")
    aws_region = pipeline_session.boto_region_name
    sqs_client = boto3.client(
        "sqs",
        region_name=aws_region,
        endpoint_url=f"https://sqs.{aws_region}.amazonaws.com",
    )
    PROCESSING_JOB_LOCAL_BASE_PATH = "/opt/ml/processing"

    ## IAM permissions

    lambda_execution_role_name = role
    aws_account_id = boto3.client("sts").get_caller_identity().get("Account")
    LAMBDA_EXECUTION_ROLE_ARN = f"arn:aws:iam::{aws_account_id}:role/{lambda_execution_role_name}"  # to be assumed by the Lambda service

    ## SageMaker Pipelines parameters

    cache_config = CacheConfig(enable_caching=True, expire_after="T12H")
    # Required parameters.
    
    model_s3_bucket= bucket
    model_package_group_name = ParameterString(
        name="ModelPackageName",
    )
    input_uri = ParameterString(
        name="InputUri",
    )
    # Optional parameters.
    target_attribute_name = ParameterString(
        name="TargetAttributeName", 
        default_value="target",
    )
    autopilot_mode = ParameterString(
        name="AutopilotMode",
        default_value="ENSEMBLING",  # Only ENSEMBLE mode is supported at this time.
    )
    max_autopilot_candidates = ParameterInteger(
        name="MaxAutopilotCandidates", 
        default_value=16, # Only has an effect if AutopilotMode is HYPERPARAMETER_TUNING
    )
    max_autopilot_job_runtime = ParameterInteger(
        name="MaxAutoMLJobRuntimeInSeconds", 
        default_value=7200,  # 2 hours
    )
    max_autopilot_training_job_runtime = ParameterInteger(
        name="MaxRuntimePerTrainingJobInSeconds", 
        default_value=3600,  # 1 hour
    )
    instance_count = ParameterInteger(
        name="InstanceCount", 
        default_value=1,
    )
    instance_type = ParameterString(
        name="InstanceType", 
        default_value="ml.m5.xlarge",
    )
    inference_instance_types = ParameterString(
        name="InferenceInstanceType", 
        default_value="ml.m5.xlarge,ml.m5.4xlarge,ml.m5.12xlarge",
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", 
        default_value="Approved",
    )
    s3_bucket = ParameterString(
        name="S3Bucket", 
        default_value=model_s3_bucket,
    )

    # Unique names based on the pipeline execution ID.
    # The autopilot job name must match '^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,31}$':
    # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateAutoMLJob.html#API_CreateAutoMLJob_RequestSyntax
    # Since PIPELINE_EXECUTION_ID plus a dash is 13 characters long, we limit the pipeline name to 19 characters.
    # when used to form the autopilot job name.
    autopilot_job_name = Join(
        on="-", values=[pipeline_name[:19], ExecutionVariables.PIPELINE_EXECUTION_ID])
    training_output_s3_path = Join(
        on="/", values=["s3:/", s3_bucket, pipeline_name, 
                        ExecutionVariables.PIPELINE_EXECUTION_ID, "training-output"])
    batch_transform_output_s3_path = Join(
        on="/", values=["s3:/", s3_bucket, pipeline_name, 
                        ExecutionVariables.PIPELINE_EXECUTION_ID, "batch-transform-output"])

    ## Data Preprocessing Step

    # We use the publicly available [UCI Adult 1994 Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult) to predict if a person has an annual income of greater than $50,000 per year. This is a binary classification problem; the options for the income target variable are either <=50K or >50K. The dataset contains demographic information about individuals and `class` as the target column indicating the income class.

    preprocessing_script = """
import argparse
import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def problem_type(target):
    # Problem type.
    # See https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateAutoMLJob.html#sagemaker-CreateAutoMLJob-request-ProblemType
    unique_values = target.unique().shape[0]
    if is_bool_dtype(target) or unique_values == 2:
        ProblemType = "BinaryClassification"
    elif is_numeric_dtype(target):
        ProblemType = "Regression"
    else:
        ProblemType = "MulticlassClassification"
    # Objective metric for that problem type. Uses the AutoML defaults.
    # See https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AutoMLJobObjective.html
    if ProblemType == "BinaryClassification":
        JobObjective = "F1"
    elif ProblemType == "Regression":
        JobObjective = "MSE"
    else:
        JobObjective = "Accuracy"
    return ProblemType, JobObjective

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default="/opt/ml/processing")
    parser.add_argument("--input-data", type=str, default="data.csv")
    parser.add_argument("--target-name", type=str, default="target")
    parser.add_argument("--test-size", type=float, default=0.1)
    args, _ = parser.parse_known_args()

    # Input.
    input_data = args.input_data.split('/')[-1]
    input_path = f"{args.base_path}/input/{input_data}"
    print(f"Reading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Problem type and evaluation metric.
    ProblemType, JobObjective = problem_type(df[args.target_name])
    print(f"Problem type: {ProblemType!r} Job Objective: {JobObjective!r}")
    
    # Split the data.
    train_val, test = train_test_split(df, test_size=args.test_size)
    
    # Report on the target name and features.
    feature_names = [column for column in df.columns if column != args.target_name]
    print(f"Target name {args.target_name!r}")
    print(f"Feature names: {feature_names}")
    
    # Outputs.
    os.makedirs(f"{args.base_path}/train", exist_ok=True)
    train_val.to_csv(f"{args.base_path}/train/train_val.csv", header=True, index=False)
    os.makedirs(f"{args.base_path}/x_test", exist_ok=True)
    test.to_csv(f"{args.base_path}/x_test/x_test.csv", header=False, index=False, columns=feature_names)
    os.makedirs(f"{args.base_path}/y_test", exist_ok=True)
    test.to_csv(f"{args.base_path}/y_test/y_test.csv", header=False, index=False, columns=[args.target_name])
    problem_type_dict = {
        "problem_type": {
            "problem_type": {
                "value":  ProblemType,
            },
            "job_objective":{
                "value": JobObjective,
            },
        }
    }
    os.makedirs(f"{args.base_path}/problem_type", exist_ok=True)
    with open(f"{args.base_path}/problem_type/problem_type.json", "w") as f:
        f.write(json.dumps(problem_type_dict))"""

    with open('/tmp/preprocessing.py', "w") as f:
        f.write(preprocessing_script)

    # %run preprocessing.py --base-path="."

    sklearn_framework_version = "1.0-1"
    train_test_split_processor = SKLearnProcessor(
        role=execution_role,
        framework_version=sklearn_framework_version,
        instance_count=instance_count,
        instance_type=instance_type.default_value,
        sagemaker_session=pipeline_session,
    )

    train_test_split_processor_step_args = train_test_split_processor.run(
        inputs=[
            ProcessingInput(source=input_uri, destination=f"{PROCESSING_JOB_LOCAL_BASE_PATH}/input"),  
        ],
        outputs=[
            ProcessingOutput(output_name="train_val", source=f"{PROCESSING_JOB_LOCAL_BASE_PATH}/train"),
            ProcessingOutput(output_name="x_test", source=f"{PROCESSING_JOB_LOCAL_BASE_PATH}/x_test"),
            ProcessingOutput(output_name="y_test", source=f"{PROCESSING_JOB_LOCAL_BASE_PATH}/y_test"),
            ProcessingOutput(output_name="problem_type", source=f"{PROCESSING_JOB_LOCAL_BASE_PATH}/problem_type"), 
        ],
        code="/tmp/preprocessing.py",
        arguments=[
            "--input-data", input_uri,
            "--target-name", target_attribute_name,
        ]
    )

    problem_type = PropertyFile(
        name="problem_type", output_name="problem_type", path="problem_type.json"
    )

    step_train_test_split_processor = ProcessingStep(
        name="TrainTestSplitProcessing",
        step_args=train_test_split_processor_step_args,
        property_files=[problem_type],
        cache_config=cache_config,
    )

    ## Start Autopilot Job Step

    # This pipeline step uses a [Lambda step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-lambda) which runs a serverless Lambda function we create. The Lambda function in the *start_autopilot_job.py* script creates a [SageMaker Autopilot job](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_auto_ml_job).

    start_autopilot_job_script = """
import sys
from pip._internal import main

# Upgrading boto3 to the newest release to be able to use the latest SageMaker features
main(
    [
        "install",
        "-I",
        "-q",
        "boto3",
        "--target",
        "/tmp/",
        "--no-cache-dir",
        "--disable-pip-version-check",
    ]
)
sys.path.insert(0, "/tmp/")
import boto3

sagemaker_client = boto3.client("sagemaker")


def lambda_handler(event, context):
    sagemaker_client.create_auto_ml_job(
        AutoMLJobName=event["AutopilotJobName"],
        InputDataConfig=[
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": event["TrainValDatasetS3Path"],
                    }
                },
                "TargetAttributeName": event["TargetAttributeName"],
            }
        ],
        OutputDataConfig={"S3OutputPath": event["TrainingOutputS3Path"]},
        ProblemType=event["ProblemType"],
        AutoMLJobObjective={"MetricName": event["AutopilotObjectiveMetricName"]},
        AutoMLJobConfig={
            "CompletionCriteria": {
                "MaxCandidates": event["MaxCandidates"],
                "MaxRuntimePerTrainingJobInSeconds": event[
                    "MaxRuntimePerTrainingJobInSeconds"
                ],
                "MaxAutoMLJobRuntimeInSeconds": event["MaxAutoMLJobRuntimeInSeconds"],
            },
            "Mode": event["AutopilotMode"],
        },
        RoleArn=event["AutopilotExecutionRoleArn"],
    )"""

    with open('/tmp/start_autopilot_job.py', "w") as f:
        f.write(start_autopilot_job_script)

    lambda_start_autopilot_job = Lambda(
        function_name="StartSagemakerAutopilotJob",
        execution_role_arn=LAMBDA_EXECUTION_ROLE_ARN,
        script="/tmp/start_autopilot_job.py",
        handler="start_autopilot_job.lambda_handler",
        session=pipeline_session,
    )
    lambda_start_autopilot_job.upsert()
    step_start_autopilot_job = LambdaStep(
        name="StartAutopilotJobStep",
        lambda_func=lambda_start_autopilot_job,
        inputs={
            "TrainValDatasetS3Path": step_train_test_split_processor.properties.ProcessingOutputConfig.Outputs[
                    "train_val"
                ].S3Output.S3Uri,
            "MaxCandidates": max_autopilot_candidates,
            "MaxRuntimePerTrainingJobInSeconds": max_autopilot_training_job_runtime,
            "MaxAutoMLJobRuntimeInSeconds": max_autopilot_job_runtime,
            "TargetAttributeName": target_attribute_name,
            "TrainingOutputS3Path": training_output_s3_path,
            "AutopilotJobName": autopilot_job_name,
            "AutopilotExecutionRoleArn": execution_role,
            "ProblemType": JsonGet(
                    step_name=step_train_test_split_processor.name,
                    property_file=problem_type,
                    json_path="problem_type.problem_type.value",
                ),
            "AutopilotObjectiveMetricName": JsonGet(
                    step_name=step_train_test_split_processor.name,
                    property_file=problem_type,
                    json_path="problem_type.job_objective.value",
                ),
            "AutopilotMode": autopilot_mode.default_value,
        },
        cache_config=cache_config,
    )

    ## Check Autopilot Job Status Step

    # The step repeatedly keeps track of the training job status by leveraging a separate Lambda function in *check_autopilot_job_status.py* until the Autopilot training job’s completion.

    check_autopilot_job_status_script = """
import boto3
import json
import logging

sagemaker_client = boto3.client("sagemaker")


def lambda_handler(event, context):
    try:
        payload = json.loads(event["Records"][0]["body"])
        callback_token = payload["token"]
        autopilot_job = sagemaker_client.describe_auto_ml_job(
            AutoMLJobName=payload["arguments"]["AutopilotJobName"]
        )
        autopilot_job_status = autopilot_job["AutoMLJobStatus"]
        if autopilot_job_status == "Completed":
            sagemaker_client.send_pipeline_execution_step_success(
                CallbackToken=callback_token
            )
        elif autopilot_job_status in ["InProgress", "Stopping"]:
            raise ValueError("Autopilot training not finished yet. Retrying later...")
        else:
            sagemaker_client.send_pipeline_execution_step_failure(
                CallbackToken=callback_token,
                FailureReason=autopilot_job.get(
                    "FailureReason",
                    f"Autopilot training job (status: {autopilot_job_status}) failed to finish.",
                ),
            )
    except ValueError:
        raise
    except Exception as e:
        logging.exception(e)
        sagemaker_client.send_pipeline_execution_step_failure(
            CallbackToken=callback_token,
            FailureReason=str(e),
        )"""

    with open('/tmp/check_autopilot_job_status.py', "w") as f:
        f.write(check_autopilot_job_status_script)

    lambda_check_autopilot_job_status_function_name = "CheckSagemakerAutopilotJobStatus"
    lambda_check_autopilot_job_status = Lambda(
        function_name=lambda_check_autopilot_job_status_function_name,
        execution_role_arn=LAMBDA_EXECUTION_ROLE_ARN,
        script="/tmp/check_autopilot_job_status.py",
        handler="check_autopilot_job_status.lambda_handler",
        session=pipeline_session,
        timeout=15,
    )
    lambda_check_autopilot_job_status.upsert()
    queue_url = sqs_client.create_queue(
        QueueName="AutopilotSagemakerPipelinesSqsCallback",
        Attributes={"DelaySeconds": "5", "VisibilityTimeout": "300"},
    )[
        "QueueUrl"
    ]  # 5 minutes timeout
    # Add event source mapping
    try:
        response = lambda_client.create_event_source_mapping(
            EventSourceArn=sqs_client.get_queue_attributes(
                QueueUrl=queue_url, AttributeNames=["QueueArn"]
            )["Attributes"]["QueueArn"],
            FunctionName=lambda_check_autopilot_job_status_function_name,
            Enabled=True,
            BatchSize=1,
        )
    except lambda_client.exceptions.ResourceConflictException:
        pass
    step_check_autopilot_job_status_callback = CallbackStep(
        name="CheckAutopilotJobStatusCallbackStep",
        sqs_queue_url=queue_url,
        inputs={"AutopilotJobName": autopilot_job_name},
        outputs=[],
        depends_on=[step_start_autopilot_job],
    )

    ## Create Autopilot Model Step

    # The [SageMaker Processing step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-processing) creates a SageMaker model from the best trained SageMaker Autopilot model.

    create_autopilot_model_script = """
import json
import boto3
import time
import random
import string

def lambda_handler(event, context):
    print(f"Received Event: {event}")
    sagemaker_client = boto3.client("sagemaker")
    best_model = sagemaker_client.describe_auto_ml_job(
            AutoMLJobName=event["autopilot_job_name"]
        )["BestCandidate"]
    RANDOM_SUFFIX = "".join(random.choices(string.ascii_lowercase, k=8))
    model_name = event["autopilot_job_name"] + RANDOM_SUFFIX
    model_containers = best_model["InferenceContainers"]
    response = sagemaker_client.create_model(
        ModelName=model_name,
        Containers=model_containers,
        ExecutionRoleArn=event["execution_role_arn"],
    )

    return {"model_name": model_name}"""

    with open('/tmp/create_autopilot_model.py', "w") as f:
        f.write(create_autopilot_model_script)

    current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    lambda_create_autopilot_model_function_name = "sagemaker-create-autopilot-model-lambda-" + current_time

    lambda_create_autopilot_model_function = Lambda(
        function_name=lambda_create_autopilot_model_function_name,
        execution_role_arn=LAMBDA_EXECUTION_ROLE_ARN,
        script="/tmp/create_autopilot_model.py",
        handler="create_autopilot_model.lambda_handler",
    )
    lambda_create_autopilot_model_function.upsert()
    step_lambda_create_autopilot_model = LambdaStep(
        name="CreateAutopilotModelStep",
        lambda_func=lambda_create_autopilot_model_function,
        inputs={
            "autopilot_job_name": autopilot_job_name,
            "execution_role_arn": execution_role,
        },
        outputs=[
            LambdaOutput("model_name", LambdaOutputTypeEnum.String),
        ],
        depends_on=[step_check_autopilot_job_status_callback],
        cache_config=cache_config,
    )

    ## Batch Transform Step

    # We use the [Transformer object](https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer) for [batch inference](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html) on the test dataset, which can then be used for evaluation purposes in the next pipeline step.

    transformer = Transformer(
        model_name=step_lambda_create_autopilot_model.properties.Outputs["model_name"],
        instance_count=instance_count,
        instance_type=instance_type,
        # output_path=batch_transform_output_s3_path,
        sagemaker_session=pipeline_session,
    )
    step_batch_transform = TransformStep(
        name="BatchTransformStep",
        step_args=transformer.transform(data=step_train_test_split_processor.properties.ProcessingOutputConfig.Outputs[
                    "x_test"
                ].S3Output.S3Uri, content_type="text/csv"),
        cache_config=cache_config,
    )

    ## Evaluation Step

    # Defining the evaluation script used to compare the batch transform output `x_test.csv.out` to the actual (ground truth) target label `y_test.csv` using a Scikit-learn metrics function. We evaluate our results based on the [F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). The performance metrics are saved to a JSON file, which is referenced when registering the model in the subsequent step.

    evaluation_script = """
import argparse
import json
import os
import pathlib
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error,
    median_absolute_error,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default="/opt/ml/processing")
    parser.add_argument("--problem-type", type=str, default="Unspecified")
    parser.add_argument("--f1-registration-threshold", type=float, default=0.5)
    parser.add_argument("--nrmse-registration-threshold", type=float, default=0.2)
    args, _ = parser.parse_known_args()

    print("Problem type", args.problem_type)

    y_pred_path = f"{args.base_path}/input/predictions/x_test.csv.out"
    y_pred = pd.read_csv(y_pred_path, header=None)
    y_true_path = f"{args.base_path}/input/true_labels/y_test.csv"
    y_true = pd.read_csv(y_true_path, header=None)

    if args.problem_type in ["BinaryClassification", "MulticlassClassification"]:
        report_dict = {
            "model_metrics": {
                "accuracy": {
                    "value": accuracy_score(y_true, y_pred),
                    "standard_deviation": "NaN",
                },
                "weighted_f1": {
                    "value": f1_score(y_true, y_pred, average="weighted"),
                    "standard_deviation": "NaN",
                },
                "weighted_precision": {
                    "value": precision_score(y_pred, y_true, average="weighted"),
                    "standard_deviation": "NaN",
                },
                "weighted_recall": {
                    "value": recall_score(y_pred, y_true, average="weighted"),
                    "standard_deviation": "NaN",
                },
            },
            "problem_type": args.problem_type,
        }
        report_dict["register"] = int(report_dict["model_metrics"]["weighted_f1"]["value"] >= args.f1_registration_threshold)
    elif args.problem_type == "Regression":
        report_dict = {
            "model_metrics": {
                "r2_score": {
                    "value": r2_score(y_pred, y_true),
                    "standard_deviation": "NaN",                    
                },
                "root_mean_squared_error": {
                    "value": mean_squared_error(y_pred, y_true, squared=False),
                    "standard_deviation": "NaN",                    
                },
                "mean_absolute_error": {
                    "value": mean_absolute_error(y_pred, y_true),
                    "standard_deviation": "NaN",                    
                },
                "median_absolute_error": {
                    "value": median_absolute_error(y_pred, y_true),
                    "standard_deviation": "NaN",                    
                },
            },
            "problem_type": args.problem_type,
        }
        y_range = (y_true.max() - y_true.min())[0]
        report_dict["register"] = int(report_dict["model_metrics"]["root_mean_squared_error"]["value"] / y_range < args.nrmse_registration_threshold)
    else:
        raise ValueError(f"Unknown problem type {problem_type}")

    print(json.dumps(report_dict, indent=2))

    output_dir = f"{args.base_path}/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))"""

    with open('/tmp/evaluation.py', "w") as f:
        f.write(evaluation_script)

    # %run evaluation.py --base-path . --problem-type Regression

    # The evaluation script runs within a [SKLearnProcessor](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#sagemaker.sklearn.processing.SKLearnProcessor) ([SageMaker Processing](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html)) task:

    evaluation_processor = SKLearnProcessor(
        role=execution_role,
        framework_version=sklearn_framework_version,
        instance_count=instance_count,
        instance_type=instance_type.default_value,
        sagemaker_session=pipeline_session,
        # env={"ProblemType": step_auto_ml_training.properties.ProblemType},
    )

    step_args_evaluation_processor = evaluation_processor.run(
        inputs=[
            ProcessingInput(
                source=step_batch_transform.properties.TransformOutput.S3OutputPath,
                destination=f"{PROCESSING_JOB_LOCAL_BASE_PATH}/input/predictions",
            ),
            ProcessingInput(
                source=step_train_test_split_processor.properties.ProcessingOutputConfig.Outputs[
                    "y_test"
                ].S3Output.S3Uri, 
                destination=f"{PROCESSING_JOB_LOCAL_BASE_PATH}/input/true_labels"),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation_metrics",
                             source=f"{PROCESSING_JOB_LOCAL_BASE_PATH}/evaluation"), 
        ],
        code="/tmp/evaluation.py",
        arguments=["--problem-type",
                   JsonGet(
                        step_name=step_train_test_split_processor.name,
                        property_file=problem_type,
                        json_path="problem_type.problem_type.value",
                    ),]
    )

    evaluation_report = PropertyFile(
        name="evaluation", output_name="evaluation_metrics", path="evaluation_metrics.json"
    )

    step_evaluation = ProcessingStep(
        name="ModelEvaluationStep",
        step_args=step_args_evaluation_processor,
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    ## Model registration step

    # If the previously obtained evaluation metric is greater than or equal to a pre-defined model registration metric threshold, the ML model is being registered with the SageMaker model registry:

    # Using a Lambda step, the Lambda function in *register_autopilot_job.py* registers the SageMaker Autopilot model to the SageMaker Model Registry using the evaluation report obtained in the previous SageMaker Processing step. 

    register_autopilot_model_script = """
import boto3
import os
from botocore.exceptions import ClientError
from urllib.parse import urlparse

s3_client = boto3.client("s3")
sagemaker_client = boto3.client("sagemaker")


def get_report_json_s3_path(s3_path, report_name):
    o = urlparse(s3_path)
    bucket_name = o.netloc
    s3_prefix = o.path.strip("/")
    paginator = s3_client.get_paginator("list_objects_v2")
    response = paginator.paginate(
        Bucket=bucket_name, Prefix=s3_prefix, PaginationConfig={"PageSize": 1}
    )
    for page in response:
        files = page.get("Contents")
        for file in files:
            if report_name in file["Key"]:
                return os.path.join("s3://", bucket_name, file["Key"])


def lambda_handler(event, context):
    # Get the model quality and explainability results from the Autopilot job
    autopilot_job = sagemaker_client.describe_auto_ml_job(
        AutoMLJobName=event["AutopilotJobName"]
    )
    best_model = autopilot_job["BestCandidate"]
    model_containers = best_model["InferenceContainers"]
    if event["ProblemType"] in ["BinaryClassification", "MulticlassClassification"]:
        # For ENSEMBLING autopilot mode.
        model_containers[0]["Environment"]["SAGEMAKER_INFERENCE_OUTPUT"] = "predicted_label,probability,probabilities,labels"
        # Add case when autopilot_mode is HYPERPARAMETER_TUNING
    model_statistics_report_s3_path = best_model[
        "CandidateProperties"
    ]["CandidateArtifactLocations"]["ModelInsights"]
    model_statistics_report_s3_uri = get_report_json_s3_path(
        model_statistics_report_s3_path,
        "statistics.json",
    )
    explainability_report_s3_path = best_model[
        "CandidateProperties"
    ]["CandidateArtifactLocations"]["Explainability"]
    explainability_report_s3_uri = get_report_json_s3_path(
        explainability_report_s3_path,
        "analysis.json",
    )
    try:
        sagemaker_client.create_model_package_group(
            ModelPackageGroupName=event["ModelPackageGroupName"],
        )
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        if (error_code != "ValidationException"
            or "Model Package Group already exists" not in error_message):
            raise Exception(error_message)
    sagemaker_client.create_model_package(
        ModelPackageGroupName=event["ModelPackageGroupName"],
        InferenceSpecification={
            "Containers": model_containers,
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedTransformInstanceTypes": event["InstanceType"].split(","),
            "SupportedRealtimeInferenceInstanceTypes": event["InstanceType"].split(","),
        },
        ModelApprovalStatus=event["ModelApprovalStatus"],
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": ".json",
                    "S3Uri": model_statistics_report_s3_uri,
                },
            },
            "Explainability": {
                "Report": {
                    "ContentType": ".json",
                    "S3Uri": explainability_report_s3_uri,
                }
            },
        },
    )"""

    with open('/tmp/register_autopilot_model.py', "w") as f:
        f.write(register_autopilot_model_script)

    lambda_register_autopilot_model = Lambda(
        function_name="RegisterSagemakerAutopilotModel",
        execution_role_arn=LAMBDA_EXECUTION_ROLE_ARN,
        script="/tmp/register_autopilot_model.py",
        handler="register_autopilot_model.lambda_handler",
        session=pipeline_session,
        timeout=15,
    )
    lambda_register_autopilot_model.upsert()
    step_register_autopilot_model = LambdaStep(
        name="RegisterAutopilotModelStep",
        lambda_func=lambda_register_autopilot_model,
        inputs={
            "AutopilotJobName": autopilot_job_name,
            "ModelPackageGroupName": model_package_group_name,
            "ModelApprovalStatus": model_approval_status,
            "InstanceType": inference_instance_types,
            "ProblemType": JsonGet(
                step_name=step_train_test_split_processor.name,
                property_file=problem_type,
                json_path="problem_type.problem_type.value",
            ),
        },
        cache_config=cache_config,
    )

    ## Fail Step
    # This step will cause the pipeline to loudly fail when the evaluation bar is not met.

    step_fail = FailStep(
        name="FailStep",
        error_message="The model's performace does not pass the threshold",
    )

    ## Condition step
    # This step will execute the model registration and deployment step if the condition is met.

    step_conditional_registration = ConditionStep(
        name="ConditionalRegistrationStep",
        conditions=[
            ConditionGreaterThan(
                left=JsonGet(
                    step_name=step_evaluation.name,
                    property_file=evaluation_report,
                    json_path="register",
                ),
                right=0,
            )
        ],
        if_steps=[
            step_register_autopilot_model,
        ],
        else_steps=[
            step_fail,
        ],  # pipeline end
    )

    ## Create and run pipeline

    # Once the pipeline steps are defined, we combine them into a SageMaker Pipeline. The steps are run in sequential order. The pipeline executes all of the steps for an AutoML job leveraging SageMaker Autopilot and SageMaker Pipelines for training, model evaluation and model registration.

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_package_group_name,
            target_attribute_name,
            # autopilot_mode,    # Only ENSEMBLE mode is supported at this time.
            max_autopilot_candidates,
            max_autopilot_job_runtime,
            max_autopilot_training_job_runtime,
            instance_count,
            instance_type,
            inference_instance_types,
            model_approval_status,
            input_uri,
            s3_bucket,
        ],
        steps=[
            step_train_test_split_processor,
            step_start_autopilot_job,
            step_check_autopilot_job_status_callback,
            step_lambda_create_autopilot_model,
            step_batch_transform,
            step_evaluation,
            step_conditional_registration,
        ],
        sagemaker_session=pipeline_session,
    )
    
    # Create or update the pipeline.
    pipeline.upsert(role_arn=execution_role)
    
    return pipeline


# if __name__ == "__main__":
def lambda_handler(event, context):
    
    # Create the pipeline.
    # pipeline_name = "TrainingPipelinelambda"
    print("Log the received event")
    print("Received event: " + json.dumps(event, indent=2))
    
    request_type =event['RequestType']
    pipeline_name = event['ResourceProperties']['pipelinename']
    role_name = event['ResourceProperties']['role']
    model_bucket= event['ResourceProperties']['bucket']
    
    if request_type == 'Create':
    
        try:
            create_automl_pipeline(pipeline_name, role_name, model_bucket)
            
            # Show how to use the created pipeline by just using its name.
            # Get a handle to the pipeline.
            from sagemaker.workflow.pipeline import Pipeline
            pipeline = Pipeline(name=pipeline_name)

            # # Show the parameters.
            # # import json
            # import pandas as pd
            # pipeline_description = pipeline.describe()
            # pipeline_parameters = json.loads(pipeline_description['PipelineDefinition'])
            # print(pd.DataFrame(pipeline_parameters['Parameters']))
            
            responseData={}
            responseData['Data']="automl pipeline created successfully"
            cfnresponse.send(event, context, cfnresponse.SUCCESS, responseData)
            return "SUCCESS"
        
        except:
            responseData={}
            print("automl pipeline creation failed")
            responseData['Data']="Failed to create automl pipeline"
            cfnresponse.send(event, context, cfnresponse.FAILED, responseData)
            return "FAILED"
        
    else:
        responseData={}
        responseData['Data']="No action taken for cloudformation " + request_type + " request"
        cfnresponse.send(event, context, cfnresponse.SUCCESS, responseData)
        return "SUCCESS"
