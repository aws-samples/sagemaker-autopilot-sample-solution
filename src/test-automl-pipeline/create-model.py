import argparse
import json
import pandas as pd
from sagemaker.workflow.pipeline import Pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-name", type=str, default="TrainingPipeline")
    parser.add_argument("--input-uri", type=str, default="data.csv")
    parser.add_argument("--target-attribute-name", type=str, default="target")
    parser.add_argument("--model-package-name", type=str, default="ModelPackage")
    parser.add_argument("--wait", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    args, _ = parser.parse_known_args()

    # Get a handle to the pipeline.
    pipeline = Pipeline(name=args.pipeline_name)

    # Show its parameters.
    if args.verbose:
        pipeline_description = pipeline.describe()
        pipeline_parameters = json.loads(pipeline_description['PipelineDefinition'])
        print(pd.DataFrame(pipeline_parameters['Parameters']))

    # Start the pipeline, passing the parameter values.
    if args.verbose:
        print(f"Executing pipeline {args.pipeline_name}")
        print(f"Input URI {args.input_uri}")
        print(f"Target attribute4 name {args.target_attribute_name}")
        print(f"Model package name {args.model_package_name}")
    pipeline_execution = pipeline.start(
        parameters=dict(
            InputUri=args.input_uri,
            TargetAttributeName=args.target_attribute_name,
            ModelPackageName=args.model_package_name,
        )
    )

    # Print the pipeline execution status.
    if args.verbose:
        print(pipeline_execution.describe())

    # Wait for the pipeline to finish.
    if args.wait:
        if args.verbose:
            print("Waiting for completion.")
        pipeline_execution.wait(delay=20, max_attempts=24 * 60 * 3)
