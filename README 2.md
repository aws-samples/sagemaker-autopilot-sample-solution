## Sagemaker-Autopilot-ML-Solution
This solution automates the infrastructure required for creating and consuming ML models on Sagemaker (including training and inference) through sagemaker pipelines, sagemaker autopilot, and sagemaker endpoints. You can deploy the solution through the cloud formation templates.

## Reference Architecture
![](https://github.com/aws-samples/sagemaker-autopilot-sample-solution/blob/main/ML-Ref-Architecture.jpg)

## Pre-requisites
* AWS Account with access to IAM and other services outlined in the reference architecture
* AWS CLI
* python 3
* git
  
## Packaging code and dependencies for lambda
Sagemaker python sdk is not supported inside lambda and hence we have to package it as a layer to be used in lambda function for sagemaker pipelines. These steps are based on linux/macos terminal, please adapt to your terminal/os requirements.

steps for packaging lambda layer (run these steps in the root directory of the repo)

```
pip install sagemaker --target src/sgmkr-sdk-layer/python
cd src/sgmkr-sdk-layer/python
find . -type d -name "tests" -exec rm -rfv {} +
find . -type d -name "__pycache__" -exec rm -rfv {} +
```
## Deployment steps
* Checkout this repo in your local machine
* Create lambda layer code per the instructions in section ##Packaging code and dependencies for lambda
* Package and upload artifacts to S3 bucket with following command. This command will create a new cloudformation template(packaged-template.yaml) with updated path to lambda code artifact location in S3
```
  aws cloudformation package \
    --template combined-stack.yaml \
    --s3-bucket <s3 bucket name for uploading artifacts> \
    --output-template-file packaged-template.yaml
```
* deploy cloudformation template(using command below in the root directory) for creating required AWS resources for the solution. Update the parameter values per your requirements.
    ```
    aws cloudformation deploy \
      --template packaged-template.yaml \
      --stack-name sagemaker-automl-sample \
      --parameter-overrides EnvName=myautoml LambdaFunctionName=myautoml-pipeline-fn SourceCodeBucket=automl-sourcecode-bucket DataBucket=automl-data-bucket ModelBucket=automl-model-bucket TargetID=cbf06423-7b27-49f2-99cf-ac24393ef907 SNSARN=arn-of-sns-topic PipelineName=myautomlpipeline \
      --capabilities CAPABILITY_NAMED_IAM

    ```
* Invoke sagemaker pipeline by executing pipeline invocation lambda created by cloudformation deployment step

  ```
        aws lambda invoke \
          --function-name <pipeline invocation lambda function name from cloudformation deploy step> \
          --cli-binary-format raw-in-base64-out \
          --payload '{ "pipelinename": "myautomlpipeline","modelpackagename": "myautomlpackage","databucket": "s3://<your-data-bucket>/<your-training-data.csv>"}' \
          response.json
  ```
* Deploy model to sagemaker endpoint
    ```
      aws lambda invoke \
        --function-name <sagemaker endpoint deployment lambda function name from cloudformation deploy step> \
        --cli-binary-format raw-in-base64-out \
        --payload '{ "endpoint_name": "your-endpoint-name", "endpoint_instance_type": "ml.m5.xlarge", "model_name": ",model name created by the pipeline", "endpoint_config_name": "your-model-config-name" }' \
        response.json
    ```
* Test the model by invoking the sagemaker endpoint created in previous step
## Resources