aws cloudformation deploy \
  --template packaged-template.yaml  \
  --stack-name sagemaker-automl-sample \
  --parameter-overrides EnvName=myautoml LambdaFunctionName=myautoml-pipeline-fn SourceCodeBucket=automl-sourcecode-bucket DataBucket=automl-data-bucket ModelBucket=automl-model-bucket TargetID=cbf06423-7b27-49f2-99cf-ac24393ef907 SNSARN=arn:aws:sns:us-east-1:703606194176:appian-sgmkr PipelineName=myautomlpipeline \
  --capabilities CAPABILITY_NAMED_IAM


  aws cloudformation package \
  --template /Users/daythaku/github/sagemaker-autopilot-sample-solution/new-stack.yaml \
  --s3-bucket smpp-demo-bucket-sourcecode-llm \
  --output-template-file packaged-template.yaml

    aws cloudformation package \
  --template <local path to stack> \
  --s3-bucket <s3 bucket name for uploading artifacts> \
  --output-template-file packaged-template.yaml

  aws lambda invoke \
    --function-name invokesmpipeline-automl-myautoml \
    --cli-binary-format raw-in-base64-out \
    --payload '{ "pipelinename": "myautomlpipeline","modelpackagename": "myautomlpackage","databucket": "s3://appian-ml-sgmaker/input/data.csv"}' \
    response.json