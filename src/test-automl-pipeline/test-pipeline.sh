#! /bin/sh -f

if [[ -z $1 ]];
then
    echo "Usage: test-pipeline <Pipeline Name>"
    exit 1
fi

pipelinename=$1
timestamp=$(date "+%Y-%m-%d")

echo "Starting three simultaneous model building actions using pipeline $pipelinename"

echo "Building a binary classification model named BinaryModel."
python3 -m create-model --pipeline-name $pipelinename \
        --input-uri s3://sagemaker-example-files-prod-us-east-1/datasets/tabular/uci_bank_marketing/bank-additional-full.csv \
        --target-attribute-name y \
        --model-package-name BinaryModel

echo "Building a regression model named RegressionModel."
python3 -m create-model --pipeline-name $pipelinename \
        --input-uri s3://sagemaker-example-files-prod-us-east-1/datasets/tabular/uci_abalone/abalone-with-headers.csv \
        --target-attribute-name Rings \
        --model-package-name RegressionModel

echo "Building a multiclass classification model named MulticlassModel."
python3 -m create-model --pipeline-name $pipelinename \
        --input-uri s3://sagemaker-example-files-prod-us-east-1/datasets/tabular/online_retail/online_retail_II_20k.csv \
        --target-attribute-name Country \
        --model-package-name MulticlassModel

echo "You can monitor the status of these actions from the command line using the command:"
echo "aws sagemaker list-pipeline-executions --pipeline-name $pipelinename --created-after $timestamp"
aws sagemaker list-pipeline-executions --pipeline-name $pipelinename --created-after $timestamp
