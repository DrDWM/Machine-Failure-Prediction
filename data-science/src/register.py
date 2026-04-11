# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job using MLflow Client.
"""

import argparse
import mlflow
from mlflow.tracking import MlflowClient
import os 
import json

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name under which model will be registered')
    parser.add_argument('--sweep_job_id', type=str, required=True, help='The name/ID of the parent sweep job')
    parser.add_argument("--model_info_output_path", type=str, required=True, help="Path to write model info JSON")
    args = parser.parse_args()
    return args

def main(args):
    '''Finds the best run from the sweep and registers it'''
    client = MlflowClient()
    
    # 1. Search for all child runs of the sweep job
    # We use the sweep_job_id (passed from YAML) to filter for its children
    current_experiment_id = mlflow.active_run().info.experiment_id
    
    print(f"Searching for best run in sweep: {args.sweep_job_id}")
    runs = client.search_runs(
        experiment_ids=[current_experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{args.sweep_job_id}'",
        order_by=["metrics.Accuracy DESC"] # Ensure 'Accuracy' matches your train.py log_metric name
    )
    
    if not runs:
        raise Exception(f"No child runs found for sweep job ID: {args.sweep_job_id}. Check if the metric name 'Accuracy' is correct.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_accuracy = best_run.data.metrics.get("Accuracy")
    
    print(f"Found best run: {best_run_id} with Accuracy: {best_accuracy}")

    # 2. Register the model directly from the best run's artifact path
    # 'model_output' must match the output name defined in your train.yml
    model_uri = f"runs:/{best_run_id}/model_output"
    
    print(f"Registering model from URI: {model_uri}")
    mlflow_model = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    model_version = mlflow_model.version
    print(f"Successfully registered version: {model_version}")

    # 3. Write model info JSON for downstream pipeline steps
    os.makedirs(args.model_info_output_path, exist_ok=True)
    model_info = {"id": f"{args.model_name}:{model_version}"}
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    
    with open(output_path, "w") as of:
        json.dump(model_info, of)
    print(f"Model info written to {output_path}")

if __name__ == "__main__":
    # Start an MLflow run for the registration step itself
    with mlflow.start_run():
        args = parse_args()
        main(args)
