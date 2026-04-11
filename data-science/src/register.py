# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

"""
import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args = parser.parse_args()
    print(f'Arguments: {args}')

    return args
"""

""" older version of main(args)
def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)

    # Load model
    model = mlflow.sklearn.load_model(args.model_path)

    # Log model using mlflow
    mlflow.sklearn.log_model(model, args.model_name)

    # Register logged model using mlflow
    run_id = mlflow.active_run().info.run_id
    model_uri = f'runs:/{run_id}/{args.model_name}'
    mlflow_model = mlflow.register_model(model_uri, args.model_name)
    model_version = mlflow_model.version

    # Write model info
    print("Writing JSON")
    model_info = {"id": f"{args.model_name}:{model_version}"}
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_path, "w") as of:
        json.dump(model_info, of)
"""

""" second attempt
def main(args):
    print(f"Registering model from path: {args.model_path}")

    # 1. Log the model folder from the local disk into the CURRENT run
    # This makes the model visible in the 'Outputs + logs' tab of the register_model job
    mlflow.sklearn.log_model(
        sk_model=mlflow.sklearn.load_model(args.model_path), 
        artifact_path=args.model_name
    )

    # 2. Register it using the URI of the model we just logged
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{args.model_name}"
    
    print(f"Registering model version from URI: {model_uri}")
    mlflow_model = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    
    model_version = mlflow_model.version
    print(f"Registered version: {model_version}")

    # 3. Write model info JSON
    os.makedirs(args.model_info_output_path, exist_ok=True)
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_path, "w") as of:
        json.dump({"id": f"{args.model_name}:{model_version}"}, of)


if __name__ == "__main__":
    
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
"""


import argparse
import mlflow
from mlflow.tracking import MlflowClient
import os 
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--sweep_job_id', type=str, required=True)
    parser.add_argument("--model_info_output_path", type=str, required=True)
    return parser.parse_args()

def main(args):
    client = MlflowClient()
    
    # 1. Find the best child run of the sweep
    # We search for runs where the parent is our sweep job
    runs = client.search_runs(
        experiment_ids=[mlflow.active_run().info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{args.sweep_job_id}'",
        order_by=["metrics.Accuracy DESC"] # Match your primary_metric from YAML
    )
    
    if not runs:
        raise Exception(f"No child runs found for sweep job: {args.sweep_job_id}")
    
    best_run_id = runs[0].info.run_id
    print(f"Best Run ID: {best_run_id} with Accuracy: {runs[0].data.metrics.get('Accuracy')}")

    # 2. Register the model using the 'runs:/' URI of the best run
    # This points MLflow directly to the existing artifacts in storage
    model_uri = f"runs:/{best_run_id}/model_output"
    print(f"Registering model from URI: {model_uri}")
    
    mlflow_model = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    model_version = mlflow_model.version

    # 3. Write model info JSON
    os.makedirs(args.model_info_output_path, exist_ok=True)
    with open(os.path.join(args.model_info_output_path, "model_info.json"), "w") as of:
        json.dump({"id": f"{args.model_name}:{model_version}"}, of)

if __name__ == "__main__":
    with mlflow.start_run():
        args = parse_args()
        lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)
        
    main(args)

    mlflow.end_run()
