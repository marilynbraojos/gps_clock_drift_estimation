import argparse
import yaml
import os
import numpy as np
from dataset_utils import DatasetUtils
from mat_utils import MatUtils
from transformer_model import TransformerModel
from train import train_model
from test import evaluate_model
from performance_metrics import r2, mae

def main(config):
    # Step 1: Load and preprocess the dataset
    print("Loading and preprocessing the dataset...")
    dataset = create_dataset(config["data"]["file_path"], config["data"]["params"])
    train_data, val_data, test_data = dataset["train"], dataset["val"], dataset["test"]
    print("Dataset loaded successfully.")
    
    # Step 2: Initialize the model
    print("Initializing the Transformer model...")
    model = TransformerModel(
        input_dim=config["model"]["input_dim"],
        output_dim=config["model"]["output_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"]
    )
    print("Model initialized.")

    # Step 3: Train the model
    print("Starting training...")
    train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config["training"]
    )
    print("Training completed.")

    # Step 4: Test the model
    print("Evaluating on the test set...")
    predictions, actuals = test_model(model, test_data)
    print("Evaluation completed.")

    # Step 5: Calculate performance metrics
    print("Calculating performance metrics...")
    r2 = calculate_r2(predictions, actuals)
    mae = calculate_mae(predictions, actuals)
    rmse = calculate_rmse(predictions, actuals)
    print(f"Performance Metrics: R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    # Optional: Save results
    if config["output"]["save_results"]:
        np.save(os.path.join(config["output"]["results_dir"], "predictions.npy"), predictions)
        np.save(os.path.join(config["output"]["results_dir"], "actuals.npy"), actuals)
        print(f"Results saved to {config['output']['results_dir']}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a Transformer model.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the configuration file (environment.yaml)"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Call the main function
    main(config)
