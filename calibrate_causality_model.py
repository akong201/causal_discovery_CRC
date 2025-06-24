# calibrate_causality_model.py

import json
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
import torch
from tqdm import tqdm

from model.causality_model import CausalityModel
from confidence_calibrator import ConfidenceCalibrator

def parse_args():
    """Parses command-line arguments."""
    parser = ArgumentParser(description="Calibrate a causality detection model to find a confidence threshold (lambda).")

    parser.add_argument("--model_name", type=str, default="gpt-4.1-nano", help="OpenAI model name (e.g., gpt-3.5-turbo, gpt-4-turbo).")
    parser.add_argument("--calibration_file", type=str, default="./data/causal_data.jsonl", help="Path to the calibration data file.")
    parser.add_argument("--fewshot_file", type=str, default="./data/causal_fewshot.jsonl", help="Path to the few-shot examples file.")
    parser.add_argument("--results_file", type=str, default="./results/causality_calibration_results.jsonl", help="Path to save the model's predictions on the calibration set.")
    
    # Parameters for risk control
    parser.add_argument("--alpha", type=float, default=0.1, help="The desired maximum risk (error rate).")
    parser.add_argument("--delta", type=float, default=0.1, help="The statistical tolerance for the confidence bound.")
    
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Map friendly names to actual OpenAI model identifiers
    if args.model_name == "gpt-4-turbo":
        args.openai_model_name = "gpt-4-turbo-2024-04-09"
    elif args.model_name == "gpt-3.5-turbo":
        args.openai_model_name = "gpt-3.5-turbo-0125"
    else:
        args.openai_model_name = args.model_name

    return args

def get_model_predictions(model: CausalityModel, calibration_samples: list, fewshot_samples: list, results_file: str):
    """
    Runs the model on the calibration set to get predictions and probabilities.
    Saves results to a file to avoid re-running on subsequent executions.
    """
    processed_samples = []
    print(f"Generating predictions for {len(calibration_samples)} calibration samples...")
    with jsonlines.open(results_file, mode='w') as writer:
        for sample in tqdm(calibration_samples):
            prediction, probabilities = model.evaluate_causality(sample, fewshot_samples)
            if prediction is not None and probabilities is not None:
                # Add model outputs to the original sample
                sample['prediction'] = prediction
                sample['probabilities'] = probabilities
                processed_samples.append(sample)
                writer.write(sample)
    return processed_samples

def main():
    """Main execution function."""
    args = parse_args()
    print(f"Starting calibration with target risk (alpha)={args.alpha} and delta={args.delta}")

    # --- 1. Load Data ---
    with open(args.calibration_file, 'r') as f:
        calibration_samples = json.load(f)
    with open(args.fewshot_file, 'r') as f:
        fewshot_samples = json.load(f)

    # --- 2. Get Model Predictions ---
    # Check if results already exist
    results_path = Path(args.results_file)
    if results_path.exists():
        print(f"Loading existing predictions from {results_path}...")
        with jsonlines.open(results_path) as reader:
            processed_samples = list(reader)
    else:
        print("No existing predictions found. Running model...")
        model = CausalityModel(model_name=args.openai_model_name)
        processed_samples = get_model_predictions(model, calibration_samples, fewshot_samples, args.results_file)

    if not processed_samples:
        print("No samples were processed successfully. Exiting.")
        return

    # --- 3. Prepare Data for Calibration ---
    labels, yhats, phats = [], [], []
    for sample in processed_samples:
        # Ground truth (True -> 1, False -> 0)
        labels.append(1 if sample['is_causal'] else 0)
        # Prediction (True -> 1, False -> 0)
        yhats.append(1 if sample['prediction'] == 'True' else 0)
        # Confidence (max probability)
        phats.append(max(sample['probabilities'].values()))

    # Convert to PyTorch tensors
    labels_tensor = torch.tensor(labels, dtype=torch.int)
    yhats_tensor = torch.tensor(yhats, dtype=torch.int)
    phats_tensor = torch.tensor(phats, dtype=torch.float)

    print(f"\nPrepared {len(labels)} samples for calibration.")

    # --- 4. Perform Calibration ---
    calibrator = ConfidenceCalibrator(
        cal_phats=phats_tensor,
        cal_yhats=yhats_tensor,
        cal_labels=labels_tensor,
        delta=args.delta
    )
    
    lambda_hat = calibrator.find_lambda(alpha=args.alpha)
    
    print("\n" + "="*50)
    print("CALIBRATION COMPLETE")
    print(f"Target Risk (alpha): {args.alpha}")
    print(f"Chosen Confidence Threshold (lambda): {lambda_hat:.4f}")
    print("="*50)
    print(f"\nThis means the model should only make a prediction if its confidence is >= {lambda_hat:.4f} to likely meet the target error rate of {args.alpha*100}%.")

if __name__ == "__main__":
    main()

