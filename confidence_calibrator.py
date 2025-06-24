# confidence_calibrator.py

import torch
from scipy.optimize import brentq
from scipy.stats import binom

class ConfidenceCalibrator:
    """
    Calculates a confidence threshold (lambda) for a classifier
    to ensure the error rate is below a target 'alpha' with high probability.
    """
    def __init__(self, cal_phats: torch.Tensor, cal_yhats: torch.Tensor, cal_labels: torch.Tensor, delta: float):
        """
        Args:
            cal_phats (torch.Tensor): The model's confidence scores (max probability) for each calibration sample.
            cal_yhats (torch.Tensor): The model's predicted labels for each calibration sample.
            cal_labels (torch.Tensor): The ground-truth labels for each calibration sample.
            delta (float): The statistical tolerance, a small value like 0.1.
        """
        self.cal_phats = cal_phats
        self.cal_yhats = cal_yhats
        self.cal_labels = cal_labels
        self.delta = delta

        # Candidate lambdas: A range of possible confidence thresholds from 0 to 1.
        # We filter out lambdas where too few data points are available for stable calculation.
        self.lambdas = torch.Tensor([
            lam for lam in torch.linspace(0, 1, 2000) if self.n_lambda(lam) >= 20
        ])

    def selective_risk(self, lam: float) -> float:
        """Calculate the empirical error rate (risk) for a given confidence threshold `lam`."""
        # Consider only predictions where confidence is >= lam
        selected_indices = self.cal_phats >= lam
        
        # If nothing is selected, risk is 0
        num_total = selected_indices.sum().item()
        if num_total == 0:
            return 0.0

        # Calculate the number of incorrect predictions in the selected set
        num_incorrect = (self.cal_yhats[selected_indices] != self.cal_labels[selected_indices]).sum().item()
        
        return num_incorrect / num_total

    def selective_risk_upper_bound(self, lam: float) -> float:
        """
        Calculates a high-probability upper bound on the true selective risk,
        using a binomial confidence interval.
        """
        n = self.n_lambda(lam)
        risk = self.selective_risk(lam)
        
        if n == 0:
            return 1.0 # Worst-case risk if no samples are selected
            
        # Defines the function for which we want to find the root.
        # This is based on the cumulative distribution function (CDF) of the binomial distribution.
        def _upper_bound_condition(r: float):
            return binom.cdf(int(risk * n), int(n), r) - self.delta
        
        try:
            # `brentq` is a robust root-finding algorithm.
            return brentq(_upper_bound_condition, 0, 0.9999)
        except ValueError:
            # If brentq fails (e.g., bounds have the same sign), return a conservative estimate.
            return 1.0

    def n_lambda(self, lam: float) -> int:
        """Counts how many calibration samples have a confidence score >= `lam`."""
        return (self.cal_phats >= lam).sum().item()

    def find_lambda(self, alpha: float) -> float:
        """
        Finds the optimal confidence threshold `lambda_hat`.

        This is the lowest confidence threshold we can set such that the
        upper-bound on our error rate is still less than or equal to the
        user-defined target error rate `alpha`.

        Args:
            alpha (float): The target maximum acceptable error rate (e.g., 0.1 for 10%).

        Returns:
            The calculated confidence threshold `lambda_hat`.
        """
        if len(self.lambdas) == 0:
            print("Warning: No valid lambda values found. Not enough calibration data. Returning 1.0.")
            return 1.0
        
        # Iterate from high confidence to low confidence.
        # We want the *smallest* lambda (most coverage) that satisfies the risk constraint.
        for lam in reversed(self.lambdas):
            # The upper bound on risk for the *next* lambda step.
            # We look ahead to ensure the chosen lambda is safe.
            risk_bound = self.selective_risk_upper_bound(lam)
            
            if risk_bound > alpha:
                # If the risk for this lambda is too high, we must choose a
                # slightly more conservative (higher) lambda.
                # The next value in the reversed list is the one that was safe.
                next_lam_idx = torch.where(self.lambdas == lam)[0].item() + 1
                if next_lam_idx < len(self.lambdas):
                    return self.lambdas[next_lam_idx].item()
                else:
                    # We've reached the most conservative lambda.
                    return self.lambdas[-1].item()

        # If even the smallest lambda is safe, we can use it.
        return self.lambdas[0].item()
