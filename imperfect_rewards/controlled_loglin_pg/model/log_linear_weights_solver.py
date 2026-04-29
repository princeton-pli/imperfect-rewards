import numpy as np
import torch


def solve_weights_from_probabilities(
        output_features: torch.Tensor,
        desired_probs: torch.Tensor,
        temperature: float = 1.0,
        epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Solve for weights that produce desired probabilities using least squares.

    Args:
        output_features: shape (num_samples, num_outputs, feature_dim)
        desired_probs: shape (num_samples, num_outputs) with probabilities
        temperature: temperature parameter for logits
        epsilon: small value to avoid log(0)

    Returns:
        weights: shape (1, feature_dim)
    """
    num_samples, num_outputs, feature_dim = output_features.shape
    features_np = output_features.cpu().numpy()
    probs_np = desired_probs.cpu().numpy()

    log_probs_np = np.log(np.clip(probs_np, a_min=epsilon, a_max=1.0))

    # Build the linear system: features @ weights ≈ temperature * log(probs)
    A_list = []
    b_list = []

    for s in range(num_samples):
        for i in range(num_outputs):
            A_list.append(features_np[s, i, :])
            b_list.append(temperature * log_probs_np[s, i])

    A = np.array(A_list)  # (num_samples * num_outputs, feature_dim)
    b = np.array(b_list)  # (num_samples * num_outputs,)
    weights_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    weights = torch.from_numpy(weights_flat).float().unsqueeze(0)
    return weights
