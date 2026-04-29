import copy
from typing import Tuple

import torch

import common.utils.module as module_utils
from common.evaluation.evaluators.evaluator import VoidEvaluator
from common.train.trainer import Trainer


class LogLinearPolicyGradientTrainer(Trainer):

    def __init__(self, model, optimizer, expected_reinforce_updates: bool, report_grad_norm: bool = False, kl_regularization_coeff: float = 0,
                 load_kl_reg_ref_model_from_checkpoint: str = "", kl_reg_ref_unif: bool = False, use_forward_kl_reg: bool = False,
                 track_output_probs_thresh: int = 15, normalize_by_reward_std: bool = False, train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(),
                 callback=None, device=torch.device("cpu")):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.expected_reinforce_updates = expected_reinforce_updates
        self.report_grad_norm = report_grad_norm
        self.load_kl_reg_ref_model_from_checkpoint = load_kl_reg_ref_model_from_checkpoint
        self.kl_reg_ref_unif = kl_reg_ref_unif
        self.use_forward_kl_reg = use_forward_kl_reg
        self.normalize_by_reward_std = normalize_by_reward_std
        self.track_output_probs_thresh = track_output_probs_thresh

        self.kl_regularization_coeff = kl_regularization_coeff
        if self.kl_regularization_coeff != 0 and not self.kl_reg_ref_unif:
            self.init_model_copy = copy.deepcopy(model)

            if self.load_kl_reg_ref_model_from_checkpoint:
                checkpoint = torch.load(self.load_kl_reg_ref_model_from_checkpoint, map_location=device)
                self.init_model_copy.load_state_dict(checkpoint["model"])

            self.init_model_copy.requires_grad_(False)
            self.init_model_copy.to(device)

    def batch_update(self, batch_num, batch, total_num_batches):
        output_features, all_rewards, all_ground_truth_rewards = batch
        output_features = output_features.to(self.device)
        all_rewards = all_rewards.to(self.device)
        all_ground_truth_rewards = all_ground_truth_rewards.to(self.device)

        output = {}
        if self.report_grad_norm:
            output["params_norm"] = module_utils.compute_params_norm(self.model).item()

        y_probs = self.model(output_features)
        expected_rewards = (all_rewards * y_probs).sum(dim=1)
        expected_ground_truth_rewards = (all_ground_truth_rewards * y_probs).sum(dim=1)
        reward_var = self.__compute_reward_var(y_probs, all_rewards).item()

        if self.normalize_by_reward_std:
            reward_mean = (all_rewards * y_probs).sum(dim=1).mean()
            reward_std = torch.sqrt(((all_rewards - reward_mean) ** 2 * y_probs).sum(dim=1)).mean()
            all_rewards = all_rewards / (reward_std + 1e-8)

        regularized_objective_value, kl_reg, reward, _ = self.__compute_objective(output_features, y_probs, all_rewards)

        self.optimizer.zero_grad()
        (- regularized_objective_value).backward()
        self.optimizer.step()

        if self.report_grad_norm:
            grad_norm = module_utils.compute_grad_norm(self.model, detach=True).item()
            output["grad norm"] = grad_norm
            output["normalized grad norm"] = grad_norm / (output["params_norm"] + 1e-7)

        output.update({
            "reward": reward.item(),
            "reg objective value": regularized_objective_value.item(),
            "kl reg": kl_reg.item() if isinstance(kl_reg, torch.Tensor) else kl_reg,
            "reward var": reward_var,
            "expected reward": expected_rewards.mean().item(),
            "expected ground truth reward": expected_ground_truth_rewards.mean().item()
        })

        if output_features.shape[1] <= self.track_output_probs_thresh:
            for j in range(output_features.shape[0]):
                for i in range(output_features.shape[1]):
                    output[f"input_{j}_y_prob_{i}"] = y_probs[j][i].item()
                    output[f"input_{j}_ln_y_prob_{i}"] = torch.log(y_probs[j][i]).item()
                    output[f"input_{j}_expected_reward"] = expected_rewards[j].item()
                    output[f"input_{j}_expected_ground_truth_reward"] = expected_ground_truth_rewards[j].item()

        return output

    def __compute_objective(self, output_features, y_probs, all_rewards):
        if self.expected_reinforce_updates:
            objective_value = self.__expected_reinforce_objective(y_probs, all_rewards)
            reward = objective_value
            kl_reg = self.__expected_kl_reg(output_features, y_probs).mean() if self.kl_regularization_coeff != 0 else 0
        else:
            y_pred_samples = torch.multinomial(y_probs, 1).squeeze(dim=1)
            objective_value, rewards = self.__sample_based_reinforce_objective(all_rewards, y_probs, y_pred_samples)
            reward = rewards.mean()
            kl_reg = self.__sample_kl_reg(output_features, y_probs, y_pred_samples).mean() if self.kl_regularization_coeff != 0 else 0

        regularized_objective_value = objective_value - self.kl_regularization_coeff * kl_reg
        return regularized_objective_value, kl_reg, reward, all_rewards

    def __sample_based_reinforce_objective(self, all_rewards: torch.Tensor, y_probs: torch.Tensor,
                                           y_pred_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = all_rewards[torch.arange(all_rewards.shape[0]), y_pred_samples]
        y_sample_probs = y_probs[torch.arange(y_probs.shape[0]), y_pred_samples]
        return (rewards * torch.log(y_sample_probs)).mean(), rewards

    def __expected_reinforce_objective(self, y_probs: torch.Tensor, all_rewards: torch.Tensor) -> torch.Tensor:
        expected_reward = (all_rewards * y_probs).sum(dim=1)
        return expected_reward.mean()

    def __sample_kl_reg(self, output_features: torch.Tensor, y_probs: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
        y_sample_probs = y_probs[torch.arange(y_probs.shape[0]), y_samples]

        if self.use_forward_kl_reg:
            return self.__expected_kl_reg(output_features, y_probs).mean()

        reference_y_probs = self.init_model_copy(output_features) if not self.kl_reg_ref_unif \
            else torch.full_like(y_probs, fill_value=1 / y_probs.shape[1])
        reference_y_sample_probs = reference_y_probs[torch.arange(reference_y_probs.shape[0]), y_samples]

        return torch.log(y_sample_probs / reference_y_sample_probs)

    def __expected_kl_reg(self, output_features: torch.tensor, y_probs: torch.Tensor) -> torch.Tensor:
        reference_y_probs = self.init_model_copy(output_features) if not self.kl_reg_ref_unif \
            else torch.full_like(y_probs, fill_value=1 / y_probs.shape[1])

        if not self.use_forward_kl_reg:
            kl_div_entries = y_probs * torch.log(y_probs / reference_y_probs)
        else:
            kl_div_entries = - reference_y_probs * torch.log(y_probs / reference_y_probs)

        return kl_div_entries.sum(dim=1)

    def __compute_reward_var(self, y_probs: torch.Tensor, all_rewards: torch.Tensor) -> torch.Tensor:
        expected_rewards = (y_probs * all_rewards).sum(dim=1, keepdims=True)
        return (y_probs * torch.pow(all_rewards - expected_rewards, exponent=2)).sum(dim=1).mean()
