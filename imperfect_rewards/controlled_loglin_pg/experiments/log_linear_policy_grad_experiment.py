import json
import logging
from typing import Tuple

import torch
import torch.nn as nn

from common.data.modules import DataModule
from common.evaluation.evaluators import Evaluator, TrainEvaluator, TrainBatchOutputEvaluator, VoidEvaluator
from common.experiment import FitExperimentBase, ExperimentResult
from common.experiment.fit_experiment_base import ScoreInfo
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from controlled_loglin_pg.data.log_linear_features_datamodule import LogLinearFeaturesDataModule
from controlled_loglin_pg.model.log_linear_policy import LogLinearPolicy
from controlled_loglin_pg.model.log_linear_weights_solver import solve_weights_from_probabilities
from controlled_loglin_pg.train.log_linear_policy_grad_trainer import LogLinearPolicyGradientTrainer

INITIAL_POLICY_WEIGHTS_CONFIG_NAME = "initial_policy_weights"
INITIAL_POLICY_PROBABILITIES_CONFIG_NAME = "initial_policy_probabilities"
OUTPUT_FEATURES_CONFIG_NAME = "output_features"
REWARDS_CONFIG_NAME = "rewards"
GROUND_TRUTH_REWARDS_CONFIG_NAME = "ground_truth_rewards"


class LogLinearPolicyGradientExperiment(FitExperimentBase):

    @staticmethod
    def add_experiment_specific_args(p):
        FitExperimentBase.add_experiment_base_specific_args(p)

        p.add_argument("--num_outputs", type=int, default=10, help="Number of outputs")
        p.add_argument("--feature_dim", type=int, default=10, help="Dimension of feature embeddings")
        p.add_argument("--features_type", type=str, default="id", help="Supports 'rnd' and 'id'.")
        p.add_argument("--data_rnd_seed", type=int, default=-1, help="Random seed for generating the features and random rewards")
        p.add_argument("--load_model_from_checkpoint", type=str, default="", help="Loads the model from the given checkpoint path.")
        p.add_argument("--load_dataset_to_gpu", action="store_true", help="Stores all dataset on the main GPU (if GPU device is given)")
        p.add_argument("--init_type", type=str, default="gauss", help="Initialization type for weights. Supports 'zero' and 'gauss'.")
        p.add_argument("--init_std", type=float, default=0.1, help="Initialization standard deviation for Gaussian initialization")
        p.add_argument("--init_rnd_seed", type=int, default=-1, help="Random seed for initializing weights")
        p.add_argument("--reward_type", type=str, default="rnd",
                       help="Type of reward to use. Supports 'rnd'. "
                            "In 'rnd', for each output the reward is fixed to a uniformly sampled from value [-1, 1].")
        p.add_argument("--normalize_by_reward_std", action="store_true",
                       help="If enabled, rewards will be divided the standard deviation "
                            "(mean over the batch of reward standard deviation across inputs).")
        p.add_argument("--custom_setting_config_path", type=str, help="Loads initial policy, rewards, ground truth rewards, and output features from"
                                                                      " the given config file. If given, will override any prior configurations.")

        p.add_argument("--use_expected_reinforce_updates", action="store_true",
                       help="Use expected REINFORCE updates instead of sample-based REINFORCE updates")

        p.add_argument("--optimizer", type=str, default="sgd", help="optimizer to use. Supports: 'sgd' and 'adam'.")
        p.add_argument("--lr", type=float, default=0.01, help="Optimization learning rate")
        p.add_argument("--kl_regularization_coeff", type=float, default=0, help="KL regularization coefficient")
        p.add_argument("--kl_reg_ref_unif", action="store_true", help="Use uniform reference distribution for KL regularization")
        p.add_argument("--use_forward_kl_reg", action="store_true",
                       help="Use forward KL regularization instead of reverse. Can be used to do label-smoothing "
                            "regularization, which amounts to CE loss over uniform distribution")
        p.add_argument("--logits_temperature", type=float, default=1, help="Temperature applied to logits")
        p.add_argument("--track_output_probs_thresh", type=int, default=15,
                       help="Track output probabilities if number of outputs is at most this threshold")

    def initialize(self, config: dict, state: dict):
        super().initialize(config, state)
        if config["custom_setting_config_path"]:
            with open(config["custom_setting_config_path"]) as f:
                state["custom_setting_config"] = json.load(f)

    def create_datamodule(self, config: dict, state: dict, logger: logging.Logger) -> DataModule:
        load_dataset_to_device = state["device"] if config["load_dataset_to_gpu"] and len(state["gpu_ids"]) > 0 else None

        if config["data_rnd_seed"] >= 0:
            curr_rng_state = torch.random.get_rng_state()
            torch.random.manual_seed(config["data_rnd_seed"])

        output_features = self.__create_output_features(config, state)
        rewards = self.__create_rewards(config) if not "custom_setting_config" in state \
            else torch.tensor(state["custom_setting_config"][REWARDS_CONFIG_NAME]).float()
        ground_truth_rewards = rewards if not "custom_setting_config" in state \
            else torch.tensor(state["custom_setting_config"][GROUND_TRUTH_REWARDS_CONFIG_NAME]).float()
        datamodule = LogLinearFeaturesDataModule(output_features=output_features,
                                                 rewards=rewards,
                                                 ground_truth_rewards=ground_truth_rewards,
                                                 load_dataset_to_device=load_dataset_to_device)
        datamodule.setup()

        if config["data_rnd_seed"] >= 0:
            torch.random.set_rng_state(curr_rng_state)

        return datamodule

    def __create_output_features(self, config: dict, state: dict) -> torch.Tensor:
        if "custom_setting_config" in state:
            return torch.tensor(state["custom_setting_config"][OUTPUT_FEATURES_CONFIG_NAME]).float()
        elif config["features_type"] == "rnd":
            output_features = torch.randn(1, config["num_outputs"], config["feature_dim"]).float()
        elif config["features_type"] == "id":
            output_features = torch.eye(config["num_outputs"], config["feature_dim"]).unsqueeze(dim=0).float()
        else:
            raise ValueError(f"Unsupported features type: {config['features_type']}")

        return output_features

    def __create_rewards(self, config: dict) -> torch.Tensor:
        if config["reward_type"] == "rnd":
            rewards = (torch.rand(1, config["num_outputs"]) * 2 - 1).float()
        else:
            raise ValueError(f"Unsupported reward function type: {config['reward_type']}")

        return rewards

    def create_model(self, datamodule: DataModule, config: dict, state: dict, logger: logging.Logger) -> nn.Module:
        if config["init_rnd_seed"] >= 0:
            curr_rng_state = torch.random.get_rng_state()
            torch.random.manual_seed(config["init_rnd_seed"])

        model = self.__create_loglinear_policy(config, state)

        if config["init_rnd_seed"] >= 0:
            torch.random.set_rng_state(curr_rng_state)

        if config["load_model_from_checkpoint"]:
            checkpoint = torch.load(config["load_model_from_checkpoint"], map_location=state["device"])
            model.load_state_dict(checkpoint["model"])

        state["initial_weights"] = model.linear.weight.data.clone()
        model.to(device=state["device"])
        state["initial_probs"] = model(datamodule.output_features).detach().cpu()
        return model

    def __create_loglinear_policy(self, config: dict, state: dict):
        if "custom_setting_config" in state:
            custom_config = state["custom_setting_config"]
            if INITIAL_POLICY_PROBABILITIES_CONFIG_NAME in custom_config:
                output_features = torch.tensor(custom_config[OUTPUT_FEATURES_CONFIG_NAME]).float()
                desired_probs = torch.tensor(custom_config[INITIAL_POLICY_PROBABILITIES_CONFIG_NAME]).float()

                initial_weights = solve_weights_from_probabilities(
                    output_features=output_features,
                    desired_probs=desired_probs,
                    temperature=config["logits_temperature"]
                )

                policy = LogLinearPolicy(input_dim=initial_weights.shape[1], temperature=config["logits_temperature"])
                policy.linear.weight.data = initial_weights
            else:
                if INITIAL_POLICY_WEIGHTS_CONFIG_NAME not in custom_config:
                    raise ValueError("Config must have either initial_policy_weights or initial_policy_probabilities")

                initial_weights = torch.tensor(custom_config[INITIAL_POLICY_WEIGHTS_CONFIG_NAME]).float()
                policy = LogLinearPolicy(input_dim=initial_weights.shape[1], temperature=config["logits_temperature"])
                policy.linear.weight.data = initial_weights
        else:
            policy = LogLinearPolicy(input_dim=config["feature_dim"], temperature=config["logits_temperature"])
            if config["init_type"] == "zero":
                policy.linear.weight.data.fill_(0)
            elif config["init_type"] == "gauss":
                policy.linear.weight.data.normal_(std=config["init_std"])
            else:
                raise ValueError(f"Unsupported init_type: {config['init_type']}")

        return policy

    def create_train_and_validation_evaluators(self, model: nn.Module, datamodule: LogLinearFeaturesDataModule, device, config: dict, state: dict,
                                               logger: logging.Logger) -> Tuple[TrainEvaluator, Evaluator]:
        batch_output_metrics_to_track = ["reward", "reg objective value", "kl reg", "params_norm", "grad norm", "normalized grad norm",
                                         "reward var", "expected reward", "expected ground truth reward"]
        metrics_tags = ["reward", "reg objective value", "kl reg", "params_norm", "grad norm", "normalized grad norm",
                        "reward var", "expected reward", "expected ground truth reward"]
        if config["num_outputs"] <= config["track_output_probs_thresh"]:
            for j in range(datamodule.output_features.shape[0]):
                batch_output_metrics_to_track.append(f"input_{j}_expected_reward")
                batch_output_metrics_to_track.append(f"input_{j}_expected_ground_truth_reward")
                metrics_tags.append(f"expected reward")
                metrics_tags.append(f"expected ground truth reward")
                for i in range(datamodule.output_features.shape[1]):
                    batch_output_metrics_to_track.append(f"input_{j}_y_prob_{i}")
                    batch_output_metrics_to_track.append(f"input_{j}_ln_y_prob_{i}")
                    metrics_tags.append(f"input_{j}_y_prob")
                    metrics_tags.append(f"input_{j}_ln_y_prob")

        return TrainBatchOutputEvaluator(metric_names=batch_output_metrics_to_track, metric_tags=metrics_tags), VoidEvaluator()

    def get_default_score_info(self, config: dict, state: dict) -> ScoreInfo:
        return ScoreInfo(metric_name="reward", is_train_metric=True, largest=False, return_best_score=False)

    def create_additional_metadata_to_log(self, model: torch.nn.Module, datamodule: LogLinearFeaturesDataModule,
                                          config: dict, state: dict, logger: logging.Logger) -> dict:
        additional_metadata = super().create_additional_metadata_to_log(model, datamodule, config, state, logger)
        additional_metadata["num_train_samples"] = datamodule.output_features.shape[0]
        additional_metadata["num_outputs"] = datamodule.output_features.shape[1]
        additional_metadata["feature_dim"] = datamodule.output_features.shape[2]
        additional_metadata["rewards"] = datamodule.rewards.tolist()
        additional_metadata["ground_truth_rewards"] = datamodule.ground_truth_rewards.tolist()
        additional_metadata["initial_weights"] = state["initial_weights"].tolist()
        additional_metadata["initial_probs"] = state["initial_probs"].tolist()
        additional_metadata["output_features"] = datamodule.output_features.tolist()
        return additional_metadata

    def create_trainer(self, model: nn.Module, datamodule: LogLinearFeaturesDataModule, train_evaluator: TrainEvaluator, val_evaluator: Evaluator,
                       callback: Callback, device, config: dict, state: dict, logger: logging.Logger) -> Trainer:
        optimizer = self.__create_optimizer(model, config)
        return LogLinearPolicyGradientTrainer(model, optimizer,
                                              expected_reinforce_updates=config["use_expected_reinforce_updates"],
                                              report_grad_norm=True,
                                              kl_regularization_coeff=config["kl_regularization_coeff"],
                                              kl_reg_ref_unif=config["kl_reg_ref_unif"],
                                              use_forward_kl_reg=config["use_forward_kl_reg"],
                                              normalize_by_reward_std=config["normalize_by_reward_std"],
                                              track_output_probs_thresh=config["track_output_probs_thresh"],
                                              train_evaluator=train_evaluator, val_evaluator=val_evaluator,
                                              callback=callback, device=device)

    def __create_optimizer(self, model: nn.Module, config: dict):
        if config["optimizer"] == "adam":
            return torch.optim.Adam(model.parameters(), lr=config["lr"])
        elif config["optimizer"] == "sgd":
            return torch.optim.SGD(model.parameters(), lr=config["lr"])
        else:
            raise ValueError(f"Unsupported optimizer type: '{config['optimizer']}'")

    def on_experiment_end(self, model: nn.Module, datamodule: LogLinearFeaturesDataModule, trainer: Trainer, fit_output: FitOutput,
                          experiment_result: ExperimentResult, config: dict, state: dict, logger: logging.Logger):
        super().on_experiment_end(model, datamodule, trainer, fit_output, experiment_result, config, state, logger)
        experiment_result.summary["rewards"] = datamodule.rewards.tolist()
        experiment_result.summary["ground_truth_rewards"] = datamodule.ground_truth_rewards.tolist()
        experiment_result.summary["initial_weights"] = state["initial_weights"].tolist()
        experiment_result.summary["initial_probs"] = state["initial_probs"].tolist()
        experiment_result.summary["output_features"] = datamodule.output_features.tolist()
