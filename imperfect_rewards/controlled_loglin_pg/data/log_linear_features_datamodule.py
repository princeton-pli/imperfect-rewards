import torch

from common.data.loaders.fast_tensor_dataloader import FastTensorDataLoader
from common.data.modules.datamodule import DataModule


class LogLinearFeaturesDataModule(DataModule):

    def __init__(self, output_features: torch.Tensor, rewards: torch.Tensor, ground_truth_rewards: torch.Tensor = None, load_dataset_to_device=None):
        """
        @param output_features: tensor of shape (num_samples, num_outputs, num_features) of the features for each output
        @param rewards: tensor of shape (num_samples, num_outputs) of the rewards for each output
        @param ground_truth_rewards: tensor of shape (num_samples, num_outputs) of the ground truth rewards for each output
        """
        self.output_features = output_features
        self.rewards = rewards
        self.ground_truth_rewards = ground_truth_rewards if ground_truth_rewards is not None else rewards
        self.load_dataset_to_device = load_dataset_to_device

        if self.load_dataset_to_device is not None:
            self.output_features = self.output_features.to(load_dataset_to_device)
            self.rewards = self.rewards.to(load_dataset_to_device)
            self.ground_truth_rewards = self.ground_truth_rewards.to(load_dataset_to_device)

    def setup(self):
        pass

    def train_dataloader(self, shuffle: bool = True) -> FastTensorDataLoader:
        return FastTensorDataLoader(self.output_features, self.rewards, self.ground_truth_rewards,
                                    batch_size=self.output_features.shape[0], shuffle=shuffle)

    def val_dataloader(self) -> FastTensorDataLoader:
        return FastTensorDataLoader(self.output_features, self.rewards, self.ground_truth_rewards,
                                    batch_size=self.output_features.shape[0], shuffle=False)

    def test_dataloader(self) -> FastTensorDataLoader:
        return self.val_dataloader()
