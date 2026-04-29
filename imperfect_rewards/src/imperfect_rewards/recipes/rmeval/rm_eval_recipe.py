import gc
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import DataLoader

import imperfect_rewards.utils.single_process_logging as logging_utils
from imperfect_rewards.data.utils import CausalLMWrapper, compute_length_normalized_log_probs
from imperfect_rewards.data.utils.rm_utils import RewardModelWrapper, get_reward_model_wrapper
from imperfect_rewards.recipes import BaseRecipe
from imperfect_rewards.recipes.rmeval.dataloaders import PromptDataset, PromptCompletionDataset
from imperfect_rewards.utils import DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PADDING_TOKEN, update_tokenizer, \
    update_model_num_embeddings_and_special_tokens
from imperfect_rewards.utils.strings import (
    PER_PROMPT_RESPONSES,
    PER_PROMPT_REWARDS,
    OFFLINE_RESPONSES_NAME,
    ONLINE_RESPONSES_NAME,
    DATASET_KEY_NAME,
    SPLIT_KEY_NAME,
    GOLD_RM_KEY_NAME,
    LM_KEY_NAME,
    RM_KEY_NAME,
    PROMPTS,
    PROMPT_INDICES_KEY_NAME,
    GENERATIONS_TYPE_KEY_NAME,
    LM_NAME_FOR_PROBS_KEY_NAME,
    PER_PROMPT_LOGPROBS,
    PER_PROMPT_SEQUENCE_LENGTHS,
    IS_ONE_VS_MANY_KEY_NAME
)
from imperfect_rewards.metrics.accuracy import (
    acc,
    acc_w,
    hacc_w,
    one_vs_many_acc,
    one_vs_many_acc_w,
    one_vs_many_hacc_w,
)


@dataclass
class ProbabilityMetrics:
    """Container for probability metric of the current policy + sequence lengths and the ref policy + sequence lengths."""
    log_probs: Optional[torch.Tensor] = None
    sequence_lengths: Optional[torch.Tensor] = None


@dataclass
class EvaluationMetadata:
    """Container for evaluation metadata used across multiple functions."""
    lm_name: str
    lm_name_for_probs: str
    type_of_generations: str
    rm_name: str
    split: str
    prompt_indices: torch.Tensor
    dataset_name: str = ""
    lm_file_name: str = ""

    def __post_init__(self):
        """Set file names to match names if not provided."""
        if not self.lm_file_name:
            self.lm_file_name = self.lm_name


@dataclass
class RewardData:
    """Container for reward tensors and their normalization parameters."""
    rm_rewards: torch.Tensor  # Proxy/evaluated RM rewards
    gold_rewards: Optional[torch.Tensor] = None  # Ground truth RM rewards


@dataclass
class SplitMetrics:
    """Container for all metrics for a single split (train or test)."""
    dataset: Optional[Dataset] = None
    dataset_name: str = ""
    prompt_indices: Optional[torch.Tensor] = None
    split_name: str = ""
    offline_gold_rewards: Optional[torch.Tensor] = None
    online_gold_rewards: Optional[torch.Tensor] = None
    offline_log_probs: Optional[ProbabilityMetrics] = None
    online_log_probs: Optional[ProbabilityMetrics] = None


class RMEvalRecipe(BaseRecipe):

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.lm_name = self.__extract_name_from_path(self.config.rm_eval_config.language_model_path)
        self.lm_file_name = self.__extract_file_name_from_path(self.config.rm_eval_config.language_model_path)
        self.gold_rm_name = self.__extract_name_from_path(self.config.rm_eval_config.ground_truth_reward_model_path)
        self.lm_generation_train_path = os.path.join(self.config.rm_eval_config.output_dir,
                                                     f"{PER_PROMPT_RESPONSES}_{self.lm_name}_train.pt")
        self.prompt_train_path = os.path.join(self.config.rm_eval_config.output_dir,
                                              f"{PROMPTS}_{self.lm_name}_train.pt")
        self.lm_generation_test_path = os.path.join(self.config.rm_eval_config.output_dir,
                                                    f"{PER_PROMPT_RESPONSES}_{self.lm_name}_test.pt")
        self.prompt_test_path = os.path.join(self.config.rm_eval_config.output_dir,
                                             f"{PROMPTS}_{self.lm_name}_test.pt")
        self.logger = logging_utils.create_logger(console_logging=True, file_logging=True, log_dir=self.config.rm_eval_config.output_dir,
                                                  log_file_name_prefix="rm_eval")

        os.makedirs(self.config.rm_eval_config.output_dir, exist_ok=True)

    def __shorten_ds_name(self, dataset_name: str):
        # If too long, extract rid from dataset name
        if len(dataset_name) > 20:
            m = re.search(r"rid_\d+", dataset_name)
            rid = m.group(0) if m else "0"
        return "custom_dataset_" + rid if len(dataset_name) > 20 else dataset_name

    def __extract_name_from_path(self, path: str):
        return "_".join(path.split("/")[-3:]).replace(".", "-")

    def __extract_file_name_from_path(self, path: str):
        """The name used in file paths is shorter to avoid file names exceeding allowed length."""
        return "_".join(path.split("/")[-2:]).replace(".", "-")

    @staticmethod
    def __has_multiple_rejected_responses(rejected):
        """Check if rejected is a list of conversations (multiple rejected) vs a single conversation."""
        if not isinstance(rejected, list) or len(rejected) == 0:
            return False
        # If the first element is a list, we have multiple rejected responses
        # If the first element is a dict, we have a single rejected response
        return isinstance(rejected[0], list)

    def __dataset_has_multiple_rejected(self, train_dataset, test_dataset) -> bool:
        if train_dataset is not None and len(train_dataset) > 0:
            first_example = train_dataset[0]
            return self.__has_multiple_rejected_responses(first_example.get('rejected', []))
        elif test_dataset is not None and len(test_dataset) > 0:
            first_example = test_dataset[0]
            return self.__has_multiple_rejected_responses(first_example.get('rejected', []))

        return False

    def __create_dict_for_saving(self, metadata: EvaluationMetadata, values, values_key: str):
        if isinstance(values, torch.Tensor):
            values = values.cpu()

        return {
            DATASET_KEY_NAME: metadata.dataset_name,  # self.config.rm_eval_config.dataset_path,
            SPLIT_KEY_NAME: metadata.split,
            GOLD_RM_KEY_NAME: self.gold_rm_name,
            LM_KEY_NAME: metadata.lm_name,
            LM_NAME_FOR_PROBS_KEY_NAME: metadata.lm_name_for_probs,
            GENERATIONS_TYPE_KEY_NAME: metadata.type_of_generations,
            RM_KEY_NAME: metadata.rm_name,
            PROMPT_INDICES_KEY_NAME: metadata.prompt_indices,
            values_key: values
        }

    def __get_lm_wrapper(self, lm_str: str, device, config):
        try:
            lm_wrapper = CausalLMWrapper(lm_str, device=device, config=config)
            lm_wrapper.prepare()

            # If the language model does not have a chat template, adds a default one for debugging purposes (chat template needs to already exist)
            if not lm_wrapper.tokenizer.chat_template:
                self.logger.warning("Language model does not have a chat template. "
                                    "Adding a default one, which should only be used for debugging purposes.")
                update_tokenizer(tokenizer=lm_wrapper.tokenizer, num_added_toks=0, pad_token=DEFAULT_PADDING_TOKEN,
                                 eos_token=DEFAULT_EOS_TOKEN, logger=self.logger, user_token=DEFAULT_USER_TOKEN,
                                 assistant_token=DEFAULT_ASSISTANT_TOKEN)
                lm_wrapper.lm.config.pad_token_id = lm_wrapper.tokenizer.pad_token_id
                update_model_num_embeddings_and_special_tokens(lm_wrapper.lm, lm_wrapper.tokenizer)
            elif lm_wrapper.tokenizer.pad_token_id is None:
                lm_wrapper.tokenizer.pad_token_id = lm_wrapper.tokenizer.eos_token_id
                lm_wrapper.lm.config.pad_token_id = lm_wrapper.tokenizer.pad_token_id

            return lm_wrapper
        except Exception:
            self.logger.exception("Exception while trying to load language model.")
            raise

    def __save_log_probs(self, per_prompt_log_probs: torch.Tensor, per_prompt_sequence_lengths: torch.Tensor,
                         metadata: EvaluationMetadata):
        shortened_ds_name = self.__shorten_ds_name(metadata.dataset_name)
        probs_file_name = (f"{PER_PROMPT_LOGPROBS}_lm={metadata.lm_file_name}_rm={metadata.rm_name}_"
                           f"type_gen={metadata.type_of_generations}_{shortened_ds_name}_{metadata.split}.pt")
        seq_length_file_name = (f"{PER_PROMPT_SEQUENCE_LENGTHS}_lm={metadata.lm_file_name}_rm={metadata.rm_name}_"
                                f"type_gen={metadata.type_of_generations}_{shortened_ds_name}_{metadata.split}.pt")
        per_prompt_statistics_path = os.path.join(self.config.rm_eval_config.output_dir, probs_file_name)
        per_prompt_sequence_length_path = os.path.join(self.config.rm_eval_config.output_dir, seq_length_file_name)

        self.logger.info("=" * 180)
        self.logger.info(
            f"Saving per prompt LOG PROBS at {per_prompt_statistics_path}:\n"
            f"Saving per prompt SEQUENCE LENGTHS at {per_prompt_sequence_length_path}:\n"
            f"DATASET: {metadata.dataset_name} , "
            f"SPLIT: {metadata.split} , "
            f"LANGUAGE MODEL: {metadata.lm_name} , "
            f"TYPE OF GENERATIONS: {metadata.type_of_generations} , "
            f"LANGUAGE MODEL FOR LOG PROBS: {metadata.lm_name_for_probs} , "
            f"REWARD MODEL: {metadata.rm_name} , "
            f"NUM PROMPTS: {per_prompt_log_probs.shape[0]} , "
            f"NUM RESPONSES PER PROMPT: {per_prompt_log_probs.shape[1]}"
        )
        self.logger.info("=" * 180)
        self.logger.info(f"Mean log prob: {per_prompt_log_probs.mean():.6f}")
        length_norm_log_probs = compute_length_normalized_log_probs(per_prompt_log_probs, per_prompt_sequence_lengths)
        self.logger.info(f"Mean length-normalized log prob: {length_norm_log_probs.mean():.6f}")
        self.logger.info("=" * 180)

        torch.save(self.__create_dict_for_saving(metadata=metadata, values=per_prompt_log_probs, values_key=PER_PROMPT_LOGPROBS),
                   per_prompt_statistics_path)
        torch.save(self.__create_dict_for_saving(metadata=metadata, values=per_prompt_sequence_lengths, values_key=PER_PROMPT_SEQUENCE_LENGTHS),
                   per_prompt_sequence_length_path)

    def __save_log_probs_for_train_and_test(self, dataset_path: str, lm_name: str, type_of_generations: str, train_logprobs=None,
                                            train_response_lengths=None, train_prompt_indices=None, train_split: str = None, test_logprobs=None,
                                            test_response_lengths=None, test_prompt_indices=None, test_split: str = None):
        if train_logprobs is not None:
            train_offline_log_probs_metadata = EvaluationMetadata(
                lm_name=lm_name,
                lm_name_for_probs=self.lm_name,
                type_of_generations=type_of_generations,
                rm_name="",
                dataset_name=dataset_path,
                split=train_split,
                prompt_indices=train_prompt_indices
            )
            self.__save_log_probs(
                per_prompt_log_probs=train_logprobs,
                per_prompt_sequence_lengths=train_response_lengths,
                metadata=train_offline_log_probs_metadata
            )

        if test_logprobs is not None:
            test_offline_log_probs_metadata = EvaluationMetadata(
                lm_name=lm_name,
                lm_name_for_probs=self.lm_name,
                type_of_generations=type_of_generations,
                rm_name="",
                dataset_name=dataset_path,
                split=test_split,
                prompt_indices=test_prompt_indices
            )
            self.__save_log_probs(
                per_prompt_log_probs=test_logprobs,
                per_prompt_sequence_lengths=test_response_lengths,
                metadata=test_offline_log_probs_metadata
            )

    def __save_rewards_and_log_reward_metrics(self, per_prompt_rewards: torch.Tensor, metadata: EvaluationMetadata,
                                              is_one_vs_many_dataset: bool = False):
        shortened_ds_name = self.__shorten_ds_name(metadata.dataset_name)
        rewards_file_name = f"{PER_PROMPT_REWARDS}_lm={metadata.lm_file_name}_rm={metadata.rm_name}_type_gen={metadata.type_of_generations}_{shortened_ds_name}_{metadata.split}.pt"
        per_prompt_statistics_path = os.path.join(self.config.rm_eval_config.output_dir, rewards_file_name)

        overall_reward_mean = per_prompt_rewards.mean()
        overall_reward_std = per_prompt_rewards.std()
        per_prompt_whitened_rewards = (per_prompt_rewards - overall_reward_mean) / overall_reward_std

        self.logger.info("=" * 180)
        self.logger.info(
            f"Saving per prompt rewards at {per_prompt_statistics_path}:\n"
            f"DATASET: {metadata.dataset_name} , "
            f"SPLIT: {metadata.split} , "
            f"LANGUAGE MODEL: {metadata.lm_name} , "
            f"TYPE OF GENERATIONS: {metadata.type_of_generations} , "
            f"LANGUAGE MODEL FOR PROBS: {metadata.lm_name_for_probs} , "
            f"REWARD MODEL: {metadata.rm_name} , "
            f"NUM PROMPTS: {per_prompt_rewards.shape[0]} , "
            f"NUM RESPONSES PER PROMPT: {per_prompt_rewards.shape[1]}"
        )
        dict_for_save = self.__create_dict_for_saving(metadata=metadata, values=per_prompt_rewards, values_key=PER_PROMPT_REWARDS)
        if is_one_vs_many_dataset:
            dict_for_save[IS_ONE_VS_MANY_KEY_NAME] = True

        torch.save(dict_for_save, per_prompt_statistics_path)

        # Calculate mean, std and var of rewards from the aggregated
        self.logger.info("-" * 180)
        self.logger.info(f"Per prompt reward stats aggregated over the dataset:")
        self.__log_stats_for_quantity(per_prompt_rewards.mean(dim=1), "Reward Mean")
        self.__log_stats_for_quantity(per_prompt_rewards.var(dim=1), "Reward Variance")
        self.__log_stats_for_quantity(per_prompt_whitened_rewards.mean(dim=1), "Whitened Reward Mean")
        self.__log_stats_for_quantity(per_prompt_whitened_rewards.var(dim=1), "Whitened Reward Variance")
        self.logger.info("=" * 180)

    def __save_rewards_and_log_reward_metrics_one_vs_many(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor,
                                                          metadata: EvaluationMetadata):
        """
        Save rewards for one_vs_many format where chosen is first column and rejected are subsequent columns.
        Adds is_one_vs_many_dataset=True to the saved dictionary.
        """
        # Combine chosen and rejected: first column is chosen, rest are rejected
        per_prompt_rewards = torch.cat([chosen_rewards.unsqueeze(1), rejected_rewards], dim=1)  # shape: (num_prompts, num_rejected+1)

        shortened_ds_name = self.__shorten_ds_name(metadata.dataset_name)
        rewards_file_name = f"{PER_PROMPT_REWARDS}_lm={metadata.lm_file_name}_rm={metadata.rm_name}_type_gen={metadata.type_of_generations}_{shortened_ds_name}_{metadata.split}.pt"
        per_prompt_rewards_path = os.path.join(self.config.rm_eval_config.output_dir, rewards_file_name)

        overall_reward_mean = per_prompt_rewards.mean()
        overall_reward_std = per_prompt_rewards.std()
        per_prompt_whitened_rewards = (per_prompt_rewards - overall_reward_mean) / overall_reward_std

        self.logger.info("=" * 180)
        self.logger.info(
            f"Saving per prompt rewards at {per_prompt_rewards_path}:\n"
            f"DATASET: {metadata.dataset_name} , "
            f"SPLIT: {metadata.split} , "
            f"LANGUAGE MODEL: {metadata.lm_name} , "
            f"TYPE OF GENERATIONS: {metadata.type_of_generations} , "
            f"LANGUAGE MODEL FOR PROBS: {metadata.lm_name_for_probs} , "
            f"REWARD MODEL: {metadata.rm_name} , "
            f"NUM PROMPTS: {per_prompt_rewards.shape[0]} , "
            f"NUM RESPONSES PER PROMPT: {per_prompt_rewards.shape[1]} (ONE-VS-MANY)"
        )

        # Create dict with is_one_vs_many_dataset field
        save_dict = self.__create_dict_for_saving(metadata=metadata, values=per_prompt_rewards, values_key=PER_PROMPT_REWARDS)
        save_dict[IS_ONE_VS_MANY_KEY_NAME] = True

        torch.save(save_dict, per_prompt_rewards_path)

        # Calculate mean, std and var of rewards from the aggregated
        self.logger.info("-" * 180)
        self.logger.info(f"Per prompt reward stats aggregated over the dataset:")
        self.__log_stats_for_quantity(per_prompt_rewards.mean(dim=1), "Reward Mean")
        self.logger.info(f"Chosen reward mean: {chosen_rewards.mean()}")
        # For rejected, compute mean across all rejected rewards (excluding -inf padding)
        rejected_rewards_masked = rejected_rewards.clone()
        rejected_rewards_masked[rejected_rewards == float('-inf')] = float('nan')
        rejected_mean = torch.nanmean(rejected_rewards_masked)
        self.logger.info(f"Rejected reward mean: {rejected_mean}")
        self.__log_stats_for_quantity(per_prompt_rewards.var(dim=1), "Reward Variance")
        self.__log_stats_for_quantity(per_prompt_whitened_rewards.mean(dim=1), "Whitened Reward Mean")
        self.__log_stats_for_quantity(per_prompt_whitened_rewards.var(dim=1), "Whitened Reward Variance")
        self.logger.info("=" * 180)

    def __log_stats_for_quantity(self, values: torch.Tensor, quantity_name: str):
        self.logger.info(
            f"{quantity_name}: "
            f"mean {values.mean()} , "
            f"min {values.min()} , "
            f"25th percentile {torch.quantile(values, q=0.25)} , "
            f"median {values.median()} , "
            f"75th percentile {torch.quantile(values, q=0.75)} , "
            f"max {values.max()}"
        )

    def __compute_num_batches(self, dataset_len: int, batch_size: int):
        num_batches = dataset_len // batch_size
        if dataset_len % batch_size != 0:
            num_batches += 1

        return num_batches

    def __create_dataset_with_generated_responses(self,
                                                  lm_wrapper: CausalLMWrapper,
                                                  responses_out_file: str,
                                                  prompts_out_file: str,
                                                  dataset,
                                                  dataset_name: str,
                                                  split: str,
                                                  prompt_indices: torch.Tensor) -> Tuple[Dataset, Dataset]:
        tokenizer = lm_wrapper.tokenizer
        num_return_sequences = self.config.rm_eval_config.generation_config["num_return_sequences"]

        buffer = []
        buffer_probs = []

        prompts_dataset = PromptDataset(dataset)
        prompts_dataloader = DataLoader(
            prompts_dataset,
            batch_size=self.config.rm_eval_config.lm_batch_size,
            shuffle=False,
            collate_fn=lambda batch: {
                "prompts": [item["prompt"] for item in batch],
                "prompt_ids": [item["prompt_id"] for item in batch]
            }
        )

        num_batches = self.__compute_num_batches(len(prompts_dataset), self.config.rm_eval_config.lm_batch_size)
        # Generate responses for each prompt
        for i, batch in enumerate(prompts_dataloader):
            if i % 10 == 0:
                self.logger.info(f"Generating responses for batch {i + 1} / {num_batches}")

            batch_prompts = batch["prompts"]
            batch_prompts_ids = batch["prompt_ids"]

            batch_prompts_chat_templated = tokenizer.apply_chat_template(
                [
                    [{"content": prompt, "role": "user"}] for prompt in batch_prompts
                ],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False
            )
            inputs = tokenizer(batch_prompts_chat_templated, return_tensors="pt", padding=True,
                               truncation=False, add_special_tokens=False).to(self.device)

            # Generate multiple completions per prompt
            decoded_generations, log_probs, response_lengths = lm_wrapper.compute_batch_generation(inputs)

            # Group outputs into structured format
            for k, (prompt, prompt_id) in enumerate(zip(batch_prompts, batch_prompts_ids)):
                prompt_generations = [
                    {
                        "prompt": prompt,
                        "prompt_id": prompt_id,
                        "response": decoded_generations[j]
                    } for j in range(k * num_return_sequences, (k + 1) * num_return_sequences)
                ]
                buffer.extend(prompt_generations)

                prompt_generations_probs = [
                    {
                        "prompt_id": prompt_id,
                        "response": decoded_generations[j],
                        "response_log_probs": log_probs[j],
                        "response_sequence_length": response_lengths[j].item()
                    } for j in range(k * num_return_sequences, (k + 1) * num_return_sequences)
                ]
                buffer_probs.extend(prompt_generations_probs)

        self.__log_and_save_prompts_and_generated_responses(buffer, dataset_name, prompt_indices, prompts_out_file, responses_out_file, split)
        return Dataset.from_list(buffer), Dataset.from_list(buffer_probs)

    def __log_and_save_prompts_and_generated_responses(self, buffer: list, dataset_name: str, prompt_indices: torch.Tensor,
                                                       prompts_out_file: str, responses_out_file: str, split: str, num_examples_to_log: int = 5):
        prompts = [buffer[0]["prompt"]]
        per_prompt_responses = []
        curr_prompt_id = buffer[0]["prompt_id"]
        curr_prompt_responses = []
        for example in buffer:
            if example["prompt_id"] != curr_prompt_id:
                per_prompt_responses.append(curr_prompt_responses)
                curr_prompt_id = example["prompt_id"]
                curr_prompt_responses = []
                prompts.append(example["prompt"])
            curr_prompt_responses.append(example["response"])
        per_prompt_responses.append(curr_prompt_responses)

        responses_metadata = EvaluationMetadata(
            lm_name=self.lm_name,
            lm_name_for_probs=self.lm_name,
            type_of_generations=OFFLINE_RESPONSES_NAME,
            rm_name="",
            dataset_name=dataset_name,
            split=split,
            prompt_indices=prompt_indices
        )

        prompts_metadata = EvaluationMetadata(
            lm_name="",
            lm_name_for_probs="",
            type_of_generations="",
            rm_name="",
            dataset_name=dataset_name,
            split=split,
            prompt_indices=prompt_indices
        )

        # Logs random examples of prompts and generated responses
        num_to_sample = min(num_examples_to_log, len(prompts))
        sampled_indices = np.random.choice(len(prompts), size=num_to_sample, replace=False)

        self.logger.info("=" * 180)
        self.logger.info(f"Logging {num_to_sample} randomly sampled prompt-response examples:")
        self.logger.info("=" * 180)

        for i, prompt_idx in enumerate(sampled_indices):
            prompt = prompts[prompt_idx]
            responses_for_prompt = per_prompt_responses[prompt_idx]

            # Randomly select 1 response from this prompt's responses
            if len(responses_for_prompt) > 0:
                selected_response_idx = np.random.choice(len(responses_for_prompt))
                selected_response = responses_for_prompt[selected_response_idx]

                self.logger.info(f"\nExample {i} (Prompt Index: {prompt_idx}):")
                self.logger.info(f"Prompt: {prompt}")
                self.logger.info(f"Response: {selected_response}")
                self.logger.info("-" * 180)

            self.logger.info("=" * 180)

        if self.config.rm_eval_config.save_generated_responses:
            self.logger.info("=" * 180)
            self.logger.info(f"Saving prompts at {prompts_out_file}")
            torch.save(self.__create_dict_for_saving(metadata=prompts_metadata, values=prompts, values_key=PROMPTS), prompts_out_file)

            self.logger.info(f"Saving per prompt responses using {self.lm_name} at {responses_out_file}")
            torch.save(self.__create_dict_for_saving(metadata=responses_metadata, values=per_prompt_responses, values_key=PER_PROMPT_RESPONSES),
                       responses_out_file)
            self.logger.info("=" * 180)

    def __compute_per_prompt_rewards(self, rm_wrapper: RewardModelWrapper, dataset):
        tokenizer = rm_wrapper.tokenizer

        data_loader = DataLoader(
            dataset,
            batch_size=self.config.rm_eval_config.rm_batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )

        num_batches = self.__compute_num_batches(len(dataset), self.config.rm_eval_config.rm_batch_size)

        rewards = []
        for i, batch in enumerate(data_loader):
            if i % 10 == 0:
                self.logger.info(f"Computing rewards for batch {i + 1} / {num_batches}")

            prompts = [entry["prompt"] for entry in batch]
            responses = [entry["response"] for entry in batch]
            formatted_inputs = tokenizer.apply_chat_template(
                [
                    [{"content": prompt, "role": "user"}, {"content": response, "role": "assistant"}]
                    for prompt, response in zip(prompts, responses)
                ],
                tokenize=False,
                enable_thinking=False
            )
            formatted_inputs = [text for text in formatted_inputs]

            tokenized_inputs = tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False
            ).to(self.device)

            batch_rewards = rm_wrapper.compute_batch_rewards(tokenized_inputs=tokenized_inputs).cpu().tolist()
            rewards.extend(batch_rewards)

        # Group rewards per prompt
        per_prompt_rewards = []
        curr_prompt_id = dataset[0]["prompt_id"]
        curr_prompt_rewards = []
        for example, reward in zip(dataset, rewards):
            if example["prompt_id"] != curr_prompt_id:
                curr_prompt_id = example["prompt_id"]
                per_prompt_rewards.append(curr_prompt_rewards)
                curr_prompt_rewards = []
            curr_prompt_rewards.append(reward)
        per_prompt_rewards.append(curr_prompt_rewards)

        return torch.tensor(per_prompt_rewards, device=self.device)

    def __subsample_dataset(self, dataset, num_samples: int, seed: int = -1):
        if num_samples >= len(dataset) or num_samples <= 0:
            return dataset, torch.arange(len(dataset))

        if seed > 0:
            perm = np.random.RandomState(seed=seed).permutation(len(dataset))
        else:
            perm = np.random.permutation(len(dataset))

        chosen_indices = perm[:num_samples]
        return dataset.select(chosen_indices), torch.tensor(chosen_indices)

    def __compute_and_log_accuracy_measures(self,
                                            metadata: EvaluationMetadata,
                                            reward_data: RewardData,
                                            per_prompt_online_rm_reward_mean: torch.Tensor,
                                            prob_metrics: Optional[ProbabilityMetrics],
                                            is_offline: bool = False):
        """
        Computes and logs all accuracy metrics for either offline or online responses.
        """
        all_per_prompt_accuracies = {}

        per_prompt_accs = self.__compute_accuracies(
            per_prompt_rm_rewards=reward_data.rm_rewards,
            per_prompt_online_rm_reward_mean=per_prompt_online_rm_reward_mean,
            per_prompt_log_probs=prob_metrics.log_probs,
            per_prompt_gold_rewards=reward_data.gold_rewards,
            use_length_normalized=False,
            is_offline_data=is_offline
        )
        all_per_prompt_accuracies.update(per_prompt_accs)

        length_norm_log_probs = compute_length_normalized_log_probs(prob_metrics.log_probs, prob_metrics.sequence_lengths)
        per_prompt_accs_ln = self.__compute_accuracies(
            per_prompt_rm_rewards=reward_data.rm_rewards,
            per_prompt_online_rm_reward_mean=per_prompt_online_rm_reward_mean,
            per_prompt_log_probs=length_norm_log_probs,
            per_prompt_gold_rewards=reward_data.gold_rewards,
            use_length_normalized=True,
            is_offline_data=is_offline
        )
        all_per_prompt_accuracies.update(per_prompt_accs_ln)

        self.logger.info("=" * 180)
        self.logger.info(f"{'OFFLINE' if is_offline else 'ONLINE'} Accuracy Metrics ({metadata.split}):")
        self.logger.info(f"DATASET: {metadata.dataset_name}")
        self.logger.info(f"LANGUAGE MODEL: {metadata.lm_name}")
        self.logger.info(f"REWARD MODEL: {metadata.rm_name}")
        self.logger.info(f"NUM PROMPTS: {reward_data.rm_rewards.shape[0]}")
        self.logger.info(f"NUM RESPONSES PER PROMPT: {reward_data.rm_rewards.shape[1]}")
        self.logger.info("-" * 180)

        self.logger.info(f"Per prompt accuracy stats aggregated over the dataset, using the current LM:{metadata.lm_name_for_probs} for probs")
        self.logger.info(f"{json.dumps(all_per_prompt_accuracies, indent=2)}")
        self.logger.info("=" * 180)

    def __compute_and_log_accuracy_measures_one_vs_many(self,
                                                        metadata: EvaluationMetadata,
                                                        chosen_rewards: torch.Tensor,
                                                        rejected_rewards: torch.Tensor,
                                                        per_prompt_online_rm_reward_mean: torch.Tensor,
                                                        prob_metrics: ProbabilityMetrics):
        """
        Computes and logs all one-vs-many accuracy metrics.
        chosen_rewards: shape (num_prompts,)
        rejected_rewards: shape (num_prompts, num_rejected)
        prob_metrics: Required. log_probs should have shape (num_prompts, num_rejected+1) where column 0 is chosen, rest are rejected.
        """
        all_per_prompt_accuracies = {}

        if prob_metrics.log_probs.shape[1] != rejected_rewards.shape[1] + 1:
            raise ValueError(f"log_probs shape mismatch: expected (num_prompts, {rejected_rewards.shape[1] + 1}), got {prob_metrics.log_probs.shape}")

        log_probs_chosen = prob_metrics.log_probs[:, 0]
        log_probs_rejected = prob_metrics.log_probs[:, 1:]

        per_prompt_accs = self.__compute_accuracies_one_vs_many(
            per_prompt_rm_rewards_chosen=chosen_rewards,
            per_prompt_rm_rewards_rejected=rejected_rewards,
            per_prompt_online_rm_reward_mean=per_prompt_online_rm_reward_mean,
            per_prompt_log_probs_chosen=log_probs_chosen,
            per_prompt_log_probs_rejected=log_probs_rejected,
            use_length_normalized=False
        )
        all_per_prompt_accuracies.update(per_prompt_accs)

        length_norm_log_probs = compute_length_normalized_log_probs(prob_metrics.log_probs, prob_metrics.sequence_lengths)
        length_norm_log_probs_chosen = length_norm_log_probs[:, 0]
        length_norm_log_probs_rejected = length_norm_log_probs[:, 1:]

        per_prompt_accs_ln = self.__compute_accuracies_one_vs_many(
            per_prompt_rm_rewards_chosen=chosen_rewards,
            per_prompt_rm_rewards_rejected=rejected_rewards,
            per_prompt_online_rm_reward_mean=per_prompt_online_rm_reward_mean,
            per_prompt_log_probs_chosen=length_norm_log_probs_chosen,
            per_prompt_log_probs_rejected=length_norm_log_probs_rejected,
            use_length_normalized=True
        )
        all_per_prompt_accuracies.update(per_prompt_accs_ln)

        self.logger.info("=" * 180)
        self.logger.info(f"OFFLINE Accuracy Metrics (ONE-VS-MANY) ({metadata.split}):")
        self.logger.info(f"DATASET: {metadata.dataset_name}")
        self.logger.info(f"LANGUAGE MODEL: {metadata.lm_name}")
        self.logger.info(f"REWARD MODEL: {metadata.rm_name}")
        self.logger.info(f"NUM PROMPTS: {chosen_rewards.shape[0]}")
        self.logger.info(f"NUM REJECTED PER PROMPT: {rejected_rewards.shape[1]}")
        self.logger.info("-" * 180)
        self.logger.info(f"Per prompt accuracy stats aggregated over the dataset, using the current LM:{metadata.lm_name_for_probs} for probs")
        self.logger.info(f"{json.dumps(all_per_prompt_accuracies, indent=2)}")
        self.logger.info("=" * 180)

    def __prepare_train_and_test_datasets(self,
                                          dataset_path: str,
                                          load_dataset_from_file: bool,
                                          train_split: str = None,
                                          test_split: str = None,
                                          num_train_samples: int = -1,
                                          num_test_samples: int = -1):
        if not load_dataset_from_file:
            preference_dataset = load_dataset(dataset_path, trust_remote_code=True, cache_dir=self.config.cache_dir)
        else:
            preference_dataset = load_from_disk(dataset_path)

        random_seed = self.config.rm_eval_config.data_selection_seed
        if train_split:
            train_dataset, train_prompt_indices = self.__subsample_dataset(preference_dataset[train_split],
                                                                           num_samples=num_train_samples,
                                                                           seed=random_seed)
        else:
            train_dataset, train_prompt_indices = None, None

        if test_split:
            test_dataset, test_prompt_indices = self.__subsample_dataset(preference_dataset[test_split],
                                                                         num_samples=num_test_samples,
                                                                         seed=random_seed + 1 if random_seed > 0 else -1)
        else:
            test_dataset, test_prompt_indices = None, None

        return train_dataset, train_prompt_indices, test_dataset, test_prompt_indices

    def __compute_sequence_log_probs(self, lm_wrapper, prompts, responses):
        full_texts = [
            lm_wrapper.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
                tokenize=False,
                enable_thinking=False
            )
            for p, r in zip(prompts, responses)
        ]
        prompt_texts = [
            lm_wrapper.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False
            )
            for p in prompts
        ]

        full_tokenized = lm_wrapper.tokenizer(full_texts, return_tensors="pt", padding=True,
                                              truncation=False, add_special_tokens=False)
        only_prompt_tokenized = lm_wrapper.tokenizer(prompt_texts, return_tensors="pt", padding=True,
                                                     truncation=False, add_special_tokens=False)

        input_ids = full_tokenized["input_ids"].to(lm_wrapper.lm.device)
        attn_full = full_tokenized["attention_mask"].to(lm_wrapper.lm.device)
        full_seq_len = attn_full.sum(dim=1)
        prompt_len = only_prompt_tokenized["attention_mask"].sum(dim=1).to(lm_wrapper.lm.device)

        # response mask positions (left or right padding!! - in prepare of CausalLMWrapper it should be left padding)
        pos = torch.arange(input_ids.size(1), device=lm_wrapper.lm.device).unsqueeze(0)
        left_pad = (getattr(lm_wrapper.tokenizer, "padding_side", "right") == "left")
        if left_pad:
            seq_start = input_ids.size(1) - full_seq_len
            resp_start = seq_start + prompt_len
            resp_end = seq_start + full_seq_len
        else:
            resp_start = prompt_len
            resp_end = full_seq_len

        resp_mask = (pos >= resp_start.unsqueeze(1)) & (pos < resp_end.unsqueeze(1))

        with torch.no_grad():
            out = lm_wrapper.lm(input_ids=input_ids, attention_mask=attn_full)
            logp = torch.nn.functional.log_softmax(out.logits, dim=-1)

        response_ids = input_ids[:, 1:]
        logp = logp[:, :-1, :]
        resp_mask = resp_mask[:, 1:]

        per_token_logp = logp.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
        per_token_logp = per_token_logp.masked_fill(~resp_mask, 0.0)

        seq_logp = per_token_logp.sum(dim=1)
        resp_len = resp_mask.sum(dim=1)

        return seq_logp, resp_len

    def __compute_offline_log_probs_for_dataset(self, lm_wrapper, dataset, split_name: str):
        """
        Computes the probabilities pi(chosen |x) and pi(rejected|x) for offline responses.
        Returns a tensor of shape (num_examples, 2) where column 0 is pi(chosen|x)
        and column 1 is pi(rejected|x).
        """
        self.logger.info(f"Using LM {self.lm_name}, starting to get probs for the OFFLINE {split_name.upper()} split")

        prompts_dataset = PromptCompletionDataset(dataset)
        prompts_dataloader = DataLoader(
            prompts_dataset,
            batch_size=self.config.rm_eval_config.lm_batch_size,
            shuffle=False,
            collate_fn=lambda batch: {
                "prompts": [item["prompt"] for item in batch],
                "prompt_ids": [item["prompt_id"] for item in batch],
                "chosen": [item["chosen"] for item in batch],
                "rejected": [item["rejected"] for item in batch]
            }
        )

        num_batches = self.__compute_num_batches(len(prompts_dataset), self.config.rm_eval_config.lm_batch_size)

        all_chosen_log_probs, all_chosen_lengths = [], []
        all_rejected_log_probs, all_rejected_lengths = [], []

        for i, batch in enumerate(prompts_dataloader):
            if i % 10 == 0:
                self.logger.info(f"Getting probs for batch {i + 1} / {num_batches}")

            batch_prompts = batch["prompts"]
            batch_chosen = batch["chosen"]
            batch_rejected = batch["rejected"]

            batch_chosen_responses = [chosen[1]["content"] for chosen in batch_chosen]
            batch_rejected_responses = [rejected[1]["content"] for rejected in batch_rejected]

            # Compute log probs and sequence lengths for chosen responses
            chosen_log_probs, chosen_lengths = self.__compute_sequence_log_probs(
                lm_wrapper, batch_prompts, batch_chosen_responses
            )
            all_chosen_log_probs.append(chosen_log_probs)
            all_chosen_lengths.append(chosen_lengths)

            # Compute log probs and sequence lengths for rejected responses
            rejected_log_probs, rejected_lengths = self.__compute_sequence_log_probs(
                lm_wrapper, batch_prompts, batch_rejected_responses
            )
            all_rejected_log_probs.append(rejected_log_probs)
            all_rejected_lengths.append(rejected_lengths)

        all_chosen_log_probs = torch.cat(all_chosen_log_probs)
        all_rejected_log_probs = torch.cat(all_rejected_log_probs)
        all_chosen_lengths = torch.cat(all_chosen_lengths)
        all_rejected_lengths = torch.cat(all_rejected_lengths)

        log_probs = torch.stack([all_chosen_log_probs, all_rejected_log_probs], dim=1)
        response_lengths = torch.stack([all_chosen_lengths, all_rejected_lengths], dim=1)
        return log_probs, response_lengths

    def __compute_response_logprobs_and_lengths(self,
                                                train_dataset=None,
                                                train_split: str = None,
                                                test_dataset=None,
                                                test_split: str = None):
        """
        Computes log(pi(y_chosen|x)) and log(pi(y_rejected|x)) for responses in the train and test splits.
        """
        lm_wrapper = self.__get_lm_wrapper(self.config.rm_eval_config.language_model_path,
                                           device=self.device,
                                           config=self.config.rm_eval_config)

        train_offline_log_probs, train_offline_lengths = None, None
        test_offline_log_probs, test_offline_lengths = None, None

        if train_split:
            train_offline_log_probs, train_offline_lengths = self.__compute_offline_log_probs_for_dataset(
                lm_wrapper, train_dataset, train_split
            )

        if test_split:
            test_offline_log_probs, test_offline_lengths = self.__compute_offline_log_probs_for_dataset(
                lm_wrapper, test_dataset, test_split
            )

        del lm_wrapper
        torch.cuda.empty_cache()
        gc.collect()

        return train_offline_log_probs, test_offline_log_probs, train_offline_lengths, test_offline_lengths

    def __compute_offline_log_probs_for_dataset_multiple_rejected(self, lm_wrapper, dataset, split_name: str):
        """
        Computes the probabilities pi(chosen|x) and pi(rejected_i|x) for offline responses with multiple rejected.
        Returns log_probs of shape (num_examples, max_rejected+1) where column 0 is pi(chosen|x)
        and columns 1: are pi(rejected_i|x) for each rejected response.
        Also returns sequence_lengths of the same shape.
        """
        self.logger.info(f"Using LM {self.lm_name}, starting to get probs for the OFFLINE {split_name.upper()} split (MULTIPLE REJECTED)")

        # First pass: find max number of rejected responses
        max_rejected = 0
        for example in dataset:
            rejected_convs = example["rejected"]
            max_rejected = max(max_rejected, len(rejected_convs))

        prompts_dataset = PromptCompletionDataset(dataset)
        prompts_dataloader = DataLoader(
            prompts_dataset,
            batch_size=self.config.rm_eval_config.lm_batch_size,
            shuffle=False,
            collate_fn=lambda batch: {
                "prompts": [item["prompt"] for item in batch],
                "prompt_ids": [item["prompt_id"] for item in batch],
                "chosen": [item["chosen"] for item in batch],
                "rejected": [item["rejected"] for item in batch]
            }
        )

        num_batches = self.__compute_num_batches(len(prompts_dataset), self.config.rm_eval_config.lm_batch_size)

        all_chosen_log_probs, all_chosen_lengths = [], []
        all_rejected_log_probs_per_example = []  # List of lists, each inner list has log probs for all rejected
        all_rejected_lengths_per_example = []

        for i, batch in enumerate(prompts_dataloader):
            if i % 10 == 0:
                self.logger.info(f"Getting probs for batch {i + 1} / {num_batches}")

            batch_prompts = batch["prompts"]
            batch_chosen = batch["chosen"]
            batch_rejected = batch["rejected"]

            # Compute log probabilities for chosen responses
            batch_chosen_responses = [chosen[1]["content"] for chosen in batch_chosen]
            chosen_log_probs, chosen_lengths = self.__compute_sequence_log_probs(
                lm_wrapper, batch_prompts, batch_chosen_responses
            )
            all_chosen_log_probs.append(chosen_log_probs)
            all_chosen_lengths.append(chosen_lengths)

            # Compute log probabilities for all rejected responses
            for example_idx, rejected_convs in enumerate(batch_rejected):
                example_rejected_log_probs = []
                example_rejected_lengths = []
                prompt = batch_prompts[example_idx]

                # Flatten all rejected responses for this example
                rejected_responses = [rej_conv[1]["content"] for rej_conv in rejected_convs]

                if len(rejected_responses) > 0:
                    rej_log_probs, rej_lengths = self.__compute_sequence_log_probs(
                        lm_wrapper, [prompt] * len(rejected_responses), rejected_responses
                    )
                    example_rejected_log_probs = rej_log_probs.tolist()
                    example_rejected_lengths = rej_lengths.tolist()

                all_rejected_log_probs_per_example.append(example_rejected_log_probs)
                all_rejected_lengths_per_example.append(example_rejected_lengths)

        # Stack chosen log probs
        all_chosen_log_probs = torch.cat(all_chosen_log_probs)
        all_chosen_lengths = torch.cat(all_chosen_lengths)

        # Pad rejected log probs to same length and stack
        padded_rejected_log_probs = []
        padded_rejected_lengths = []
        for rej_log_probs, rej_lengths in zip(all_rejected_log_probs_per_example, all_rejected_lengths_per_example):
            padding_size = max_rejected - len(rej_log_probs)
            rej_log_probs.extend([0.0] * padding_size)
            rej_lengths.extend([1] * padding_size)

            padded_rejected_log_probs.append(torch.tensor(rej_log_probs, device=all_chosen_log_probs.device))
            padded_rejected_lengths.append(torch.tensor(rej_lengths, device=all_chosen_lengths.device))

        all_rejected_log_probs = torch.stack(padded_rejected_log_probs)  # shape: (num_examples, max_rejected)
        all_rejected_lengths = torch.stack(padded_rejected_lengths)

        # Stack chosen (column 0) and rejected (columns 1:)
        log_probs = torch.cat([all_chosen_log_probs.unsqueeze(1), all_rejected_log_probs], dim=1)  # shape: (num_examples, max_rejected+1)
        sequence_lengths = torch.cat([all_chosen_lengths.unsqueeze(1), all_rejected_lengths], dim=1)

        self.logger.info(f"Computed log probs shape: {log_probs.shape}")
        self.logger.info("=" * 180)

        return log_probs, sequence_lengths

    def __compute_accuracies(self,
                             per_prompt_rm_rewards: torch.Tensor,
                             per_prompt_online_rm_reward_mean: torch.Tensor,
                             per_prompt_log_probs: torch.Tensor,
                             per_prompt_gold_rewards: Optional[torch.Tensor],
                             use_length_normalized: bool = False,
                             is_offline_data: bool = False) -> dict:
        """
        Compute overall and per-prompt accuracy metrics from per-prompt data.
        Uses the accuracy functions from accuracy.py with the 'overall' flag.
        """
        # Get indices for all pairs (i, j) where i < j
        num_responses = per_prompt_rm_rewards.shape[1]
        idx1, idx2 = torch.triu_indices(num_responses, num_responses, offset=1, device=per_prompt_rm_rewards.device)

        # Extract pairs based on indices
        rm_reward_1 = per_prompt_rm_rewards[:, idx1]  # shape: (num_prompts, num_pairs)
        rm_reward_2 = per_prompt_rm_rewards[:, idx2]
        log_prob_1 = per_prompt_log_probs[:, idx1]
        log_prob_2 = per_prompt_log_probs[:, idx2]

        if not is_offline_data:
            # For online data use ground truth rewards to determine which response is preferred
            gt_reward_1 = per_prompt_gold_rewards[:, idx1]
            gt_reward_2 = per_prompt_gold_rewards[:, idx2]

            # Chosen is the one with higher ground truth reward
            gt_1_preferred = gt_reward_1 > gt_reward_2

            # Swap pairs where j has higher GT reward than i
            rm_chosen_per_prompt = torch.where(gt_1_preferred, rm_reward_1, rm_reward_2)
            rm_rejected_per_prompt = torch.where(gt_1_preferred, rm_reward_2, rm_reward_1)
            log_prob_chosen_per_prompt = torch.where(gt_1_preferred, log_prob_1, log_prob_2)
            log_prob_rejected_per_prompt = torch.where(gt_1_preferred, log_prob_2, log_prob_1)
        else:
            # For offline data idx1/idx2 already represent chosen/rejected from dataset
            rm_chosen_per_prompt = rm_reward_1
            rm_rejected_per_prompt = rm_reward_2
            log_prob_chosen_per_prompt = log_prob_1
            log_prob_rejected_per_prompt = log_prob_2

        suffix = "_length_normalized" if use_length_normalized else ""
        per_prompt_accuracy_measures = {}

        per_prompt_accuracy_measures[f'acc'] = acc(rm_chosen_per_prompt, rm_rejected_per_prompt).item()

        per_prompt_accuracy_measures[f'acc_w{suffix}'] = acc_w(
            rm_chosen_per_prompt, rm_rejected_per_prompt,
            log_prob_chosen_per_prompt, log_prob_rejected_per_prompt
        ).item()

        per_prompt_accuracy_measures[f'hacc_w{suffix}'] = hacc_w(
            rm_chosen_per_prompt, rm_rejected_per_prompt,
            log_prob_chosen_per_prompt, log_prob_rejected_per_prompt,
            per_prompt_mean_reward=per_prompt_online_rm_reward_mean
        ).item()

        return per_prompt_accuracy_measures

    def __compute_accuracies_one_vs_many(self,
                                         per_prompt_rm_rewards_chosen: torch.Tensor,
                                         per_prompt_rm_rewards_rejected: torch.Tensor,
                                         per_prompt_online_rm_reward_mean: torch.Tensor,
                                         per_prompt_log_probs_chosen: torch.Tensor,
                                         per_prompt_log_probs_rejected: torch.Tensor,
                                         use_length_normalized: bool = False) -> dict:
        """
        Compute one-vs-many accuracy metrics from per-prompt data.
        For each prompt, checks if the chosen reward is greater than all rejected rewards.
        
        Args:
            per_prompt_rm_rewards_chosen: Tensor of shape (num_prompts,) containing chosen rewards
            per_prompt_rm_rewards_rejected: Tensor of shape (num_prompts, num_rejected) containing rejected rewards
            per_prompt_online_rm_reward_mean: Tensor of shape (num_prompts,) containing mean rewards for computing MaRAcc
            per_prompt_log_probs_chosen: Tensor of shape (num_prompts,) containing chosen log probs
            per_prompt_log_probs_rejected: Tensor of shape (num_prompts, num_rejected) containing rejected log probs
            use_length_normalized: Whether to use length-normalized suffix
        """
        suffix = "_length_normalized" if use_length_normalized else ""
        per_prompt_accuracy_measures = {}

        per_prompt_accuracy_measures[f'one_vs_many_acc'] = one_vs_many_acc(
            per_prompt_rm_rewards_chosen, per_prompt_rm_rewards_rejected
        ).item()

        per_prompt_accuracy_measures[f'one_vs_many_acc_w{suffix}'] = one_vs_many_acc_w(
            per_prompt_rm_rewards_chosen, per_prompt_rm_rewards_rejected,
            per_prompt_log_probs_chosen, per_prompt_log_probs_rejected
        ).item()

        per_prompt_accuracy_measures[f'one_vs_many_hacc_w{suffix}'] = one_vs_many_hacc_w(
            per_prompt_rm_rewards_chosen, per_prompt_rm_rewards_rejected,
            per_prompt_log_probs_chosen, per_prompt_log_probs_rejected,
            per_prompt_mean_reward=per_prompt_online_rm_reward_mean
        ).item()

        return per_prompt_accuracy_measures

    def __convert_probs_dataset_to_tensor(self, probs_dataset):
        """
        Converts a dataset with 'prompt_id', 'response_log_probs', and 'response_sequence_length'
        to tensors of shape (num_prompts, num_responses_per_prompt).
        Returns probs tensor and sequence_lengths tensor.
        """
        per_prompt_probs = []
        per_prompt_sequence_lengths = []
        curr_prompt_id = probs_dataset[0]["prompt_id"]
        curr_prompt_probs = []
        curr_prompt_lengths = []

        for example in probs_dataset:
            if example["prompt_id"] != curr_prompt_id:
                per_prompt_probs.append(curr_prompt_probs)
                per_prompt_sequence_lengths.append(curr_prompt_lengths)
                curr_prompt_id = example["prompt_id"]
                curr_prompt_probs = []
                curr_prompt_lengths = []
            curr_prompt_probs.append(example["response_log_probs"])
            curr_prompt_lengths.append(example["response_sequence_length"])
        per_prompt_probs.append(curr_prompt_probs)
        per_prompt_sequence_lengths.append(curr_prompt_lengths)

        probs_tensor = torch.tensor(per_prompt_probs, device=self.device)
        sequence_lengths_tensor = torch.tensor(per_prompt_sequence_lengths, device=self.device, dtype=torch.long)

        return probs_tensor, sequence_lengths_tensor

    def __create_online_responses_datasets(self,
                                           dataset_name: str,
                                           train_split: str = None,
                                           train_dataset=None,
                                           train_prompt_indices=None,
                                           test_split: str = None,
                                           test_dataset=None,
                                           test_prompt_indices=None):
        lm_wrapper = self.__get_lm_wrapper(self.config.rm_eval_config.language_model_path,
                                           device=self.device,
                                           config=self.config.rm_eval_config)
        online_train_dataset = None
        online_train_log_probs = None
        online_train_sequence_lengths = None
        if train_split:
            self.logger.info(f"Using LM {self.lm_name}, starting to generate responses for the TRAIN split")
            online_train_dataset, online_train_dataset_probs_ds = self.__create_dataset_with_generated_responses(lm_wrapper=lm_wrapper,
                                                                                                                 responses_out_file=self.lm_generation_train_path,
                                                                                                                 prompts_out_file=self.prompt_train_path,
                                                                                                                 dataset=train_dataset,
                                                                                                                 dataset_name=dataset_name,
                                                                                                                 split=train_split,
                                                                                                                 prompt_indices=train_prompt_indices)
            online_train_log_probs, online_train_sequence_lengths = self.__convert_probs_dataset_to_tensor(online_train_dataset_probs_ds)

        online_test_dataset = None
        online_test_log_probs = None
        online_test_sequence_lengths = None
        if test_split:
            self.logger.info(f"Using LM {self.lm_name}, starting to generate responses for the TEST split")
            online_test_dataset, online_test_dataset_probs_ds = self.__create_dataset_with_generated_responses(lm_wrapper=lm_wrapper,
                                                                                                               responses_out_file=self.lm_generation_test_path,
                                                                                                               prompts_out_file=self.prompt_test_path,
                                                                                                               dataset=test_dataset,
                                                                                                               dataset_name=dataset_name,
                                                                                                               split=test_split,
                                                                                                               prompt_indices=test_prompt_indices)
            online_test_log_probs, online_test_sequence_lengths = self.__convert_probs_dataset_to_tensor(online_test_dataset_probs_ds)

        # Free device from language model
        del lm_wrapper

        if self.config.rm_eval_config.delete_language_model_checkpoint_after_eval:
            self.logger.warning(f"DELETING language model checkpoint from: {self.config.rm_eval_config.language_model_path}")
            shutil.rmtree(self.config.rm_eval_config.language_model_path, ignore_errors=True)

        torch.cuda.empty_cache()
        gc.collect()
        return (online_train_dataset, online_test_dataset, online_train_log_probs, online_test_log_probs,
                online_train_sequence_lengths, online_test_sequence_lengths)

    def __create_prompt_response_dataset_from_offline_responses(self, dataset):
        """
        Receives a Hugging Face dataset that contains columns named 'prompt', 'prompt_id', 'chosen', and 'rejected',
        and returns a dataset with columns 'prompt', 'prompt_id', 'response', where the order of the prompts is maintained and for
        each prompt the chosen and rejected responses appear in consecutive rows (first the chosen and then the rejected)..
        """
        buffer = []
        for example in dataset:
            prompt = example["prompt"]
            prompt_id = example["prompt_id"]
            chosen = example["chosen"][1]["content"]
            rejected = example["rejected"][1]["content"]

            buffer.append({
                "prompt": prompt,
                "prompt_id": prompt_id,
                "response": chosen
            })
            buffer.append({
                "prompt": prompt,
                "prompt_id": prompt_id,
                "response": rejected
            })

        return Dataset.from_list(buffer)

    def __compute_and_save_rewards_and_accuracies_for_rm(
            self,
            online_train_prompt_response_dataset,
            online_test_prompt_response_dataset,
            rm_name: str,
            rm_wrapper: RewardModelWrapper,
            train_metrics: SplitMetrics,
            test_metrics: SplitMetrics):

        # Compute rewards over the ONLINE responses
        train_online_rm_rewards, train_online_metadata = None, None
        if train_metrics.split_name:
            self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for ONLINE TRAIN responses")
            train_online_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                        dataset=online_train_prompt_response_dataset)
            train_online_metadata = EvaluationMetadata(
                lm_name=self.lm_name,
                lm_name_for_probs=self.lm_name,
                type_of_generations=ONLINE_RESPONSES_NAME,
                rm_name=rm_name,
                dataset_name=train_metrics.dataset_name,
                split=train_metrics.split_name,
                prompt_indices=train_metrics.prompt_indices,
                lm_file_name=self.lm_file_name
            )
            self.__save_rewards_and_log_reward_metrics(per_prompt_rewards=train_online_rm_rewards, metadata=train_online_metadata)

        test_online_rm_rewards, test_online_metadata = None, None
        if test_metrics.split_name:
            self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for ONLINE TEST responses")
            test_online_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                       dataset=online_test_prompt_response_dataset)
            test_online_metadata = EvaluationMetadata(
                lm_name=self.lm_name,
                lm_name_for_probs=self.lm_name,
                type_of_generations=ONLINE_RESPONSES_NAME,
                rm_name=rm_name,
                dataset_name=test_metrics.dataset_name,
                split=test_metrics.split_name,
                prompt_indices=test_metrics.prompt_indices,
                lm_file_name=self.lm_file_name
            )
            self.__save_rewards_and_log_reward_metrics(per_prompt_rewards=test_online_rm_rewards, metadata=test_online_metadata)

        # Compute rewards over the OFFLINE responses
        train_offline_rm_rewards, train_offline_metadata = None, None
        test_offline_rm_rewards, test_offline_metadata = None, None
        if not self.config.rm_eval_config.only_lm_eval:
            if train_metrics.split_name:
                self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for OFFLINE TRAIN responses")
                train_offline_prompt_response_dataset = self.__create_prompt_response_dataset_from_offline_responses(train_metrics.dataset)
                train_offline_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                             dataset=train_offline_prompt_response_dataset)
                train_offline_metadata = EvaluationMetadata(
                    lm_name=OFFLINE_RESPONSES_NAME,
                    lm_name_for_probs=self.lm_name,
                    type_of_generations=OFFLINE_RESPONSES_NAME,
                    rm_name=rm_name,
                    dataset_name=train_metrics.dataset_name,
                    split=train_metrics.split_name,
                    prompt_indices=train_metrics.prompt_indices
                )
                self.__save_rewards_and_log_reward_metrics(per_prompt_rewards=train_offline_rm_rewards, metadata=train_offline_metadata)

            if test_metrics.split_name:
                self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for OFFLINE TEST responses")
                test_offline_prompt_response_dataset = self.__create_prompt_response_dataset_from_offline_responses(test_metrics.dataset)
                test_offline_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                            dataset=test_offline_prompt_response_dataset)
                test_offline_metadata = EvaluationMetadata(
                    lm_name=OFFLINE_RESPONSES_NAME,
                    lm_name_for_probs=self.lm_name,
                    type_of_generations=OFFLINE_RESPONSES_NAME,
                    rm_name=rm_name,
                    dataset_name=test_metrics.dataset_name,
                    split=test_metrics.split_name,
                    prompt_indices=test_metrics.prompt_indices
                )
                self.__save_rewards_and_log_reward_metrics(per_prompt_rewards=test_offline_rm_rewards, metadata=test_offline_metadata)

        del rm_wrapper
        torch.cuda.empty_cache()
        gc.collect()

        if train_online_rm_rewards is not None:
            # Compute all accuracy metrics for ONLINE data
            train_online_reward_data = RewardData(
                rm_rewards=train_online_rm_rewards,
                gold_rewards=train_metrics.online_gold_rewards
            )
            per_prompt_train_online_rm_reward_mean = train_online_rm_rewards.mean(dim=1)
            self.__compute_and_log_accuracy_measures(
                metadata=train_online_metadata,
                reward_data=train_online_reward_data,
                per_prompt_online_rm_reward_mean=per_prompt_train_online_rm_reward_mean,
                prob_metrics=train_metrics.online_log_probs,
                is_offline=False
            )

            if train_offline_rm_rewards is not None:
                # Compute all accuracy metrics for OFFLINE data
                train_offline_reward_data = RewardData(
                    rm_rewards=train_offline_rm_rewards,
                    gold_rewards=train_metrics.offline_gold_rewards
                )
                self.__compute_and_log_accuracy_measures(
                    metadata=train_offline_metadata,
                    reward_data=train_offline_reward_data,
                    per_prompt_online_rm_reward_mean=per_prompt_train_online_rm_reward_mean,
                    prob_metrics=train_metrics.offline_log_probs,
                    is_offline=True
                )

        if test_online_rm_rewards is not None:
            # Compute all accuracy metrics for ONLINE data - test
            test_online_reward_data = RewardData(
                rm_rewards=test_online_rm_rewards,
                gold_rewards=test_metrics.online_gold_rewards
            )
            per_prompt_test_online_rm_reward_mean = test_online_rm_rewards.mean(dim=1)
            self.__compute_and_log_accuracy_measures(
                metadata=test_online_metadata,
                reward_data=test_online_reward_data,
                per_prompt_online_rm_reward_mean=per_prompt_test_online_rm_reward_mean,
                prob_metrics=test_metrics.online_log_probs,
                is_offline=False
            )

            if test_offline_rm_rewards is not None:
                # Compute all accuracy metrics for OFFLINE data
                test_offline_reward_data = RewardData(
                    rm_rewards=test_offline_rm_rewards,
                    gold_rewards=test_metrics.offline_gold_rewards
                )
                self.__compute_and_log_accuracy_measures(
                    metadata=test_offline_metadata,
                    reward_data=test_offline_reward_data,
                    per_prompt_online_rm_reward_mean=per_prompt_test_online_rm_reward_mean,
                    prob_metrics=test_metrics.offline_log_probs,
                    is_offline=True
                )

    def __log_reward_metrics_and_save_rewards_for_gt_reward(self, rm_wrapper, dataset_path: str, dataset,
                                                            split: str, prompt_indices, is_offline: bool) -> RewardData:
        self.logger.info(f"For RM {self.gold_rm_name}, starting to compute per prompt rewards for "
                         f"{'OFFLINE' if is_offline else 'ONLINE'} {split} responses")
        if is_offline:
            gold_rewards = torch.stack([torch.tensor(dataset[key], device=self.device) for key in ["score_chosen", "score_rejected"]],
                                       dim=1)
        else:
            gold_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper, dataset=dataset)

        gold_metadata = EvaluationMetadata(
            lm_name=OFFLINE_RESPONSES_NAME if is_offline else self.lm_name,
            lm_name_for_probs=self.lm_name,
            type_of_generations=OFFLINE_RESPONSES_NAME if is_offline else ONLINE_RESPONSES_NAME,
            rm_name=self.gold_rm_name,
            dataset_name=dataset_path,
            split=split,
            prompt_indices=prompt_indices,
            lm_file_name=OFFLINE_RESPONSES_NAME if is_offline else self.lm_file_name
        )
        gold_reward_data = RewardData(
            rm_rewards=gold_rewards,
            gold_rewards=gold_rewards
        )

        self.__save_rewards_and_log_reward_metrics(
            per_prompt_rewards=gold_rewards,
            metadata=gold_metadata
        )

        return gold_reward_data

    def __log_reward_metrics_and_save_rewards_for_gt_reward_multiple_rejected(self, dataset_path: str, dataset, split: str, prompt_indices) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Extract rewards for multiple rejected responses and save them.
        Returns: (chosen_rewards, rejected_rewards) where chosen_rewards is (num_prompts,) and rejected_rewards is (num_prompts, num_rejected)
        """
        self.logger.info(f"For RM {self.gold_rm_name}, starting to compute per prompt rewards for "
                         f"OFFLINE {split} responses (MULTIPLE REJECTED)")
        # Extract chosen and rejected scores
        chosen_scores = torch.tensor(dataset["score_chosen"], device=self.device)  # shape: (num_prompts,)
        rejected_scores_list = dataset["score_rejected"]  # list of lists

        # Convert list of lists to tensor
        max_rejected = max(len(rej) for rej in rejected_scores_list)
        rejected_rewards = torch.full((len(rejected_scores_list), max_rejected), float('-inf'), device=self.device)
        for i, rej_scores in enumerate(rejected_scores_list):
            rejected_rewards[i, :len(rej_scores)] = torch.tensor(rej_scores, device=self.device)

        # Create metadata and save rewards
        gold_metadata = EvaluationMetadata(
            lm_name=OFFLINE_RESPONSES_NAME,
            lm_name_for_probs=self.lm_name,
            type_of_generations=OFFLINE_RESPONSES_NAME,
            rm_name=self.gold_rm_name,
            dataset_name=dataset_path,
            split=split,
            prompt_indices=prompt_indices,
            lm_file_name=OFFLINE_RESPONSES_NAME
        )
        self.__save_rewards_and_log_reward_metrics_one_vs_many(
            chosen_rewards=chosen_scores,
            rejected_rewards=rejected_rewards,
            metadata=gold_metadata
        )

        return chosen_scores, rejected_rewards

    def __compute_and_log_accuracy_measures_for_gt_reward(self, dataset_path: str, split: str, prompt_indices, is_offline: bool,
                                                          gold_reward_data: RewardData, log_probs: torch.Tensor, response_lengths: torch.Tensor,
                                                          per_prompt_online_gt_rewards: torch.Tensor):
        gold_metadata = EvaluationMetadata(
            lm_name=OFFLINE_RESPONSES_NAME if is_offline else self.lm_name,
            lm_name_for_probs=self.lm_name,
            type_of_generations=OFFLINE_RESPONSES_NAME if is_offline else ONLINE_RESPONSES_NAME,
            rm_name=self.gold_rm_name,
            dataset_name=dataset_path,
            split=split,
            prompt_indices=prompt_indices,
            lm_file_name=OFFLINE_RESPONSES_NAME if is_offline else self.lm_file_name
        )
        log_prob_metrics = ProbabilityMetrics(
            log_probs=log_probs,
            sequence_lengths=response_lengths
        )
        self.__compute_and_log_accuracy_measures(
            metadata=gold_metadata,
            reward_data=gold_reward_data,
            per_prompt_online_rm_reward_mean=per_prompt_online_gt_rewards.mean(dim=1),
            prob_metrics=log_prob_metrics,
            is_offline=is_offline
        )

    def __compute_and_log_accuracy_measures_for_gt_reward_and_all_datasets(self, dataset_path: str,
                                                                           test_offline_gold_reward_data: RewardData,
                                                                           test_offline_log_probs: torch.Tensor,
                                                                           test_offline_response_lengths: torch.Tensor,
                                                                           test_online_gold_reward_data: RewardData,
                                                                           test_online_log_probs: torch.Tensor,
                                                                           test_online_response_lengths: torch.Tensor,
                                                                           test_prompt_indices: torch.Tensor,
                                                                           test_split: str,
                                                                           train_offline_gold_reward_data: RewardData,
                                                                           train_offline_log_probs: torch.Tensor,
                                                                           train_offline_response_lengths: torch.Tensor,
                                                                           train_online_gold_reward_data: RewardData,
                                                                           train_online_log_probs: torch.Tensor,
                                                                           train_online_response_lengths: torch.Tensor,
                                                                           train_prompt_indices: torch.Tensor,
                                                                           train_split: str):
        if train_offline_gold_reward_data is not None:
            self.__compute_and_log_accuracy_measures_for_gt_reward(dataset_path=dataset_path,
                                                                   split=train_split,
                                                                   prompt_indices=train_prompt_indices,
                                                                   is_offline=True,
                                                                   gold_reward_data=train_offline_gold_reward_data,
                                                                   log_probs=train_offline_log_probs,
                                                                   response_lengths=train_offline_response_lengths,
                                                                   per_prompt_online_gt_rewards=train_online_gold_reward_data.gold_rewards)
        if test_offline_gold_reward_data is not None:
            self.__compute_and_log_accuracy_measures_for_gt_reward(dataset_path=dataset_path,
                                                                   split=test_split,
                                                                   prompt_indices=test_prompt_indices,
                                                                   is_offline=True,
                                                                   gold_reward_data=test_offline_gold_reward_data,
                                                                   log_probs=test_offline_log_probs,
                                                                   response_lengths=test_offline_response_lengths,
                                                                   per_prompt_online_gt_rewards=test_online_gold_reward_data.gold_rewards)
        if train_online_gold_reward_data is not None:
            self.__compute_and_log_accuracy_measures_for_gt_reward(dataset_path=dataset_path,
                                                                   split=train_split,
                                                                   prompt_indices=train_prompt_indices,
                                                                   is_offline=False,
                                                                   gold_reward_data=train_online_gold_reward_data,
                                                                   log_probs=train_online_log_probs,
                                                                   response_lengths=train_online_response_lengths,
                                                                   per_prompt_online_gt_rewards=train_online_gold_reward_data.gold_rewards)

        if test_online_gold_reward_data is not None:
            self.__compute_and_log_accuracy_measures_for_gt_reward(dataset_path=dataset_path,
                                                                   split=test_split,
                                                                   prompt_indices=test_prompt_indices,
                                                                   is_offline=False,
                                                                   gold_reward_data=test_online_gold_reward_data,
                                                                   log_probs=test_online_log_probs,
                                                                   response_lengths=test_online_response_lengths,
                                                                   per_prompt_online_gt_rewards=test_online_gold_reward_data.gold_rewards)

    def __run_eval_for_dataset(self, dataset_path: str, train_split: str, test_split: str,
                               train_dataset, train_prompt_indices, test_dataset, test_prompt_indices):
        self.logger.info("-" * 180)
        self.logger.info(f"STARTING EVALUATION USING DATASET: {dataset_path}")

        # Compute and save logprobs over preferences in the dataset (offline responses)
        train_offline_log_probs, test_offline_log_probs, train_offline_response_lengths, test_offline_response_lengths = self.__compute_response_logprobs_and_lengths(
            train_dataset=train_dataset,
            train_split=train_split,
            test_dataset=test_dataset,
            test_split=test_split
        )
        self.__save_log_probs_for_train_and_test(dataset_path=dataset_path, lm_name=OFFLINE_RESPONSES_NAME,
                                                 type_of_generations=OFFLINE_RESPONSES_NAME,
                                                 train_logprobs=train_offline_log_probs,
                                                 train_response_lengths=train_offline_response_lengths,
                                                 train_prompt_indices=train_prompt_indices,
                                                 train_split=train_split,
                                                 test_logprobs=test_offline_log_probs,
                                                 test_response_lengths=test_offline_response_lengths,
                                                 test_prompt_indices=test_prompt_indices, test_split=test_split)

        # Get and save per prompt reward metrics for the OFFLINE responses in the dataset (already precomputed in dataset generation)
        train_offline_gold_reward_data, test_offline_gold_reward_data = None, None
        if not self.config.rm_eval_config.only_lm_eval:
            if train_split:
                train_offline_gold_reward_data = self.__log_reward_metrics_and_save_rewards_for_gt_reward(rm_wrapper=None,
                                                                                                          dataset_path=dataset_path,
                                                                                                          dataset=train_dataset,
                                                                                                          split=train_split,
                                                                                                          prompt_indices=train_prompt_indices,
                                                                                                          is_offline=True)

            if test_split:
                test_offline_gold_reward_data = self.__log_reward_metrics_and_save_rewards_for_gt_reward(rm_wrapper=None,
                                                                                                         dataset_path=dataset_path,
                                                                                                         dataset=test_dataset,
                                                                                                         split=test_split,
                                                                                                         prompt_indices=test_prompt_indices,
                                                                                                         is_offline=True)

        # Load language model and generate 'num_return_sequences' responses per prompt.
        (train_online_prompt_response_dataset,
         test_online_prompt_response_dataset,
         train_online_log_probs, test_online_log_probs,
         train_online_response_lengths, test_online_response_lengths) = self.__create_online_responses_datasets(dataset_name=dataset_path,
                                                                                                                train_split=train_split,
                                                                                                                train_dataset=train_dataset,
                                                                                                                train_prompt_indices=train_prompt_indices,
                                                                                                                test_split=test_split,
                                                                                                                test_dataset=test_dataset,
                                                                                                                test_prompt_indices=test_prompt_indices)

        self.__save_log_probs_for_train_and_test(dataset_path=dataset_path, lm_name=self.lm_name, type_of_generations=ONLINE_RESPONSES_NAME,
                                                 train_logprobs=train_online_log_probs,
                                                 train_response_lengths=train_online_response_lengths,
                                                 train_prompt_indices=train_prompt_indices,
                                                 train_split=train_split,
                                                 test_logprobs=test_online_log_probs,
                                                 test_response_lengths=test_online_response_lengths,
                                                 test_prompt_indices=test_prompt_indices,
                                                 test_split=test_split)

        # Load ground truth reward model only after generating responses and removing the language model from the GPU
        gt_rm_wrapper = get_reward_model_wrapper(self.config.rm_eval_config.ground_truth_reward_model_path, device=self.device,
                                                 cache_dir=self.config.cache_dir, logger=self.logger)

        # Compute ground truth rewards for ONLINE data for train split
        train_online_gold_reward_data, test_online_gold_reward_data = None, None
        if train_split:
            train_online_gold_reward_data = self.__log_reward_metrics_and_save_rewards_for_gt_reward(
                rm_wrapper=gt_rm_wrapper,
                dataset_path=dataset_path,
                dataset=train_online_prompt_response_dataset,
                split=train_split,
                prompt_indices=train_prompt_indices,
                is_offline=False
            )

        # Compute ground truth rewards for ONLINE data for test split
        if test_split:
            test_online_gold_reward_data = self.__log_reward_metrics_and_save_rewards_for_gt_reward(
                rm_wrapper=gt_rm_wrapper,
                dataset_path=dataset_path,
                dataset=test_online_prompt_response_dataset,
                split=test_split,
                prompt_indices=test_prompt_indices,
                is_offline=False
            )

        del gt_rm_wrapper
        torch.cuda.empty_cache()
        gc.collect()

        # Compute and log accuracy measures for the ground truth reward (as a sanity check)
        self.__compute_and_log_accuracy_measures_for_gt_reward_and_all_datasets(dataset_path=dataset_path,
                                                                                test_offline_gold_reward_data=test_offline_gold_reward_data,
                                                                                test_offline_log_probs=test_offline_log_probs,
                                                                                test_offline_response_lengths=test_offline_response_lengths,
                                                                                test_online_gold_reward_data=test_online_gold_reward_data,
                                                                                test_online_log_probs=test_online_log_probs,
                                                                                test_online_response_lengths=test_online_response_lengths,
                                                                                test_prompt_indices=test_prompt_indices,
                                                                                test_split=test_split,
                                                                                train_offline_gold_reward_data=train_offline_gold_reward_data,
                                                                                train_offline_log_probs=train_offline_log_probs,
                                                                                train_offline_response_lengths=train_offline_response_lengths,
                                                                                train_online_gold_reward_data=train_online_gold_reward_data,
                                                                                train_online_log_probs=train_online_log_probs,
                                                                                train_online_response_lengths=train_online_response_lengths,
                                                                                train_prompt_indices=train_prompt_indices,
                                                                                train_split=train_split)

        train_metrics = SplitMetrics(
            dataset=train_dataset,
            dataset_name=dataset_path,
            prompt_indices=train_prompt_indices,
            split_name=train_split,
            offline_gold_rewards=train_offline_gold_reward_data.gold_rewards if train_offline_gold_reward_data is not None else None,
            online_gold_rewards=train_online_gold_reward_data.gold_rewards if train_online_gold_reward_data is not None else None,
            offline_log_probs=ProbabilityMetrics(
                log_probs=train_offline_log_probs,
                sequence_lengths=train_offline_response_lengths
            ),
            online_log_probs=ProbabilityMetrics(
                log_probs=train_online_log_probs,
                sequence_lengths=train_online_response_lengths
            )
        )

        test_metrics = SplitMetrics(
            dataset=test_dataset,
            dataset_name=dataset_path,
            prompt_indices=test_prompt_indices,
            split_name=test_split,
            offline_gold_rewards=test_offline_gold_reward_data.gold_rewards if test_offline_gold_reward_data is not None else None,
            online_gold_rewards=test_online_gold_reward_data.gold_rewards if test_online_gold_reward_data is not None else None,
            offline_log_probs=ProbabilityMetrics(
                log_probs=test_offline_log_probs,
                sequence_lengths=test_offline_response_lengths
            ),
            online_log_probs=ProbabilityMetrics(
                log_probs=test_online_log_probs,
                sequence_lengths=test_online_response_lengths
            )
        )

        for rm_str in self.config.rm_eval_config.proxy_reward_models:
            if rm_str == self.config.rm_eval_config.ground_truth_reward_model_path:
                # If the gold reward model is passed as one of the proxy models as well, no need to evaluate again
                continue

            rm_wrapper = get_reward_model_wrapper(rm_str, device=self.device, cache_dir=self.config.cache_dir, logger=self.logger)
            rm_name = self.__extract_name_from_path(rm_str)
            self.__compute_and_save_rewards_and_accuracies_for_rm(
                train_online_prompt_response_dataset,
                test_online_prompt_response_dataset,
                rm_name,
                rm_wrapper,
                train_metrics,
                test_metrics
            )

    def __run_eval_for_dataset_multiple_rejected(self, dataset_path: str, train_split: str, test_split: str,
                                                 train_dataset, train_prompt_indices, test_dataset, test_prompt_indices):
        """
        Evaluation method for datasets with multiple rejected responses.
        Uses one_vs_many accuracy variants and only uses generated responses for reward means.
        """
        self.logger.info(f"STARTING EVALUATION USING DATASET: {dataset_path} (MULTIPLE REJECTED RESPONSES)")
        train_offline_gt_chosen_rewards, train_offline_gt_rejected_rewards = None, None
        test_offline_gt_chosen_rewards, test_offline_gt_rejected_rewards = None, None

        if train_split and train_dataset is not None:
            train_offline_gt_chosen_rewards, train_offline_gt_rejected_rewards = self.__log_reward_metrics_and_save_rewards_for_gt_reward_multiple_rejected(
                dataset_path=dataset_path, dataset=train_dataset, split=train_split, prompt_indices=train_prompt_indices
            )

        if test_split and test_dataset is not None:
            test_offline_gt_chosen_rewards, test_offline_gt_rejected_rewards = self.__log_reward_metrics_and_save_rewards_for_gt_reward_multiple_rejected(
                dataset_path=dataset_path, dataset=test_dataset, split=test_split, prompt_indices=test_prompt_indices
            )

        # Generate online responses for computing reward means (not for accuracy)
        train_online_prompt_response_dataset, test_online_prompt_response_dataset, _, _, _, _ = self.__create_online_responses_datasets(
            dataset_name=dataset_path,
            train_split=train_split,
            train_dataset=train_dataset,
            train_prompt_indices=train_prompt_indices,
            test_split=test_split,
            test_dataset=test_dataset,
            test_prompt_indices=test_prompt_indices
        )

        # Compute ground truth rewards for online responses (for reward means only)
        gt_rm_wrapper = get_reward_model_wrapper(self.config.rm_eval_config.ground_truth_reward_model_path, device=self.device,
                                                 cache_dir=self.config.cache_dir, logger=self.logger)

        per_prompt_train_online_gt_reward_mean, per_prompt_test_online_gt_reward_mean = None, None

        if train_split and train_online_prompt_response_dataset is not None:
            train_online_gt_rewards = self.__compute_per_prompt_rewards(rm_wrapper=gt_rm_wrapper, dataset=train_online_prompt_response_dataset)
            per_prompt_train_online_gt_reward_mean = train_online_gt_rewards.mean(dim=1)

        if test_split and test_online_prompt_response_dataset is not None:
            test_online_gt_rewards = self.__compute_per_prompt_rewards(rm_wrapper=gt_rm_wrapper, dataset=test_online_prompt_response_dataset)
            per_prompt_test_online_gt_reward_mean = test_online_gt_rewards.mean(dim=1)

        del gt_rm_wrapper
        torch.cuda.empty_cache()
        gc.collect()

        lm_wrapper = self.__get_lm_wrapper(self.config.rm_eval_config.language_model_path,
                                           device=self.device,
                                           config=self.config.rm_eval_config)

        train_offline_log_probs, train_offline_lengths = None, None
        test_offline_log_probs, test_offline_lengths = None, None

        if train_split and train_dataset is not None:
            train_offline_log_probs, train_offline_lengths = self.__compute_offline_log_probs_for_dataset_multiple_rejected(
                lm_wrapper, train_dataset, train_split
            )

        if test_split and test_dataset is not None:
            test_offline_log_probs, test_offline_lengths = self.__compute_offline_log_probs_for_dataset_multiple_rejected(
                lm_wrapper, test_dataset, test_split
            )

        self.__save_log_probs_for_train_and_test(
            dataset_path=dataset_path,
            lm_name=OFFLINE_RESPONSES_NAME,
            type_of_generations=OFFLINE_RESPONSES_NAME,
            train_logprobs=train_offline_log_probs,
            train_response_lengths=train_offline_lengths,
            train_prompt_indices=train_prompt_indices,
            train_split=train_split,
            test_logprobs=test_offline_log_probs,
            test_response_lengths=test_offline_lengths,
            test_prompt_indices=test_prompt_indices,
            test_split=test_split
        )

        del lm_wrapper
        torch.cuda.empty_cache()
        gc.collect()

        # Compute accuracy using offline rewards (one_vs_many)
        if train_offline_gt_chosen_rewards is not None:
            train_offline_metadata = EvaluationMetadata(
                lm_name=OFFLINE_RESPONSES_NAME,
                lm_name_for_probs=self.lm_name,
                type_of_generations=OFFLINE_RESPONSES_NAME,
                rm_name=self.gold_rm_name,
                dataset_name=dataset_path,
                split=train_split,
                prompt_indices=train_prompt_indices,
                lm_file_name=OFFLINE_RESPONSES_NAME
            )
            train_offline_prob_metrics = ProbabilityMetrics(
                log_probs=train_offline_log_probs,
                sequence_lengths=train_offline_lengths
            )
            self.__compute_and_log_accuracy_measures_one_vs_many(
                metadata=train_offline_metadata,
                chosen_rewards=train_offline_gt_chosen_rewards,
                rejected_rewards=train_offline_gt_rejected_rewards,
                per_prompt_online_rm_reward_mean=per_prompt_train_online_gt_reward_mean,
                prob_metrics=train_offline_prob_metrics
            )

        if test_offline_gt_chosen_rewards is not None:
            test_offline_metadata = EvaluationMetadata(
                lm_name=OFFLINE_RESPONSES_NAME,
                lm_name_for_probs=self.lm_name,
                type_of_generations=OFFLINE_RESPONSES_NAME,
                rm_name=self.gold_rm_name,
                dataset_name=dataset_path,
                split=test_split,
                prompt_indices=test_prompt_indices,
                lm_file_name=OFFLINE_RESPONSES_NAME
            )
            test_offline_prob_metrics = ProbabilityMetrics(
                log_probs=test_offline_log_probs,
                sequence_lengths=test_offline_lengths
            )
            self.__compute_and_log_accuracy_measures_one_vs_many(
                metadata=test_offline_metadata,
                chosen_rewards=test_offline_gt_chosen_rewards,
                rejected_rewards=test_offline_gt_rejected_rewards,
                per_prompt_online_rm_reward_mean=per_prompt_test_online_gt_reward_mean,
                prob_metrics=test_offline_prob_metrics
            )

        # Evaluate proxy reward models
        for rm_str in self.config.rm_eval_config.proxy_reward_models:
            if rm_str == self.config.rm_eval_config.ground_truth_reward_model_path:
                continue

            rm_wrapper = get_reward_model_wrapper(rm_str, device=self.device, cache_dir=self.config.cache_dir, logger=self.logger)
            rm_name = self.__extract_name_from_path(rm_str)

            # Compute rewards for online responses using proxy RM (for reward means)
            per_prompt_train_online_rm_reward_mean, per_prompt_test_online_rm_reward_mean = None, None

            if train_split:
                self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for ONLINE TRAIN responses")
                train_online_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                            dataset=train_online_prompt_response_dataset)
                train_online_rm_metadata = EvaluationMetadata(
                    lm_name=self.lm_name,
                    lm_name_for_probs=self.lm_name,
                    type_of_generations=ONLINE_RESPONSES_NAME,
                    rm_name=rm_name,
                    dataset_name=dataset_path,
                    split=train_split,
                    prompt_indices=train_prompt_indices,
                    lm_file_name=self.lm_file_name
                )
                self.__save_rewards_and_log_reward_metrics(per_prompt_rewards=train_online_rm_rewards, metadata=train_online_rm_metadata,
                                                           is_one_vs_many_dataset=True)
                per_prompt_train_online_rm_reward_mean = train_online_rm_rewards.mean(dim=1)

            if test_split:
                self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for ONLINE TEST responses")
                test_online_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                           dataset=test_online_prompt_response_dataset)
                test_online_rm_metadata = EvaluationMetadata(
                    lm_name=self.lm_name,
                    lm_name_for_probs=self.lm_name,
                    type_of_generations=ONLINE_RESPONSES_NAME,
                    rm_name=rm_name,
                    dataset_name=dataset_path,
                    split=test_split,
                    prompt_indices=test_prompt_indices,
                    lm_file_name=self.lm_file_name
                )
                self.__save_rewards_and_log_reward_metrics(per_prompt_rewards=test_online_rm_rewards, metadata=test_online_rm_metadata,
                                                           is_one_vs_many_dataset=True)
                per_prompt_test_online_rm_reward_mean = test_online_rm_rewards.mean(dim=1)

            # Compute rewards for offline data using proxy RM
            if train_offline_gt_chosen_rewards is not None:
                self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for OFFLINE TRAIN responses")
                train_proxy_chosen_rewards = self.__compute_chosen_rewards_for_multiple_rejected(rm_wrapper, train_dataset)
                train_proxy_rejected_rewards = self.__compute_rejected_rewards_for_multiple_rejected(rm_wrapper, train_dataset)

                train_proxy_offline_metadata = EvaluationMetadata(
                    lm_name=OFFLINE_RESPONSES_NAME,
                    lm_name_for_probs=self.lm_name,
                    type_of_generations=OFFLINE_RESPONSES_NAME,
                    rm_name=rm_name,
                    dataset_name=dataset_path,
                    split=train_split,
                    prompt_indices=train_prompt_indices,
                    lm_file_name=OFFLINE_RESPONSES_NAME
                )
                self.__save_rewards_and_log_reward_metrics_one_vs_many(
                    chosen_rewards=train_proxy_chosen_rewards,
                    rejected_rewards=train_proxy_rejected_rewards,
                    metadata=train_proxy_offline_metadata
                )

                train_proxy_metadata = EvaluationMetadata(
                    lm_name=OFFLINE_RESPONSES_NAME,
                    lm_name_for_probs=self.lm_name,
                    type_of_generations=OFFLINE_RESPONSES_NAME,
                    rm_name=rm_name,
                    dataset_name=dataset_path,
                    split=train_split,
                    prompt_indices=train_prompt_indices,
                    lm_file_name=OFFLINE_RESPONSES_NAME
                )
                train_proxy_prob_metrics = ProbabilityMetrics(
                    log_probs=train_offline_log_probs,
                    sequence_lengths=train_offline_lengths
                )
                self.__compute_and_log_accuracy_measures_one_vs_many(
                    metadata=train_proxy_metadata,
                    chosen_rewards=train_proxy_chosen_rewards,
                    rejected_rewards=train_proxy_rejected_rewards,
                    per_prompt_online_rm_reward_mean=per_prompt_train_online_rm_reward_mean,
                    prob_metrics=train_proxy_prob_metrics
                )

            if test_offline_gt_chosen_rewards is not None:
                self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for OFFLINE TEST responses")
                test_proxy_chosen_rewards = self.__compute_chosen_rewards_for_multiple_rejected(rm_wrapper, test_dataset)
                test_proxy_rejected_rewards = self.__compute_rejected_rewards_for_multiple_rejected(rm_wrapper, test_dataset)

                # Save RM rewards for offline responses
                test_proxy_offline_metadata = EvaluationMetadata(
                    lm_name=OFFLINE_RESPONSES_NAME,
                    lm_name_for_probs=self.lm_name,
                    type_of_generations=OFFLINE_RESPONSES_NAME,
                    rm_name=rm_name,
                    dataset_name=dataset_path,
                    split=test_split,
                    prompt_indices=test_prompt_indices,
                    lm_file_name=OFFLINE_RESPONSES_NAME
                )
                self.__save_rewards_and_log_reward_metrics_one_vs_many(
                    chosen_rewards=test_proxy_chosen_rewards,
                    rejected_rewards=test_proxy_rejected_rewards,
                    metadata=test_proxy_offline_metadata
                )

                test_proxy_metadata = EvaluationMetadata(
                    lm_name=OFFLINE_RESPONSES_NAME,
                    lm_name_for_probs=self.lm_name,
                    type_of_generations=OFFLINE_RESPONSES_NAME,
                    rm_name=rm_name,
                    dataset_name=dataset_path,
                    split=test_split,
                    prompt_indices=test_prompt_indices,
                    lm_file_name=OFFLINE_RESPONSES_NAME
                )
                test_proxy_prob_metrics = ProbabilityMetrics(
                    log_probs=test_offline_log_probs,
                    sequence_lengths=test_offline_lengths
                )
                self.__compute_and_log_accuracy_measures_one_vs_many(
                    metadata=test_proxy_metadata,
                    chosen_rewards=test_proxy_chosen_rewards,
                    rejected_rewards=test_proxy_rejected_rewards,
                    per_prompt_online_rm_reward_mean=per_prompt_test_online_rm_reward_mean,
                    prob_metrics=test_proxy_prob_metrics
                )

    def __compute_chosen_rewards_for_multiple_rejected(self, rm_wrapper, dataset):
        """Compute rewards for chosen responses in a dataset with multiple rejected."""
        rewards = []
        for example in dataset:
            chosen_conv = example['chosen']
            chosen_text = rm_wrapper.tokenizer.apply_chat_template(chosen_conv, tokenize=False, enable_thinking=False)
            chosen_input = rm_wrapper.tokenizer(chosen_text, padding=True, truncation=False, add_special_tokens=False, return_tensors='pt').to(
                self.device)
            reward = rm_wrapper.compute_batch_rewards(chosen_input).cpu().numpy()[0]
            rewards.append(float(reward))

        return torch.tensor(rewards, device=self.device)

    def __compute_rejected_rewards_for_multiple_rejected(self, rm_wrapper, dataset):
        """Compute rewards for all rejected responses in a dataset with multiple rejected."""
        all_rejected_rewards = []
        max_rejected = 0

        for example in dataset:
            rejected_convs = example['rejected']
            rejected_rewards = []
            for rejected_conv in rejected_convs:
                rejected_text = rm_wrapper.tokenizer.apply_chat_template(rejected_conv, tokenize=False, enable_thinking=False)
                rejected_input = rm_wrapper.tokenizer(rejected_text, padding=True, truncation=False, add_special_tokens=False,
                                                      return_tensors='pt').to(self.device)
                reward = rm_wrapper.compute_batch_rewards(rejected_input).cpu().numpy()[0]
                rejected_rewards.append(float(reward))

            all_rejected_rewards.append(rejected_rewards)
            max_rejected = max(max_rejected, len(rejected_rewards))

        # Pad to same length using -inf since rewards can be negative
        for rewards in all_rejected_rewards:
            rewards.extend([float('-inf')] * (max_rejected - len(rewards)))

        return torch.tensor(all_rejected_rewards, device=self.device)

    @torch.no_grad()
    def run(self, **kwargs):
        start_time = datetime.now(timezone.utc)
        try:
            rm_eval_config = self.config.rm_eval_config
            for dataset_path, load_dataset_from_file, train_split, test_split, num_train_samples, num_test_samples in zip(
                    rm_eval_config.dataset_paths,
                    rm_eval_config.load_dataset_from_file_flags,
                    rm_eval_config.train_splits,
                    rm_eval_config.test_splits,
                    rm_eval_config.num_train_samples,
                    rm_eval_config.num_test_samples
            ):
                # Prepare datasets to check if they have multiple rejected responses
                train_dataset, train_prompt_indices, test_dataset, test_prompt_indices = self.__prepare_train_and_test_datasets(
                    dataset_path, load_dataset_from_file, train_split, test_split, num_train_samples, num_test_samples
                )

                if self.__dataset_has_multiple_rejected(train_dataset, test_dataset):
                    self.__run_eval_for_dataset_multiple_rejected(
                        dataset_path, train_split, test_split,
                        train_dataset, train_prompt_indices, test_dataset, test_prompt_indices
                    )
                else:
                    self.__run_eval_for_dataset(
                        dataset_path, train_split, test_split,
                        train_dataset, train_prompt_indices, test_dataset, test_prompt_indices
                    )

        except Exception:
            self.logger.exception("Exception while running evaluation script.")
            raise
        finally:
            end_time = datetime.now(timezone.utc)
            self.logger.info(f"Finished running evaluation script. Time took: {end_time - start_time}")
