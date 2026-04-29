import json
from typing import Union

import json
from typing import Union

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, IterableDatasetDict, Dataset, IterableDataset, load_from_disk
from transformers import AutoTokenizer

from imperfect_rewards.data.utils import CausalLMWrapper
from imperfect_rewards.data.utils.rm_utils import get_reward_model_wrapper
from imperfect_rewards.utils import DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PADDING_TOKEN, update_tokenizer, \
    update_model_num_embeddings_and_special_tokens
from imperfect_rewards.utils import single_process_logging as logging_utils
from imperfect_rewards.utils.strings import DEFAULT_TRAIN_RLHF_SPLIT_NAME, DEFAULT_TEST_SPLIT_NAME


class PreferenceDatasetCreator:

    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.device = torch.device(f"cuda:{self.config.gpu_id}" if torch.cuda.is_available() and self.config.gpu_id >= 0 else "cpu")

        if self.config.reward_model_path:
            self.rm_wrapper = get_reward_model_wrapper(self.config.reward_model_path, device=self.device, cache_dir=self.config.cache_dir)

            # If the reward model does not have a chat template, adds a default one for debugging purposes (chat template needs to already exist)
            if not self.rm_wrapper.tokenizer.chat_template:
                logging_utils.warning("Reward model does not have a chat template. "
                                      "Adding a default one, which should only be used for debugging purposes.")
                update_tokenizer(tokenizer=self.rm_wrapper.tokenizer, num_added_toks=0, pad_token=DEFAULT_PADDING_TOKEN,
                                 eos_token=DEFAULT_EOS_TOKEN, logger=logging_utils.get_default_logger(), user_token=DEFAULT_USER_TOKEN,
                                 assistant_token=DEFAULT_ASSISTANT_TOKEN)
                self.rm_wrapper.rm.config.pad_token_id = self.rm_wrapper.tokenizer.pad_token_id
                update_model_num_embeddings_and_special_tokens(self.rm_wrapper.rm, self.rm_wrapper.tokenizer)

        self.train_test_splits = [split for split in [self.config.train_split, self.config.test_split] if split is not None]

        if self.config.tokenizer_for_length_filtering:
            self.tokenizer_for_length_filtering = AutoTokenizer.from_pretrained(self.config.tokenizer_for_length_filtering,
                                                                                use_fast=True,
                                                                                trust_remote_code=True,
                                                                                cache_dir=self.config.cache_dir)
        else:
            self.tokenizer_for_length_filtering = self.rm_wrapper.tokenizer

    @staticmethod
    def __get_lm_wrapper(lm_str: str, device, config):
        try:
            lm_wrapper = CausalLMWrapper(lm_str, device=device, config=config)
            lm_wrapper.prepare()
            return lm_wrapper
        except Exception as e:
            logging_utils.error(str(e))
            raise Exception(e)

    @staticmethod
    def __has_multiple_rejected_responses(rejected):
        """Check if rejected is a list of conversations (multiple rejected) vs a single conversation."""
        if not isinstance(rejected, list) or len(rejected) == 0:
            return False
        # If the first element is a list, we have multiple rejected responses
        # If the first element is a dict, we have a single rejected response
        return isinstance(rejected[0], list)

    def __filter_by_length(self, example):
        prompt_not_too_long = False
        if self.config.max_prompt_length > 0:
            prompt_length = len(self.tokenizer_for_length_filtering.encode(example["prompt"]))
            prompt_not_too_long = prompt_length <= self.config.max_prompt_length

        responses_not_too_long = False
        if self.config.max_response_length > 0:
            chosen_length = len(self.tokenizer_for_length_filtering.encode(example["chosen"][1]["content"]))
            
            if self.__has_multiple_rejected_responses(example["rejected"]):
                rejected_lengths = [len(self.tokenizer_for_length_filtering.encode(rej[1]["content"])) 
                                   for rej in example["rejected"]]
                rejected_not_too_long = all(length <= self.config.max_response_length for length in rejected_lengths)
            else:
                rejected_length = len(self.tokenizer_for_length_filtering.encode(example["rejected"][1]["content"]))
                rejected_not_too_long = rejected_length <= self.config.max_response_length
            
            responses_not_too_long = chosen_length <= self.config.max_response_length and rejected_not_too_long

        return prompt_not_too_long and responses_not_too_long

    def __subsample_training_set(self):
        if self.config.train_split_seed > 0:
            perm = np.random.RandomState(seed=self.config.train_split_seed + 1).permutation(len(self.dataset[self.config.train_split]))
        else:
            perm = np.random.permutation(len(self.dataset[self.config.train_split]))

        num_train_samples = min(self.config.num_train_samples_from_initial_dataset, len(self.dataset[self.config.train_split]))
        self.dataset[self.config.train_split] = self.dataset[self.config.train_split].select(perm[:num_train_samples])
        logging_utils.info(f"Dataset properties after choosing {self.config.num_train_samples_from_initial_dataset} "
                           f"samples from the initial training split")
        self.log_dataset_properties()

    @torch.no_grad()
    def prepare_dataset(self):
        if not self.config.load_dataset_from_file:
            self.dataset = load_dataset(self.config.initial_dataset_path,
                                        cache_dir=self.config.cache_dir)
        else:
            self.dataset = load_from_disk(self.config.initial_dataset_path)

        logging_utils.info(f"Logging config: {self.config}")

        self.__delete_irrelevant_splits()

        # If score chosen and score rejected do not exist in the dataset, create them and initialize to 1 for chosen, 0 for rejected
        for split in self.train_test_splits:
            num_rows = self.dataset[split].num_rows
            if self.config.score_chosen_name not in self.dataset[split].column_names:
                chosen_scores = np.full(num_rows, 1.0, dtype=np.float32)
                self.dataset[split] = self.dataset[split].add_column(self.config.score_chosen_name, chosen_scores.tolist())

            if self.config.score_rejected_name not in self.dataset[split].column_names:
                rejected_scores = np.full(num_rows, 0.0, dtype=np.float32)
                self.dataset[split] = self.dataset[split].add_column(self.config.score_rejected_name, rejected_scores.tolist())

            if self.config.prompt_id_name != "prompt_id" and self.config.prompt_id_name in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].rename_column(self.config.prompt_id_name, "prompt_id")

        logging_utils.info("Dataset properties before filtering and relabeling.")
        self.log_dataset_properties()

        # Filter out long prompts/responses
        if self.config.max_prompt_length > 0 or self.config.max_response_length > 0:
            for split in self.train_test_splits:
                self.dataset[split] = self.dataset[split].filter(self.__filter_by_length)

            logging_utils.info(f"Dataset properties after filtering out samples with prompts longer than {self.config.max_prompt_length} tokens, "
                               f"or responses longer than {self.config.max_response_length} tokens")
            self.log_dataset_properties()

        if self.config.num_train_samples_from_initial_dataset > 0:
            self.__subsample_training_set()

        # Compute rewards
        if self.config.reward_model_path:
            # Check if dataset has multiple rejected responses by examining the first example
            has_multiple_rejected = False
            if len(self.train_test_splits) > 0:
                first_split = self.train_test_splits[0]
                first_example = self.dataset[first_split][0]
                has_multiple_rejected = self.__has_multiple_rejected_responses(first_example.get('rejected', []))
            
            if has_multiple_rejected:
                self.dataset = self.__updated_dataset_w_computed_rewards_multiple_rejected(batch_size=self.config.rm_batch_size)
            else:
                self.dataset = self.__updated_dataset_w_computed_rewards(batch_size=self.config.rm_batch_size)

            logging_utils.info(f"Dataset properties after relabeling preferences using reward model: {self.config.reward_model_path}")
            self.log_dataset_properties()

        if self.config.filter_equals:
            for split in self.train_test_splits:
                self.dataset[split] = self.dataset[split].filter(lambda x: x[self.config.score_chosen_name] != x[self.config.score_rejected_name])

            logging_utils.info("Dataset properties after filtering out samples where score_chosen == score_rejected")
            self.log_dataset_properties()

        self.__create_train_and_test_splits()
        self.log_dataset_properties()
        return self.dataset

    def log_dataset_properties(self):
        if self.dataset is None:
            logging_utils.info("Dataset has not yet been initialized")

        num_rows = {split: f"num of rows: {self.dataset[split].num_rows}" for split in self.dataset.keys()}
        score_means = {
            split: [f"mean of {l}: {np.mean(self.dataset[split][l])}" for l in [self.config.score_chosen_name, self.config.score_rejected_name]]
            for split in self.dataset.keys()
        }
        response_lengths = {}
        for split in self.dataset.keys():
            chosen_lengths = [len(self.tokenizer_for_length_filtering.encode(x['chosen'][1]['content'])) for x in self.dataset[split]]
            
            rejected_lengths = []
            for x in self.dataset[split]:
                if self.__has_multiple_rejected_responses(x['rejected']):
                    rejected_lengths.extend([len(self.tokenizer_for_length_filtering.encode(rej[1]['content'])) 
                                            for rej in x['rejected']])
                else:
                    rejected_lengths.append(len(self.tokenizer_for_length_filtering.encode(x['rejected'][1]['content'])))
            
            response_lengths[split] = [
                f"mean chosen response length: {np.mean(chosen_lengths)}",
                f"max chosen response length: {np.max(chosen_lengths)}",
                f"min chosen response length: {np.min(chosen_lengths)}",
                f"mean rejected response length: {np.mean(rejected_lengths)}",
                f"max rejected response length: {np.max(rejected_lengths)}",
                f"min rejected response length: {np.min(rejected_lengths)}"
            ]

        logging_utils.info("=" * 110)
        logging_utils.info("Dataset characteristics:\n%s", json.dumps(num_rows, indent=2))
        logging_utils.info("Dataset characteristics regarding score means:\n%s", json.dumps(score_means, indent=2))
        logging_utils.info("Dataset characteristics regarding response lengths:\n%s", json.dumps(response_lengths, indent=2))

        if not self.config.language_model_path and "was_chosen_rejected_swapped" in self.dataset[next(iter(self.dataset.keys()))].column_names:
            gold_rm_acc = {k: [f"GT RM Accuracy: {1 - np.mean(self.dataset[k]['was_chosen_rejected_swapped'])}"] for k in self.dataset.keys()}
            logging_utils.info("Dataset gold RM accuracy:\n%s", json.dumps(gold_rm_acc, indent=2))

        logging_utils.info("=" * 110 + "\n")

    def __delete_irrelevant_splits(self):
        for split in list(self.dataset.keys()):
            if split not in self.train_test_splits:
                del self.dataset[split]

    @torch.no_grad()
    def __process_batch_with_multiple_rejected(self, batch, tokenizer):
        """
        Process a batch where rejected is a list of conversations (multiple rejected responses).
        Computes rewards for chosen and all rejected, then reselects the one with maximum reward.
        """
        raw_chosen_texts = batch['chosen']
        raw_rejected_texts = batch['rejected']
        batch_size_actual = len(raw_chosen_texts)
        
        new_chosen_texts = []
        new_rejected_texts = []
        new_score_chosen = []
        new_score_rejected = []
        new_was_swapped = []
        
        for i in range(batch_size_actual):
            chosen_conv = raw_chosen_texts[i]
            rejected_convs = raw_rejected_texts[i]
            all_convs = [chosen_conv] + rejected_convs
            
            # Apply chat template to all
            all_texts = [tokenizer.apply_chat_template(conv, tokenize=False, enable_thinking=False) for conv in all_convs]
            
            # Tokenize all
            all_inputs = tokenizer(
                all_texts,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors='pt'
            ).to(self.device)
            
            all_scores = self.rm_wrapper.compute_batch_rewards(all_inputs).cpu().numpy()
            del all_inputs
            max_idx = np.argmax(all_scores)
            
            new_chosen = all_convs[max_idx]
            new_rejected = [conv for idx, conv in enumerate(all_convs) if idx != max_idx]
            
            new_chosen_texts.append(new_chosen)
            new_rejected_texts.append(new_rejected)
            new_score_chosen.append(float(all_scores[max_idx]))
            rejected_scores = [float(all_scores[idx]) for idx in range(len(all_convs)) if idx != max_idx]
            new_score_rejected.append(rejected_scores)
            new_was_swapped.append(max_idx != 0)
        
        batch['chosen'] = new_chosen_texts
        batch['rejected'] = new_rejected_texts
        batch[self.config.score_chosen_name] = new_score_chosen
        batch[self.config.score_rejected_name] = new_score_rejected
        batch['was_chosen_rejected_swapped'] = new_was_swapped
        
        return batch

    def __updated_dataset_w_computed_rewards_multiple_rejected(self, batch_size: int) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        """Process dataset with multiple rejected responses per example."""
        tokenizer = self.rm_wrapper.tokenizer

        @torch.no_grad()
        def process_batch(batch):
            return self.__process_batch_with_multiple_rejected(batch, tokenizer)

        for split in self.train_test_splits:
            logging_utils.info("=" * 110)
            logging_utils.info(f"Relabeling for the following split of {self.config.initial_dataset_path}: {split}")
            logging_utils.info("=" * 110 + "\n")
            self.dataset[split] = self.dataset[split].map(
                process_batch, batched=True, batch_size=batch_size, desc=f"Processing batches in {split}"
            )
            if "messages" in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].remove_columns(["messages"])

        return self.dataset

    def __updated_dataset_w_computed_rewards(self, batch_size: int) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        tokenizer = self.rm_wrapper.tokenizer

        @torch.no_grad()
        def process_batch(batch):
            raw_chosen_texts = batch['chosen']
            raw_rejected_texts = batch['rejected']

            chosen_texts = tokenizer.apply_chat_template(batch['chosen'], tokenize=False, enable_thinking=False)
            rejected_texts = tokenizer.apply_chat_template(batch['rejected'], tokenize=False, enable_thinking=False)

            chosen_inputs = tokenizer(
                chosen_texts,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors='pt'
            ).to(self.device)
            rejected_inputs = tokenizer(
                rejected_texts,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors='pt'
            ).to(self.device)

            score_chosen_np = self.rm_wrapper.compute_batch_rewards(chosen_inputs).cpu().numpy()
            score_rejected_np = self.rm_wrapper.compute_batch_rewards(rejected_inputs).cpu().numpy()
            del rejected_inputs, chosen_inputs

            # Swap chosen and rejected where new score_chosen < new score_rejected
            swap_indices = score_chosen_np < score_rejected_np
            if any(swap_indices):
                # Swap texts
                chosen_array = np.array(raw_chosen_texts, dtype=object)
                rejected_array = np.array(raw_rejected_texts, dtype=object)

                chosen_array[swap_indices], rejected_array[swap_indices] = (
                    rejected_array[swap_indices],
                    chosen_array[swap_indices],
                )

                # Swap scores
                score_chosen_np[swap_indices], score_rejected_np[swap_indices] = (
                    score_rejected_np[swap_indices],
                    score_chosen_np[swap_indices],
                )

                batch['chosen'] = chosen_array.tolist()
                batch['rejected'] = rejected_array.tolist()

            batch[self.config.score_chosen_name] = score_chosen_np
            batch[self.config.score_rejected_name] = score_rejected_np
            batch['was_chosen_rejected_swapped'] = swap_indices.tolist()

            return batch

        for split in self.train_test_splits:
            logging_utils.info("=" * 110)
            logging_utils.info(f"Relabeling for the following split of {self.config.initial_dataset_path}: {split}")
            logging_utils.info("=" * 110 + "\n")
            self.dataset[split] = self.dataset[split].map(
                process_batch, batched=True, batch_size=batch_size, desc=f"Processing batches in {split}"
            )
            if "messages" in self.dataset[split].column_names:
                self.dataset[split] = self.dataset[split].remove_columns(["messages"])

        return self.dataset

    def __create_train_and_test_splits(self):
        if self.config.train_split and self.config.test_split:
            self.dataset = DatasetDict({
                DEFAULT_TRAIN_RLHF_SPLIT_NAME: self.dataset[self.config.train_split],
                DEFAULT_TEST_SPLIT_NAME: self.dataset[self.config.test_split]
            })
            return

        if not self.config.train_split:
            self.dataset = DatasetDict({
                DEFAULT_TEST_SPLIT_NAME: self.dataset[self.config.test_split]
            })
            return

        train_split = self.dataset[self.config.train_split]
        if self.config.frac_train_for_rlhf == 0:
            self.dataset = DatasetDict({
                DEFAULT_TEST_SPLIT_NAME: train_split
            })
            return

        if self.config.frac_train_for_rlhf == 1.0:
            self.dataset = DatasetDict({
                DEFAULT_TRAIN_RLHF_SPLIT_NAME: train_split
            })
            return

        train_and_test_splits = train_split.train_test_split(test_size=1 - float(self.config.frac_train_for_rlhf),
                                                             seed=self.config.train_split_seed)
        train_split, test_split = train_and_test_splits["train"], train_and_test_splits["test"]
        self.dataset = DatasetDict({
            DEFAULT_TRAIN_RLHF_SPLIT_NAME: train_split,
            DEFAULT_TEST_SPLIT_NAME: test_split
        })
