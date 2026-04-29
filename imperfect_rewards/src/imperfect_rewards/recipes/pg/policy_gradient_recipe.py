import logging
import os

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from imperfect_rewards.data.utils.rm_utils import compute_reward_normalization, get_reward_model_wrapper
from imperfect_rewards.recipes import BaseRecipe
from imperfect_rewards.recipes.pg.algorithm_registry import ALGORITHM_TRAINER_CLASSES
from imperfect_rewards.utils import is_main_process, get_logger, DEFAULT_EOS_TOKEN, DEFAULT_PADDING_TOKEN, update_tokenizer, \
    update_model_num_embeddings_and_special_tokens, DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN
from imperfect_rewards.utils.strings import PER_PROMPT_REWARDS

logger = get_logger(__name__)
if not is_main_process():
    logger.setLevel(logging.WARNING)

from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback


class UpdateCheckpointPathCallback(TrainerCallback):
    """Custom callback to track checkpoint paths when they are saved."""

    def __init__(self, pg_recipe):
        self.pg_recipe = pg_recipe

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.pg_recipe.checkpoint_paths.append(checkpoint_path)
        logger.info(f"Updated checkpoint paths with: {checkpoint_path}")


class PolicyGradientRecipe(BaseRecipe):
    def __init__(self, config):
        super().__init__(config)
        self.train_dataset = None
        self.eval_dataset = None
        self.num_added_toks = 0
        self.policy_dir = os.path.join(self.config.pg_config.output_dir, "policy_model")
        self.checkpoint_paths = []
        self.torch_dtype = torch.bfloat16 if self.config.pg_config.bf16 else (torch.float16 if self.config.pg_config.fp16 else torch.float32)

        logger.info(f"Initialized PolicyGradientRecipe with config: {self.config}")

    def __prepare_tokenizer(self) -> None:
        """In TRL - RLOO, it seems that the tokenizer of the reward needs to be the same as the policy
        (because we get the rewards by directly passing the tokenized sequence instead of using the tokenizer of the reward model.
        See https://github.com/huggingface/trl/blob/b2696578ce6db1749a250661b507bf8b90e14dd5/trl/trainer/rloo_trainer.py#L345)

        For now, in order to add reward var and mean calculation, I am adding the ability to load a different type of RM and tokenizer; this will replace the one from the policy. 
        This mechanism will only be applied when reward_var_analysis is True, meaning that there will be no training using this mechanism.

        In the future, we need to add the possibility to add a RM different from the policy ...
        """

        self.policy_tokenizer = AutoTokenizer.from_pretrained(
            self.config.pg_config.language_model_path,
            trust_remote_code=self.config.pg_config.trust_remote_code,
            cache_dir=self.config.cache_dir,
        )
        self.policy_tokenizer, self.num_added_toks = update_tokenizer(self.policy_tokenizer, self.num_added_toks, DEFAULT_PADDING_TOKEN,
                                                                      DEFAULT_EOS_TOKEN, logger, user_token=DEFAULT_USER_TOKEN,
                                                                      assistant_token=DEFAULT_ASSISTANT_TOKEN)

    def __load_dataset(self, dataset_path: str):
        if not self.config.pg_config.load_dataset_from_file:
            dataset = load_dataset(dataset_path, trust_remote_code=True, cache_dir=self.config.cache_dir)
        else:
            dataset = load_from_disk(dataset_path)
        return dataset

    def __prepare_dataset(self, **kwargs):
        '''
        This function is mostly taken from
        https://github.com/huggingface/trl/blob/b2696578ce6db1749a250661b507bf8b90e14dd5/examples/scripts/ppo/ppo.py#L99
        '''
        dataset = self.__load_dataset(self.config.pg_config.dataset_path)
        self.train_dataset = dataset[self.config.pg_config.dataset_train_split]
        self.orig_train_sample_indices = torch.arange(len(self.train_dataset))
        if self.config.pg_config.num_train_samples > 0:
            if self.config.pg_config.data_selection_seed > 0:
                perm = np.random.RandomState(seed=self.config.pg_config.data_selection_seed).permutation(len(self.train_dataset))
            else:
                perm = np.random.permutation(len(self.train_dataset))

            num_train_samples = min(self.config.pg_config.num_train_samples, len(self.train_dataset))
            self.train_dataset = self.train_dataset.select(perm[:num_train_samples])
            self.orig_train_sample_indices = self.orig_train_sample_indices[perm[:num_train_samples]]

        self.eval_dataset = dataset.get(self.config.pg_config.dataset_test_split)
        self.train_dataset = self.train_dataset.select_columns(["prompt"])
        if self.eval_dataset is not None:
            self.eval_dataset = self.eval_dataset.select_columns(["prompt"])

    def compute_reward_shift_and_scale(self):
        if not self.config.pg_config.path_to_precomputed_rewards_for_normalization:
            return 0, 1

        precomputed_rewards_info = torch.load(self.config.pg_config.path_to_precomputed_rewards_for_normalization)
        rewards = precomputed_rewards_info[PER_PROMPT_REWARDS]
        return compute_reward_normalization(rewards)

    def run(self, **kwargs):
        try:
            self.__prepare_dataset()
            self.__prepare_tokenizer()

            rm_wrapper = get_reward_model_wrapper(reward_model_name=self.config.pg_config.reward_model_path,
                                                  cache_dir=self.config.cache_dir)
            self.reward_model = rm_wrapper.rm
            self.reward_model.eval()

            if self.config.pg_config.use_reward_model_tokenizer:
                self.reward_model_tokenizer = rm_wrapper.tokenizer
            else:
                self.reward_model_tokenizer = None

            reward_shift, reward_scale = self.compute_reward_shift_and_scale()
            self.reward_model.config.reward_shift = reward_shift
            self.reward_model.config.reward_scale = reward_scale

            self.policy = AutoModelForCausalLM.from_pretrained(
                self.config.pg_config.language_model_path,
                trust_remote_code=self.config.pg_config.trust_remote_code,
                torch_dtype=self.torch_dtype,
                cache_dir=self.config.cache_dir
            )

            if self.config.pg_config.use_lora:
                # Using lora_alpha = 2 * lora_rank is according to Llama 3 8B recipe for LoRA finetuning:
                # https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora_single_device.yaml
                lora_config = LoraConfig(r=self.config.pg_config.lora_rank, lora_alpha=self.config.pg_config.lora_rank * 2,
                                         bias="all", target_modules="all-linear", use_rslora=True)
                self.policy = get_peft_model(self.policy, lora_config)

            self.ref_policy = AutoModelForCausalLM.from_pretrained(
                self.config.pg_config.language_model_path,
                trust_remote_code=self.config.pg_config.trust_remote_code,
                torch_dtype=self.torch_dtype,
                cache_dir=self.config.cache_dir
            )
            self.ref_policy.eval()

            trainer_class = ALGORITHM_TRAINER_CLASSES.get(self.config.pg_config.algorithm_name)

            if self.num_added_toks > 0:
                self.policy = update_model_num_embeddings_and_special_tokens(self.policy, self.policy_tokenizer)
                self.ref_policy = update_model_num_embeddings_and_special_tokens(self.ref_policy, self.policy_tokenizer)
                if self.reward_model_tokenizer is None:
                    self.reward_model.resize_token_embeddings(len(self.policy_tokenizer))

            if self.policy_tokenizer.model_max_length > 10000:
                self.policy_tokenizer.model_max_length = self.policy.config.max_position_embeddings
            self.config.pg_config.algo_specific.model_max_length = self.policy_tokenizer.model_max_length

            self.config.pg_config.algo_specific.stop_token_id = self.policy_tokenizer.eos_token_id  # Ensures that tokens after EOS during generation are set to padding tokens

            # Count trainable parameters before trainer creation (which may wrap the model with accelerate)
            num_trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
            algorithm_name = self.config.pg_config.algorithm_name

            if self.config.pg_config.algorithm_name == "PPO":
                value_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.pg_config.language_model_path,
                    num_labels=1,
                    trust_remote_code=self.config.pg_config.trust_remote_code,
                    torch_dtype=self.torch_dtype,
                    cache_dir=self.config.cache_dir
                )
                update_model_num_embeddings_and_special_tokens(value_model, self.policy_tokenizer)

                self.trainer = trainer_class(
                    config=self.config.pg_config.algo_specific,
                    processing_class=self.policy_tokenizer,
                    policy=self.policy,
                    ref_policy=self.ref_policy,
                    reward_model=self.reward_model,
                    reward_model_tokenizer=self.reward_model_tokenizer,
                    value_model=value_model,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset
                )
            else:
                self.trainer = trainer_class(
                    config=self.config.pg_config.algo_specific,
                    processing_class=self.policy_tokenizer,
                    policy=self.policy,
                    ref_policy=self.ref_policy,
                    reward_model=self.reward_model,
                    reward_model_tokenizer=self.reward_model_tokenizer,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset
                )

            self.trainer.add_callback(UpdateCheckpointPathCallback(self))

            logger.info(f"Starting policy gradient ({algorithm_name}) training...\n"
                        f"Number of trainable parameters in policy model: {num_trainable_params}\n"
                        f"Number of training samples: {len(self.train_dataset)}\n"
                        f"Number of test samples: {len(self.eval_dataset) if self.eval_dataset is not None else 0}")

            self.trainer.train()
            logger.info(f"Finished policy gradient training")
        except Exception:
            logger.exception("Exception while running policy gradient training script.")
            raise
