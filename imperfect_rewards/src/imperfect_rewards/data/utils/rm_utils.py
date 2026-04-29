from abc import abstractmethod

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaForSequenceClassification, AutoModel

from imperfect_rewards.utils.sharedmisc import update_tokenizer, update_model_num_embeddings_and_special_tokens
from imperfect_rewards.utils.strings import DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PADDING_TOKEN


def compute_reward_normalization(rewards: torch.Tensor) -> tuple[float, float]:
    reward_mean = rewards.mean().item()
    reward_var = rewards.var()
    return -reward_mean, torch.rsqrt(reward_var + 1e-8).item()


def normalize_rewards(rewards: torch.Tensor, shift: float, scale: float) -> torch.Tensor:
    return (rewards + shift) * scale


class RewardModelWrapper:

    @abstractmethod
    def prepare(self, **kwargs):
        """
        Setup the reward model and its related resources (e.g. tokenizer) for inference.\
        @param kwargs: Reward model specific arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_batch_rewards(self, tokenized_inputs, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class ArmoRMWrapper(RewardModelWrapper):

    def prepare(self, device=None, cache_dir: str = None, **kwargs):
        self.rm = AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",
                                                                     device_map=device,
                                                                     trust_remote_code=True,
                                                                     torch_dtype=torch.bfloat16,
                                                                     cache_dir=cache_dir)
        self.rm.eval()
        self.rm.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",
                                                       use_fast=True,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        if self.tokenizer.model_max_length > 100000:  # Fixing max length in tokenizer
            self.tokenizer.model_max_length = self.rm.config.max_position_embeddings

    @torch.no_grad()
    def compute_batch_rewards(self, tokenized_inputs, **kwargs):
        output = self.rm(**tokenized_inputs)
        return output.score.float()


class LlamaModelRMWrapper(RewardModelWrapper):

    def prepare(self, rm_str=None, device=None, cache_dir: str = None, **kwargs):
        if rm_str not in ["allenai/Llama-3.1-8B-Instruct-RM-RB2", "LxzGordon/URM-LLaMa-3.1-8B"]:
            self.rm = LlamaForSequenceClassification.from_pretrained(rm_str,
                                                                     num_labels=1,
                                                                     device_map=device,
                                                                     trust_remote_code=True,
                                                                     cache_dir=cache_dir)
        elif "allenai/Llama-3.1-8B-Instruct-RM-RB2" in rm_str:
            self.rm = LlamaForSequenceClassification.from_pretrained(rm_str,
                                                                     num_labels=1,
                                                                     revision="2",
                                                                     device_map=device,
                                                                     trust_remote_code=True,
                                                                     cache_dir=cache_dir)
        elif "LxzGordon/URM-LLaMa-3.1-8B" in rm_str:
            self.rm = LlamaForSequenceClassification.from_pretrained(rm_str,
                                                                     device_map=device,
                                                                     trust_remote_code=True,
                                                                     cache_dir=cache_dir)

        self.rm.eval()
        self.rm.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(rm_str,
                                                       use_fast=True,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        if self.tokenizer.model_max_length > 100000:  # Fixing max length in tokenizer
            self.tokenizer.model_max_length = self.rm.config.max_position_embeddings

    @torch.no_grad()
    def compute_batch_rewards(self, tokenized_inputs, **kwargs):
        output = self.rm(**tokenized_inputs)
        return output.logits.float().view(-1)


class InternLMRMWrapper(RewardModelWrapper):

    def prepare(self, rm_str=None, device=None, cache_dir: str = None, **kwargs):
        self.rm = AutoModel.from_pretrained(
            rm_str,
            device_map=device,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        self.rm.eval()
        self.rm.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(rm_str,
                                                       use_fast=True,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        if self.tokenizer.model_max_length > 100000:  # Fixing max length in tokenizer
            self.tokenizer.model_max_length = self.rm.config.max_position_embeddings

    @torch.no_grad()
    def compute_batch_rewards(self, tokenized_inputs, **kwargs):
        output = self.rm(**tokenized_inputs)
        return output.logits.float().view(-1)


class GeneralRMWrapper(RewardModelWrapper):

    def prepare(self, rm_str=None, device=None, cache_dir: str = None, **kwargs):
        self.rm = AutoModelForSequenceClassification.from_pretrained(rm_str,
                                                                     num_labels=1,
                                                                     device_map=device,
                                                                     trust_remote_code=True,
                                                                     cache_dir=cache_dir)
        self.rm.eval()
        self.rm.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(rm_str,
                                                       use_fast=True,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        if self.tokenizer.model_max_length > 100000:  # Fixing max length in tokenizer
            self.tokenizer.model_max_length = self.rm.config.max_position_embeddings

    @torch.no_grad()
    def compute_batch_rewards(self, tokenized_inputs, **kwargs):
        output = self.rm(**tokenized_inputs)
        return output.logits.float().view(-1)


REWARD_MODEL_WRAPPER_REGISTRY = {
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": ArmoRMWrapper,
    "internlm/internlm2-1_8b-reward": InternLMRMWrapper,
    "internlm/internlm2-7b-reward": InternLMRMWrapper,
    "GeneralRM": GeneralRMWrapper
}

# Used for manually selecting LlamaForSequenceClassification instead of using AutoModelForSequenceClassification since the latter can fail
# when the ground truth reward model is ArmoRM (it seems that upon loading the ArmoRM model it modifies some configuration that creates a bug when loading other Llama-based reward models afterwards)
LLAMA_RMS = [
    "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft",
    "allenai/llama-3-tulu-2-8b-uf-mean-rm",
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "Skywork/Skywork-Reward-V2-Llama-3.2-1B",
    "LxzGordon/URM-LLaMa-3.1-8B",
    "NCSOFT/Llama-3-OffsetBias-RM-8B",
    "allenai/Llama-3.1-8B-Instruct-RM-RB2",
    "allenai/Llama-3.1-Tulu-3-8B-SFT-RM-RB2",
    "HFXM/RAMO-Llama3.1-8B",
    "sfairXC/FsfairX-LLaMA3-RM-v0.1"
]


def get_reward_model_wrapper(reward_model_name: str, logger=None, **kwargs):
    kwargs["rm_str"] = reward_model_name
    is_llama_rm = reward_model_name in LLAMA_RMS

    if reward_model_name in REWARD_MODEL_WRAPPER_REGISTRY:
        rm_wrapper = REWARD_MODEL_WRAPPER_REGISTRY[reward_model_name]()
    elif is_llama_rm:
        rm_wrapper = LlamaModelRMWrapper()
        kwargs["is_llama_rm"] = is_llama_rm
    else:
        if logger is not None:
            logger.info(f"Reward model wrapper '{reward_model_name}' not found in the registry. Defaulting to general config")
        rm_wrapper = REWARD_MODEL_WRAPPER_REGISTRY["GeneralRM"]()

    rm_wrapper.prepare(**kwargs)

    if not rm_wrapper.tokenizer.chat_template:
        if logger is not None:
            logger.warning(f"Reward model {reward_model_name} does not have a chat template. "
                           "Adding a default one, which should only be used for debugging purposes.")
        update_tokenizer(tokenizer=rm_wrapper.tokenizer, num_added_toks=0, pad_token=DEFAULT_PADDING_TOKEN,
                         eos_token=DEFAULT_EOS_TOKEN, logger=logger, user_token=DEFAULT_USER_TOKEN,
                         assistant_token=DEFAULT_ASSISTANT_TOKEN)
        rm_wrapper.rm.config.pad_token_id = rm_wrapper.tokenizer.pad_token_id
        update_model_num_embeddings_and_special_tokens(rm_wrapper.rm, rm_wrapper.tokenizer)
    elif rm_wrapper.rm.config.pad_token_id is None:
        rm_wrapper.rm.config.pad_token_id = rm_wrapper.tokenizer.pad_token_id

    return rm_wrapper
