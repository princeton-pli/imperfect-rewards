from abc import abstractmethod
from typing import List

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def compute_length_normalized_log_probs(log_probs: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    lengths_safe = sequence_lengths.clamp(min=1.0)
    return log_probs / lengths_safe


class LMWrapper:

    @abstractmethod
    def prepare(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_batch_generation(self, inputs):
        raise NotImplementedError


class CausalLMWrapper(LMWrapper):

    def __init__(self, lm_string, device, config, language_model: torch.nn.Module = None, tokenizer=None):
        self.lm_string = lm_string
        self.device = device
        self.config = config
        self.lm = language_model
        self.tokenizer = tokenizer

    def prepare(self):
        if self.lm is None:
            if not self.config.is_lora_lm:
                self.lm = AutoModelForCausalLM.from_pretrained(self.lm_string,
                                                               cache_dir=self.config.cache_dir,
                                                               device_map=self.device,
                                                               trust_remote_code=True)
            else:
                self.lm = AutoModelForCausalLM.from_pretrained(
                    self.config.lora_base_language_model_path,
                    cache_dir=self.config.cache_dir,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                self.lm = PeftModel.from_pretrained(model=self.lm, model_id=self.lm_string)
                self.lm = self.lm.merge_and_unload()

        self.lm.eval()
        self.lm.to(self.device)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.lm_string,
                                                           cache_dir=self.config.cache_dir,
                                                           use_fast=True,
                                                           trust_remote_code=True)
        self.tokenizer.init_kwargs['padding_side'] = "left"
        self.tokenizer.padding_side = "left"
        self.tokenizer.init_kwargs['truncation_side'] = "right"
        self.tokenizer.truncation_side = "right"

        self.generation_config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **self.config.generation_config
        )

        if self.tokenizer.model_max_length > 100000:  # Fixing max length in tokenizer
            self.tokenizer.model_max_length = self.lm.config.max_position_embeddings

    @torch.inference_mode()
    def compute_batch_generation(self, tokenized_inputs: List[str]):
        outputs = self.lm.generate(
            **tokenized_inputs,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            use_model_defaults=False
        )

        gen_ids = outputs.sequences[:, tokenized_inputs["input_ids"].shape[1]:]
        scores = torch.stack(outputs.scores, dim=0)
        logp = F.log_softmax(scores, dim=-1).permute(1, 0, 2)
        per_token_logprobs = torch.gather(logp, dim=-1, index=gen_ids.unsqueeze(-1)).squeeze(-1)

        # zero out logprobs of padding tokens
        mask = gen_ids.ne(self.tokenizer.pad_token_id)
        per_token_logprobs = per_token_logprobs.masked_fill(~mask, 0.0)

        log_probs_sum = per_token_logprobs.sum(dim=1)
        seq_lens = mask.sum(dim=1).long()

        generated_responses = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return generated_responses, log_probs_sum, seq_lens

    @torch.inference_mode()
    def compute_batch_probs(self, tokenized_inputs: List[str], **kwargs) -> torch.Tensor:
        outputs = self.lm(**tokenized_inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs
