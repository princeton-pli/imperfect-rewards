# fixed from trl/trainer/utils.py

from typing import Tuple

import torch
from transformers import GenerationConfig
from trl.trainer.utils import pad


def generate_no_model_defaults(
        lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        use_model_defaults=False,  # Don't overwrite with model defaults
        return_dict_in_generate=True,
        output_scores=True
    )
    logits = torch.stack(output.scores, 1)

    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


@torch.no_grad()
def batch_generation_no_model_defaults(
        model: torch.nn.Module,
        queries: torch.Tensor,
        local_rollout_forward_batch_size: int,
        pad_token_id: int,
        generation_config: GenerationConfig
):
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i: i + local_rollout_forward_batch_size]
        query_response, logits = generate_no_model_defaults(
            model,
            query,
            pad_token_id,
            generation_config
        )
        query_responses.append(query_response)
        logitss.append(logits)

    # padding tensors
    padded_query_responses = pad(query_responses, padding_value=pad_token_id, padding_side="right")
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # reshaping
    padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    return padded_query_responses, padded_logitss
