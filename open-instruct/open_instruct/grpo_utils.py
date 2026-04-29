import torch
import torch.distributed as dist

from open_instruct import data_types, model_utils
from open_instruct.utils import (
    INVALID_LOGPROB
)
from open_instruct.grpo_fast import Args

def compute_grpo_loss(
    new_logprobs: torch.Tensor,
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    config: Args,
    tis_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.loss_fn == "dapo":
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - config.clip_lower, 1.0 + config.clip_higher)
    elif config.loss_fn == "cispo":
        # cispo: directly clip ratio, no lower bound.
        # reinforce loss, so multiply by new logprobs
        pg_losses = -advantages * torch.clamp(ratio.detach(), max=1.0 + config.clip_higher) * new_logprobs
        pg_losses2 = pg_losses
    else:
        raise ValueError(f"Invalid loss function: {config.loss_fn}")

    if tis_weights is not None:
        pg_losses = pg_losses * tis_weights
        pg_losses2 = pg_losses2 * tis_weights

    pg_loss_max = torch.max(pg_losses, pg_losses2)

    if ref_logprobs is not None:
        # We want the KL loss to backpropagate through the model.
        # We also clamp the KL loss to avoid numerical instability.
        # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
        ref_logprobs_diff = (new_logprobs - ref_logprobs).clamp(-40.0, 40.0)
        kl_all = model_utils.estimate_kl(ref_logprobs_diff, ratio)
        kl = kl_all[config.kl_estimator]
    else:
        kl = torch.zeros_like(pg_loss_max)

    return pg_losses, pg_losses2, pg_loss_max, kl


def forward_for_logprobs(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    pad_token_id: int,
    temperature: float,
    return_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward pass to compute log probabilities."""
    output = model(input_ids=query_responses, attention_mask=attention_mask, position_ids=position_ids)
    logits = getattr(output, "logits", output)
    logits = logits / temperature
    # The logits at position i predict token i+1, so we align them with labels shifted by 1
    logits = logits[:, :-1]
    labels = query_responses[:, 1:].clone()
    # Replace pad tokens with 0 to avoid index out of bounds errors in gather
    labels[labels == pad_token_id] = 0
    logprob_BT = model_utils.log_softmax_and_gather(logits, labels)

    # For now, entropy is just for monitoring, and we don't pass gradients through it.
    entropy = None
    if return_entropy:
        with torch.no_grad():
            entropy = model_utils.entropy_from_logits(logits)

    return logprob_BT, entropy


def compute_logprobs(
    model: torch.nn.Module,
    data_BT: data_types.CollatedBatchData,
    pad_token_id: int,
    temperature: float,
    use_grad: bool = False,
    batch_size: int | None = None,
) -> list[torch.Tensor]:
    """Compute log probabilities for all samples in batch."""
    logprobs_BT: list[torch.Tensor] = []
    num_samples = len(data_BT.query_responses)

    if batch_size is None:
        batch_size = 1

    context = torch.enable_grad() if use_grad else torch.no_grad()
    with context:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))

            batch_query_responses = torch.cat([data_BT.query_responses[i] for i in batch_indices], dim=0)
            batch_attention_masks = torch.cat([data_BT.attention_masks[i] for i in batch_indices], dim=0)
            batch_position_ids = torch.cat([data_BT.position_ids[i] for i in batch_indices], dim=0)

            batch_logprobs, _ = forward_for_logprobs(
                model,
                batch_query_responses,
                batch_attention_masks,
                batch_position_ids,
                pad_token_id,
                temperature,
                False,
            )

            sample_sizes = [data_BT.query_responses[i].shape[0] for i in batch_indices]
            split_logprobs = torch.split(batch_logprobs, sample_sizes, dim=0)

            for i, logprob_BT in zip(batch_indices, split_logprobs):
                response_mask_BT = data_BT.response_masks[i].to(logprob_BT.device)
                logprob_BT = torch.masked_fill(logprob_BT, ~response_mask_BT[:, 1:].bool(), INVALID_LOGPROB)
                logprobs_BT.append(logprob_BT)

            torch.cuda.empty_cache()

    return logprobs_BT


def calculate_token_counts(
    accumulation_steps: int,
    data_BT: data_types.CollatedBatchData,
    device: torch.device,
    process_group: dist.ProcessGroup | None = None,
) -> dict[int, float]:
    """Compute total token counts per accumulation group, all-reduced across DP ranks.

    Copied from grpo_fast.py to share logic with olmo_core_train_modules.py.
    """
    accumulation_counts: dict[int, float] = {}
    local_counts = [mask[:, 1:].sum().float() for mask in data_BT.response_masks]
    if not local_counts:
        return accumulation_counts

    counts_tensor = torch.stack(local_counts).to(device)
    dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM, group=process_group)

    for i, count in enumerate(counts_tensor):
        group_idx = i // accumulation_steps
        key = int(group_idx * accumulation_steps)
        accumulation_counts[key] = accumulation_counts.get(key, 0.0) + count.item()

    return accumulation_counts