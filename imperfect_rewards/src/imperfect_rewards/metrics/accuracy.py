import torch


def acc(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    Computes ranking accuracy (Acc) based on rewards of chosen and rejected responses.
    Will compare each reward in reward_chosen to the reward in the
    corresponding location of reward_rejected.
    Args:
        reward_chosen: Tensor of shape (num_prompts, num_responses) containing the rewards for the chosen responses
        reward_rejected: Tensor of shape (num_prompts, num_responses) containing the rewards for the rejected responses
    """
    indicators = (reward_chosen > reward_rejected).float()
    return indicators.mean()


def one_vs_many_acc(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    Computes one-vs-many ranking accuracy (Acc). For each prompt, checks if the reward of the chosen response
    is greater than all rewards of the rejected responses.
    Args:
        reward_chosen: Tensor of shape (num_prompts, ) containing the rewards for the chosen responses
        reward_rejected: Tensor of shape (num_prompts, num_rejected_responses) containing the rewards for the rejected responses
    """
    indicators = (reward_chosen > reward_rejected.max(dim=1).values).float()
    return indicators.mean()


def acc_w(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor,
          log_prob_chosen: torch.Tensor, log_prob_rejected: torch.Tensor) -> torch.Tensor:
    """
    Computes the Weighted Accuracy (Acc-W) metric, which weights accuracy by the probability of the chosen and rejected responses.
    Will compare each reward in reward_chosen to the corresponding reward in reward_rejected.

    Args:
        reward_chosen: Tensor of shape (num_prompts, num_responses) containing the rewards for the chosen responses
        reward_rejected: Tensor of shape (num_prompts, num_responses) containing the rewards for the rejected responses
        log_prob_chosen: Tensor of shape (num_prompts, num_responses) containing the probabilities of the chosen responses
        log_prob_rejected: Tensor of shape (num_prompts, num_responses) containing the probabilities of the rejected responses
    """
    log_probs_sum = log_prob_chosen + log_prob_rejected
    weights = log_probs_sum - torch.max(log_probs_sum)
    weights = torch.exp(weights - torch.logsumexp(weights.flatten(), dim=0))
    indicators = (reward_chosen > reward_rejected).float()
    weighted_acc = (weights * indicators).sum()
    return weighted_acc


def one_vs_many_acc_w(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor,
                      log_prob_chosen: torch.Tensor, log_prob_rejected: torch.Tensor) -> torch.Tensor:
    """
    Computes the one-vs-many Weighted Accuracy (Acc-W) metric, which weights accuracy by the probability of the chosen and rejected responses.
    For each prompt, checks if the reward of the chosen response is greater than all rewards of the rejected responses.

    Args:
        reward_chosen: Tensor of shape (num_prompts, ) containing the rewards for the chosen responses
        reward_rejected: Tensor of shape (num_prompts, num_rejected_responses) containing the rewards for the rejected responses
        log_prob_chosen: Tensor of shape (num_prompts, ) containing the probabilities of the chosen responses
        log_prob_rejected: Tensor of shape (num_prompts, num_rejected_responses) containing the probabilities of the rejected responses
    """
    log_probs_sum = log_prob_chosen + log_prob_rejected.sum(dim=1)
    weights = log_probs_sum - torch.max(log_probs_sum)
    weights = torch.exp(weights - torch.logsumexp(weights.flatten(), dim=0))
    indicators = (reward_chosen > reward_rejected.max(dim=1).values).float()
    weighted_acc = (weights * indicators).sum()
    return weighted_acc


def hacc_w(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor,
           log_prob_chosen: torch.Tensor, log_prob_rejected: torch.Tensor, per_prompt_mean_reward: torch.Tensor,
           return_indicator_means: bool = False) -> torch.Tensor:
    """
    Computes the Harm-Aware Weighted Accuracy (HAcc-W) metric.
    Will compare each reward in reward_chosen to the corresponding reward in reward_rejected.

    Args:
        reward_chosen: Tensor of shape (num_prompts, num_responses) containing the rewards for the chosen responses
        reward_rejected: Tensor of shape (num_prompts, num_responses) containing the rewards for the rejected responses
        log_prob_chosen: Tensor of shape (num_prompts, num_responses) containing the probabilities of the chosen responses
        log_prob_rejected: Tensor of shape (num_prompts, num_responses) containing the probabilities of the rejected responses
        per_prompt_mean_reward: Tensor of shape (num_prompts,) containing for each prompt an estimate of the expected reward obtained by the policy
    """
    log_probs_sum = log_prob_chosen + log_prob_rejected
    weights = log_probs_sum - torch.max(log_probs_sum)
    weights = torch.exp(weights - torch.logsumexp(weights.flatten(), dim=0))
    correct_ranking_indicator = reward_chosen > reward_rejected

    per_prompt_mean_reward = per_prompt_mean_reward.unsqueeze(dim=-1) if len(per_prompt_mean_reward.shape) == 1 and len(reward_rejected.shape) > 1 \
        else per_prompt_mean_reward
    reward_rejected_less_than_mean = reward_rejected < per_prompt_mean_reward
    indicators = (correct_ranking_indicator | reward_rejected_less_than_mean).float()

    weighted_acc = (weights * indicators).sum()

    if return_indicator_means:
        return weighted_acc, correct_ranking_indicator.float().mean(), reward_rejected_less_than_mean.float().mean()

    return weighted_acc


def hacc(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor, per_prompt_mean_reward: torch.Tensor,
         return_indicator_means: bool = False) -> torch.Tensor:
    """
    Computes the Harm-Aware Accuracy (HAcc) metric with uniform weighting over pairs.

    Args:
        reward_chosen: Tensor of shape (num_prompts, num_responses) containing the rewards for the chosen responses
        reward_rejected: Tensor of shape (num_prompts, num_responses) containing the rewards for the rejected responses
        per_prompt_mean_reward: Tensor of shape (num_prompts,) containing for each prompt an estimate of the expected reward obtained by the policy
    """
    correct_ranking_indicator = reward_chosen > reward_rejected

    per_prompt_mean_reward = per_prompt_mean_reward.unsqueeze(dim=-1) if len(per_prompt_mean_reward.shape) == 1 and len(reward_rejected.shape) > 1 \
        else per_prompt_mean_reward
    reward_rejected_less_than_mean = reward_rejected < per_prompt_mean_reward
    indicators = (correct_ranking_indicator | reward_rejected_less_than_mean).float()

    acc = indicators.mean()

    if return_indicator_means:
        return acc, correct_ranking_indicator.float().mean(), reward_rejected_less_than_mean.float().mean()

    return acc


def one_vs_many_hacc_w(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor,
                       log_prob_chosen: torch.Tensor, log_prob_rejected: torch.Tensor, per_prompt_mean_reward: torch.Tensor) -> torch.Tensor:
    """
    Computes the one-vs-many Harm-Aware Weighted Accuracy (HAcc-W) metric.
    For each prompt, checks if the reward of the chosen response is greater than all rewards of the rejected responses,
    or if all rejected rewards are less than the mean reward.
    Args:
        reward_chosen: Tensor of shape (num_prompts, ) containing the rewards for the chosen responses
        reward_rejected: Tensor of shape (num_prompts, num_rejected_responses) containing the rewards for the rejected responses
        log_prob_chosen: Tensor of shape (num_prompts, ) containing the probabilities of the chosen responses
        log_prob_rejected: Tensor of shape (num_prompts, num_rejected_responses) containing the probabilities of the rejected responses
        per_prompt_mean_reward: Tensor of shape (num_prompts,) containing for each prompt an estimate of the expected reward obtained by the policy
    """
    log_probs_sum = log_prob_chosen + log_prob_rejected.sum(dim=1)
    weights = log_probs_sum - torch.max(log_probs_sum)
    weights = torch.exp(weights - torch.logsumexp(weights.flatten(), dim=0))
    correct_ranking_indicator = reward_chosen > reward_rejected.max(dim=1).values

    if len(per_prompt_mean_reward.shape) == 1 and len(reward_rejected.shape) > 1:
        per_prompt_mean_reward = per_prompt_mean_reward.unsqueeze(dim=-1)

    reward_rejected_less_than_mean = (reward_rejected < per_prompt_mean_reward).all(dim=1)
    indicators = (correct_ranking_indicator | reward_rejected_less_than_mean).float()

    weighted_acc = (weights * indicators).sum()
    return weighted_acc


def one_vs_many_hacc(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor,
                     per_prompt_mean_reward: torch.Tensor) -> torch.Tensor:
    """
    Computes the one-vs-many Harm-Aware Accuracy (HAcc) metric with uniform weighting over prompts.

    Args:
        reward_chosen: Tensor of shape (num_prompts, ) containing the rewards for the chosen responses
        reward_rejected: Tensor of shape (num_prompts, num_rejected_responses) containing the rewards for the rejected responses
        per_prompt_mean_reward: Tensor of shape (num_prompts,) containing for each prompt an estimate of the expected reward obtained by the policy
    """
    correct_ranking_indicator = reward_chosen > reward_rejected.max(dim=1).values

    if len(per_prompt_mean_reward.shape) == 1 and len(reward_rejected.shape) > 1:
        per_prompt_mean_reward = per_prompt_mean_reward.unsqueeze(dim=-1)

    reward_rejected_less_than_mean = (reward_rejected < per_prompt_mean_reward).all(dim=1)
    indicators = (correct_ranking_indicator | reward_rejected_less_than_mean).float()

    return indicators.mean()
