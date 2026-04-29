DEFAULT_USER_TOKEN = "<|user|>"
DEFAULT_ASSISTANT_TOKEN = "<|assistant|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_PADDING_TOKEN = "<|padding|>"

DATASET_KEY_NAME = "dataset"
SPLIT_KEY_NAME = "split"
GOLD_RM_KEY_NAME = "gold_rm_name"
LM_KEY_NAME = "lm_name"
RM_KEY_NAME = "rm_name"
GENERATIONS_TYPE_KEY_NAME = "type_of_generations"
LM_NAME_FOR_PROBS_KEY_NAME = "lm_name_for_probs"
PROMPT_INDICES_KEY_NAME = "prompt_indices"
IS_ONE_VS_MANY_KEY_NAME = "is_one_vs_many_dataset"

OFFLINE_RESPONSES_NAME = "offline"
ONLINE_RESPONSES_NAME = "online"
PROMPTS = "prompts"
PER_PROMPT_RESPONSES = "per_prompt_responses"
PER_PROMPT_REWARDS = "per_prompt_rewards"
PER_PROMPT_LOGPROBS = "per_prompt_log_probs"
PER_PROMPT_SEQUENCE_LENGTHS = "per_prompt_sequence_lengths"
PER_PROMPT_LENGTH_NORMALIZED_LOGPROBS = "per_prompt_length_normalized_log_probs"
PER_PROMPT_RANKING_ACCURACY = "per_prompt_ranking_acc"
PER_PROMPT_WEIGHTED_RANKING_ACCURACY = "per_prompt_weighted_ranking_acc"
NORMALIZED_PER_PROMPT_WEIGHTED_RANKING_ACCURACY = "normalized_per_prompt_weighted_ranking_acc"  # normalized the agreement weights
PER_PROMPT_LENGTH_NORMALIZED_WEIGHTED_RANKING_ACCURACY = "per_prompt_length_normalized_weighted_ranking_acc"
NORMALIZED_PER_PROMPT_LENGTH_NORMALIZED_WEIGHTED_RANKING_ACCURACY = "normalized_per_prompt_length_normalized_weighted_ranking_acc"

DEFAULT_TRAIN_RM_SPLIT_NAME = "train_rm_prefs"
DEFAULT_TRAIN_RLHF_SPLIT_NAME = "train_rlhf_prefs"
DEFAULT_TEST_SPLIT_NAME = "test_prefs"

SUFFIX_ALPACAEVAL2 = "_AlpacaEval2.json"


def get_chat_template():
    CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '{DEFAULT_USER_TOKEN}' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '{DEFAULT_ASSISTANT_TOKEN}' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '{DEFAULT_ASSISTANT_TOKEN}'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '{DEFAULT_ASSISTANT_TOKEN}' }}{% endif %}{% endfor %}"
    # CHAT_TEMPLATE = """{%- for message in messages %}{%- if message['role'] == 'user' %}{{ '{DEFAULT_USER_TOKEN}' }}{%- elif message['role'] == 'system' or message['role'] == 'assistant' %}{{ '{DEFAULT_ASSISTANT_TOKEN}' }}{%- endif %}{{ message['content'] + eos_token }}{%- if not loop.last %}\n{%- endif %}{%- endfor %}{%- if add_generation_prompt %}\n{{ '{DEFAULT_ASSISTANT_TOKEN}' }}{%- endif %}"""

    CHAT_TEMPLATE = CHAT_TEMPLATE.replace(
        '{DEFAULT_USER_TOKEN}', DEFAULT_USER_TOKEN
    ).replace(
        '{DEFAULT_ASSISTANT_TOKEN}', DEFAULT_ASSISTANT_TOKEN
    )

    return CHAT_TEMPLATE
