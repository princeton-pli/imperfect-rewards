import argparse
import copy
import gc
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.distributed import destroy_process_group
from torch.distributed import is_initialized
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    unset_hf_deepspeed_config,
)

from imperfect_rewards.arguments import CustomRMEvalArguments, CustomTrainingArguments, CustomPolicyGradientTrainingArguments
from imperfect_rewards.arguments.arg_utils import update_dict_config_with_dot_notation_override_args, parse_unknown_argparse_args_into_dict
from imperfect_rewards.recipes.pg import PolicyGradientRecipe
from imperfect_rewards.recipes.rmeval import RMEvalRecipe
from imperfect_rewards.utils import get_logger

logger = get_logger(__name__)


def __set_initial_random_seed(random_seed: int):
    if random_seed > 0:
        seed_offset = torch.distributed.get_rank() if is_initialized() else 0
        random_seed += seed_offset
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def __save_yaml_config(config_dict, output_dir):
    with open(os.path.join(output_dir, "run_config.yaml"), "w") as file:
        yaml.safe_dump(config_dict, file, default_flow_style=False)


def __was_launched_with_accelerate():
    return (
            os.environ.get("ACCELERATE_USE_DISTRIBUTED") == "1"
            or "RANK" in os.environ
            or "WORLD_SIZE" in os.environ
    )


def __is_main_process():
    return (not __was_launched_with_accelerate()) or os.environ.get("RANK", "0") == "0"


def __auto_set_gradient_accumulation_steps_for_accelerate(config: dict):
    if __was_launched_with_accelerate():
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", "1")
        os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(gradient_accumulation_steps)
        logger.info(
            f"Set 'ACCELERATE_GRADIENT_ACCUMULATION_STEPS' environment variable to {gradient_accumulation_steps} for accelerate based on "
            f"'gradient_accumulation_steps' in run config. This overrides any value set in the deepspeed configuration "
            f"so make sure to keep it at 'auto'."
        )


def __create_output_dir_and_save_yaml_config(config_dict: dict, output_dir: str):
    if __is_main_process():  # only create dir and save in the main process
        logger.info("Creating output directory and saving yaml configuration from main process")
        os.makedirs(output_dir, exist_ok=True)
        __save_yaml_config(config_dict, output_dir)


if __name__ == "__main__":
    start_time = datetime.now(timezone.utc)
    try:
        arguments_str = "\n".join([arg for arg in sys.argv[1:]])
        logger.info(f"Arguments received by the script:\n{arguments_str}")

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration")

        args, unknown_args = parser.parse_known_args()
        config_path = args.config
        overrides_dict = parse_unknown_argparse_args_into_dict(unknown_args)
        with open(os.path.abspath(config_path), 'r') as file:
            dict_config = yaml.safe_load(file)

        update_dict_config_with_dot_notation_override_args(dict_config, overrides_dict)

        # Filter out config overrides for disabled features (in preparation to downstream eval)
        filtered_configs = []
        if filtered_configs:
            logger.info(f"Filtered out config for disabled features: {', '.join(filtered_configs)}")

        training_args = CustomTrainingArguments(
            cache_dir=dict_config.get('cache_dir', None),
            seed=dict_config.get('seed', -1),
            do_pg=dict_config.get('do_pg', False),
            do_rm_eval=dict_config.get('do_rm_eval', False),
            exp_dir_prefix=dict_config.get('exp_dir_prefix', "")
        )

        run_id = random.randint(0, 10 ** 6)
        __set_initial_random_seed(training_args.seed)

        if training_args.do_pg:
            training_args.pg_config = CustomPolicyGradientTrainingArguments.from_dict(dict_config["pg_config"],
                                                                                      exp_dir_prefix=training_args.exp_dir_prefix,
                                                                                      seed=training_args.seed,
                                                                                      run_id=run_id)

            __auto_set_gradient_accumulation_steps_for_accelerate(dict_config["pg_config"])
            __create_output_dir_and_save_yaml_config(dict_config, training_args.pg_config.output_dir)

            pg_recipe = PolicyGradientRecipe(training_args)
            pg_recipe.run()
            pg_recipe.trainer.accelerator.wait_for_everyone()
           
            if training_args.pg_config.eval_final_policy_at_end:
                policy_dir = pg_recipe.policy_dir if not pg_recipe.checkpoint_paths else pg_recipe.checkpoint_paths[-1]
                if not pg_recipe.checkpoint_paths:
                    logger.info(f"Temporarily saving policy to {policy_dir} for reloading during evaluation. "
                                f"This is necessary to ensure all distributed wrappers are removed, as otherwise the process hangs"
                                f" when trying to use the policy in RMEvalRecipe.")
                    pg_recipe.trainer.save_model(policy_dir)
                    logger.info(f"Finished temporarily saving policy")

                if is_deepspeed_zero3_enabled():
                    unset_hf_deepspeed_config()

                if pg_recipe.trainer.accelerator.is_main_process:
                    # Deleting previous recipe to free GPU memory and finalize distributed context
                    del pg_recipe
                    torch.cuda.empty_cache()
                    gc.collect()

                    __set_initial_random_seed(training_args.seed)  # Reset random seed for evaluation
                    training_args.rm_eval_config = CustomRMEvalArguments.from_dict(dict_config["rm_eval_config"], seed=training_args.seed,
                                                                                   cache_dir=training_args.cache_dir, run_id=run_id)
                    training_args.rm_eval_config.only_lm_eval = True
                    training_args.rm_eval_config.output_dir = training_args.pg_config.output_dir
                    training_args.rm_eval_config.output_dir_base = training_args.pg_config.output_dir_base
                    training_args.rm_eval_config.language_model_path = policy_dir
                    training_args.rm_eval_config.proxy_reward_models = [training_args.pg_config.reward_model_path]
                    training_args.rm_eval_config.dataset_path = training_args.pg_config.dataset_path
                    training_args.rm_eval_config.delete_language_model_checkpoint_after_eval = training_args.pg_config.delete_language_model_checkpoint_after_eval
                    training_args.rm_eval_config.is_lora_lm = training_args.pg_config.use_lora
                    if training_args.pg_config.use_lora:
                        training_args.rm_eval_config.lora_base_language_model_path = training_args.pg_config.language_model_path

                    rm_eval_recipe = RMEvalRecipe(training_args)
                    rm_eval_recipe.run()

        if training_args.do_rm_eval:
            training_args.rm_eval_config = CustomRMEvalArguments.from_dict(dict_config["rm_eval_config"],
                                                                           exp_dir_prefix=training_args.exp_dir_prefix,
                                                                           seed=training_args.seed,
                                                                           cache_dir=training_args.cache_dir,
                                                                           run_id=run_id)

            __auto_set_gradient_accumulation_steps_for_accelerate(dict_config["rm_eval_config"])
            __create_output_dir_and_save_yaml_config(dict_config, training_args.rm_eval_config.output_dir)

            rm_eval_recipe = RMEvalRecipe(training_args)
            rm_eval_recipe.run()

    except Exception:
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        logger.info(f"Finished running main script. Time took: {end_time - start_time}")
        if torch.distributed.is_initialized():
            destroy_process_group()
