import argparse
import os
import glob
import torch
import json

from imperfect_rewards.utils.strings import SUFFIX_ALPACAEVAL2

def gather_saved_prompts_or_generations(path):
    if os.path.isfile(path):
        if path.endswith(".pt"):
            return [path]
        else:
            raise ValueError(
                f"Provided file '{path}' does not end with .json."
            )
    else:
        raise ValueError(f"Path '{path}' is not a valid file or directory.")

def format_alpaca_eval(prompts, dataset_name:str, model_name: str, split_name: str):
    formatted_results = []
    for prompt in prompts:
        formatted_entry = {
            "dataset": dataset_name,
            "instruction": prompt,
            "generator": "prompts_UF", 
            "datasplit": split_name
        }
        formatted_results.append(formatted_entry)
    return formatted_results

def main():
    parser = argparse.ArgumentParser(description="Format the generations into Alpaca Eval format")

    parser.add_argument(
        "--path_to_prompts",
        type=str,
        required=True,
        help="Path to prompts",
    )

    args = parser.parse_args()

    path_to_prompts = gather_saved_prompts_or_generations(args.path_to_prompts)

    for prompt_path in path_to_prompts:
        print(f"Processing prompts from {prompt_path}")
        prompts_dict = torch.load(prompt_path)

        model_name = prompts_dict["lm_name"]
        dataset_name = prompts_dict["dataset"]
        prompts = prompts_dict["prompts"]
        split_name = prompts_dict["split"]

        formatted_results = format_alpaca_eval(prompts, dataset_name=dataset_name, model_name=model_name, split_name=split_name)

        json_output_path = prompt_path.replace(".pt", SUFFIX_ALPACAEVAL2)
        print(f"Saving formatted results to {json_output_path}")
        with open(json_output_path, "w") as f:
            json.dump(formatted_results, f, indent=2)

if __name__ == "__main__":
    main()