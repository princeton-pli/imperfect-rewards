import argparse
import os
import glob
import torch
import json

from imperfect_rewards.utils.strings import SUFFIX_ALPACAEVAL2

def gather_saved_prompts_or_generations(path, use_train_pg: bool =True):
    str_train_pg = "train" if use_train_pg else "test"
    if os.path.isfile(path):
        if path.endswith(".pt"):
            return [path]
        else:
            raise ValueError(
                f"Provided file '{path}' does not end with .json."
            )
    elif os.path.isdir(path):
        return glob.glob(os.path.join(path, "**", f"per_prompt_responses_*_{str_train_pg}.pt"), recursive=True)
    else:
        raise ValueError(f"Path '{path}' is not a valid file or directory.")

def format_alpaca_eval(prompts, generations, dataset_name:str, model_name: str, split_name: str):
    formatted_results = []
    for prompt, generation in zip(prompts, generations):
        formatted_entry = {
            "dataset": dataset_name,
            "instruction": prompt,
            "output": generation,
            "generator": model_name, 
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
    parser.add_argument(
        "--path_to_generations",
        type=str,
        required=False,
        help="Path to dir usually containing the PG results (or initial RM Eval)",
    )
    parser.add_argument(
        "--use_train_pg",
        action="store_true",
        help="Whether to use the train or test generations",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for selecting the generations",
    )

    args = parser.parse_args()

    path_to_generations = gather_saved_prompts_or_generations(args.path_to_generations, use_train_pg=True)
    path_to_prompts = gather_saved_prompts_or_generations(args.path_to_prompts)
    path_to_prompts = path_to_prompts * len(path_to_generations)

    for prompt_path, generations_path in zip(path_to_prompts, path_to_generations):
        print(f"Processing prompts from {prompt_path} and generations from {generations_path}")
        prompts_dict = torch.load(prompt_path)
        generations_dict = torch.load(generations_path)

        model_name = generations_dict["lm_name"]
        dataset_name = generations_dict["dataset"]
        generations = generations_dict["per_prompt_responses"]
        prompts = prompts_dict["prompts"]
        split_name = generations_dict["split"]

        if args.random_seed is None:
            torch.manual_seed(args.random_seed)
            sampled_generations = []
            for per_prompt_gens in generations:
                if not per_prompt_gens:
                    raise ValueError("Found prompt with zero generations.")
                idx = torch.randint(low=0, high=len(per_prompt_gens), size=(1,)).item()
                sampled_generations.append(per_prompt_gens[idx])
        else:
            sampled_generations = [per_prompt_gens[0] for per_prompt_gens in generations]
        
        formatted_results = format_alpaca_eval(prompts, sampled_generations, dataset_name=dataset_name, model_name=model_name, split_name=split_name)

        json_output_path = generations_path.replace(".pt", SUFFIX_ALPACAEVAL2)
        print(f"Saving formatted results to {json_output_path}")
        with open(json_output_path, "w") as f:
            json.dump(formatted_results, f, indent=2)

if __name__ == "__main__":
    main()