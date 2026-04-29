import argparse
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Union

import datasets
from datasets import Dataset, DatasetDict

import imperfect_rewards.utils.single_process_logging as logging_utils


def create_conversation(prompt: str, response: str) -> List[Dict[str, str]]:
    return [
        {"content": prompt, "role": "user"},
        {"content": response, "role": "assistant"}
    ]


def create_one_vs_many_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a single example with one chosen conversation and a list of rejected conversations.
    """
    prompt = example['prompt']
    chosen_list = example['chosen'] if isinstance(example['chosen'], list) else [example['chosen']]
    rejected_list = example['rejected'] if isinstance(example['rejected'], list) else [example['rejected']]

    chosen_response = chosen_list[0]
    chosen_formatted = create_conversation(prompt, chosen_response)
    rejected_formatted = [create_conversation(prompt, rejected_response) for rejected_response in rejected_list]

    new_example = {
        'prompt': prompt,
        'chosen': chosen_formatted,
        'rejected': rejected_formatted,
        'orig_id': example['id']
    }

    for key in example.keys():
        if key not in ['id', 'prompt', 'chosen', 'rejected']:
            new_example[key] = example[key]

    return new_example


def process_single_split(logger, dataset: Dataset, split_name: str) -> Dataset:
    """Process a single split of the dataset."""
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Processing split: {split_name}")
    logger.info(f"{'=' * 50}")
    logger.info(f"Original dataset size: {len(dataset)}")

    logger.info("Filtering out rows where subset == 'Ties'...")
    dataset = dataset.filter(lambda x: x['subset'] != 'Ties')
    logger.info(f"Dataset size after filtering: {len(dataset)}")

    new_examples = []

    for idx, example in enumerate(dataset):
        if idx % 100 == 0:
            logger.info(f"Processing example {idx}/{len(dataset)}...")

        new_example = create_one_vs_many_example(example)
        new_examples.append(new_example)

    logger.info(f"Created {len(new_examples)} examples for {split_name}")
    for i, example in enumerate(new_examples):
        example['id'] = i

    return Dataset.from_list(new_examples)


def reformat_rewardbench(logger, dataset_name: str = "allenai/reward-bench-2", split: str = "test") -> \
        Union[Dataset, DatasetDict]:
    logger.info(f"Loading dataset {dataset_name}...")
    dataset = datasets.load_dataset(dataset_name, split=split)
    reformatted_dataset = process_single_split(logger, dataset, split)
    return DatasetDict({"test": reformatted_dataset})


def main(output_path: str = "rewardbench_2_one_vs_many_reformatted", dataset_name: str = "allenai/reward-bench-2", split: str = "test"):
    os.makedirs(output_path, exist_ok=True)
    logger = logging_utils.create_logger(console_logging=True, file_logging=True, log_dir=output_path,
                                         log_file_name_prefix="rewardbench_reformatting")
    start_time = datetime.now(timezone.utc)
    reformatted_dataset = reformat_rewardbench(logger, dataset_name=dataset_name, split=split)

    logger.info("\n" + "=" * 50)
    logger.info("Reformatting complete!")
    logger.info("=" * 50)

    logger.info(f"Splits: {list(reformatted_dataset.keys())}")
    for split_name, split_data in reformatted_dataset.items():
        logger.info(f"  {split_name}: {len(split_data)} examples")

    first_split = list(reformatted_dataset.keys())[0]
    logger.info(f"\nColumns: {reformatted_dataset[first_split].column_names}")

    logger.info("\n" + "=" * 50)
    logger.info(f"Sample example from '{first_split}' split:")
    logger.info("=" * 50)
    if len(reformatted_dataset[first_split]) > 0:
        sample = reformatted_dataset[first_split][0]
        logger.info(f"{json.dumps(sample, indent=2)}")

    logger.info(f"\nSaving dataset to {output_path}...")
    reformatted_dataset.save_to_disk(output_path)
    logger.info("Done!")

    end_time = datetime.now(timezone.utc)
    logger.info(f"Finished running script. Time took: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat RewardBench2 dataset from one chosen vs many rejected to conversational format.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data_files/rewardbench_2_one_vs_many_reformatted",
        help="Path where the reformatted dataset will be saved"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/reward-bench-2",
        help="Name of the dataset to load"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Name of the dataset split to process"
    )

    args = parser.parse_args()
    main(output_path=args.output_path, dataset_name=args.dataset_name, split=args.split)
