import argparse
import subprocess
import os
import glob
import sys
import tempfile
import yaml
from pathlib import Path

from imperfect_rewards.utils.strings import SUFFIX_ALPACAEVAL2

_SRC_DIR = Path(__file__).resolve().parents[3]
os.environ['OPENAI_CLIENT_CONFIG_PATH'] = str(_SRC_DIR / "configs/alpaca_eval/openai_config/config.yaml")

def gather_alpacaeval2_files(path):
    if os.path.isfile(path):
        if path.endswith(".json"):
            return [path]
        else:
            raise ValueError(
                f"Provided file '{path}' does not end with .json."
            )
    elif os.path.isdir(path):
        return glob.glob(os.path.join(path, "**", f"*{SUFFIX_ALPACAEVAL2}"), recursive=True)
    else:
        raise ValueError(f"Path '{path}' is not a valid file or directory.")

def build_annotators_config() -> str:
    """Load the annotators config YAML, patch the prompt_template to an absolute path,
    and write it to a temporary file whose path is returned."""
    configs_dir = _SRC_DIR / "configs/alpaca_eval/weighted_alpaca_eval_gpt41"
    clf_path = configs_dir / "alpaca_eval_clf.txt"
    annotators_config_path = configs_dir / "configs.yaml"

    with open(annotators_config_path) as f:
        config_data = yaml.safe_load(f)

    for key in config_data:
        config_data[key]["prompt_template"] = str(clf_path)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config_data, tmp)
    tmp.close()
    return tmp.name

def main():
    parser = argparse.ArgumentParser(description="Run alpaca_eval using pre-computed outputs")

    parser.add_argument(
        "--model_outputs",
        type=str,
        required=True,
        help="Path to model outputs to use for Alpaca Eval (single file or directory).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=False,
        help="Use only outputs from these steps",
    )
    parser.add_argument(
        "--reference_outputs",
        type=str,
        required=False,
        help="Path to reference outputs to use for Alpaca Eval (single file or directory).",
    )  
    args = parser.parse_args()

    try:
        model_output_files = gather_alpacaeval2_files(args.model_outputs)
        if args.steps is not None:
            model_output_files = [f for f in model_output_files if f"checkpoint-{args.steps}_eval" in f]
    except ValueError as e:
        print(e)
        sys.exit(1)

    reference_output_files = []
    if args.reference_outputs:
        try:
            reference_output_files = gather_alpacaeval2_files(args.reference_outputs)
            assert len(reference_output_files) == 1, "One reference file allowed for now"
        except ValueError as e:
            print(e)
            sys.exit(1)

    annotators_config = build_annotators_config()

    print()
    print("Model output files to process:")
    print(model_output_files)
    print()
    for mo_file in model_output_files:
        mo_folder = Path(mo_file).parent
        mo_output_folder = mo_folder / "alpaca_eval_results"
        mo_output_folder.mkdir(parents=True, exist_ok=True)

        command = ["alpaca_eval", "--model_outputs", mo_file]

        command += ["--reference_outputs", reference_output_files[0]]

        command += ["--output_path", mo_output_folder.as_posix()]
        caching_file = mo_output_folder / "annotations_cache.json"
        command += ["--caching_path", caching_file.as_posix()]

        # Remove instruction_difficulty and regularization per https://github.com/tatsu-lab/alpaca_eval/issues/346
        command += ["--metric_kwargs", "{'glm_name':'length_controlled_minimal'}"]
        command += ["--annotators_config", annotators_config]

        print(f"Running: {' '.join(command)}")
        with open(mo_output_folder / "command.txt", "w") as f:
            f.write(' '.join(command))

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")

if __name__ == "__main__":
    main()
