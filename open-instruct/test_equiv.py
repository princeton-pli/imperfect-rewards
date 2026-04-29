from open_instruct.math_utils import (
    last_boxed_only_string, remove_boxed, is_equiv, hendrycks_is_equiv, strip_string
)

ground_truth = "pi\\10"
model_response = "\\boxed{\\pi\10}"

