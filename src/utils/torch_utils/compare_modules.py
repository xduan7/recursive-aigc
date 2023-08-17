import torch
from torch import nn


def _is_eq_modules(
    source_module: nn.Module,
    target_module: nn.Module,
    weight_only: bool,
) -> bool:
    """Compare two modules and return whether they are the same."""
    if weight_only:
        return torch.equal(source_module.weight, target_module.weight) and (
            source_module.bias is None
            or torch.equal(source_module.bias, target_module.bias)
        )
    else:
        return bool(torch.equal(source_module, target_module))


def compare_modules(
    source_module: nn.Module,
    target_module: nn.Module,
    weight_only: bool,
    verbose: bool = False,
) -> bool:
    """Compare two modules and return whether they are the same.

    Args:
        source_module: The source PyTorch module.
        target_module: The target PyTorch module.
        weight_only: Whether to compare only the weights (and bias) of the
            modules.
        verbose: Boolean indicating whether to print the details of the
            comparison (e.g. the names of the modules that are different).

    Returns:
        A boolean indicating whether the modules are the same.

    """
    source_dict = source_module.state_dict()
    target_dict = target_module.state_dict()
    if source_dict.keys() != target_dict.keys():
        raise ValueError(
            "Parameter length mismatch. "
            "Ensure the modules share the same architecture."
        )
    differences = []
    for param_name, source_param in source_dict.items():
        target_param = target_dict.get(param_name)
        # If comparing only weights and biases,
        # continue if current param is not weight or bias
        if not _is_eq_modules(
            source_module=source_param,
            target_module=target_param,
            weight_only=weight_only,
        ):
            differences.append(param_name)

    if differences and verbose:
        print(
            f"Found the following {len(differences)} out of "
            f"{len(source_dict)} differences: {differences}"
        )
    return len(differences) == 0
