import warnings
from argparse import Namespace
from typing import Dict, Union


def merge_params(
    base_params: Union[Namespace, Dict],
    override_params: Dict,
) -> Union[Namespace, Dict]:
    """
    Update the parameters in ``base_params`` with ``override_params``.
    Can be useful to override parsed command line arguments.'

    Args:
        base_params: Namespace or dictionary base parameters.
        override_params: Dictionary of parameters to override. Usually the
            parameters got from ``get_next_parameters()``. When it is none,
            nothing will happen.

    Returns:
        The updated ``base_params``. Note that ``base_params`` will
        be updated inplace. The return value is only for convenience.

    """
    # if not override_params:
    #     return base_params
    if isinstance(base_params, dict):
        _is_dict = True
        base_params_ = base_params
    else:
        _is_dict = False
        base_params_ = vars(base_params)
    for __k, __v in override_params.items():
        __t = type(base_params_[__k])
        if not isinstance(__v, __t) and base_params_[__k] is not None:
            warnings.warn(
                f"Expected {__k} in override parameters to have type {__t}, "
                f"but found type {type(__v)} of value {__v} instead. "
                "Overriding anyway ..."
            )
        base_params_[__k] = __v
    return base_params_ if _is_dict else Namespace(**base_params_)
