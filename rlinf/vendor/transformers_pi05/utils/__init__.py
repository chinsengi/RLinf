import transformers.utils as _utils

from . import generic as _generic

globals().update(vars(_utils))

for _name in (
    "LossKwargs",
    "ModelOutput",
    "auto_docstring",
    "can_return_tuple",
    "filter_out_non_signature_kwargs",
    "torch_int",
):
    if hasattr(_generic, _name):
        globals()[_name] = getattr(_generic, _name)
