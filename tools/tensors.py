import torch
from torch import Tensor

# =====================================================
# =*= TENSOR-RELATED TOOLS USED ACCROSS THE PROJECT =*=
# ======================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

def add_into_tensor(tensor_list: Tensor | None, tensor_val: Tensor):
    if tensor_list is None:
        tensor_list = tensor_val
    else:
        tensor_list = torch.cat((tensor_list, tensor_val), dim=0)
    return tensor_list

def to_binary(booleanVal: bool):
    return 1 if booleanVal else 0

def features2tensor(features: list, device: str):
    for f in features:
        if isinstance(f, bool):
            f = to_binary(f)
    return torch.tensor([[f for f in features]], dtype=torch.float, device=device)

def id2tensor(id1: int, id2: int, device: str):
    return torch.tensor([[id1], [id2]], dtype=torch.long, device=device)