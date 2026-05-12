import torch


@torch.inference_mode()
def output_shape(module: torch.nn.Module, dummy: torch.Tensor):
    output = module(dummy)
    return output.shape
