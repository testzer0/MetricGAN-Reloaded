"""
Adithya Bhaskar, 2022.
All global variables that need to be initialized go here instead of the
config file, as do those that are not user-defined parameters.
"""

import torch

def get_device(log=False):
    """
    Are we on a CPU or GPU?
    """
    if torch.cuda.is_available():
        if log:
            print("Using GPU: {}".format(torch.cuda.get_device_name(0)))
        device = torch.device("cuda")
    else:
        if log:
            print("No GPUs available, using CPU")
        device = torch.device("cpu")
    return device

device = get_device()

if __name__ == '__main__':
    pass