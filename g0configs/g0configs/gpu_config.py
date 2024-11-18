import torch


def get_device(use_gpu = False):

    # CUDA for PyTorch
    if use_gpu:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    else:
        device = torch.device("cpu")

    print("using device:", device)
    return device