import jax
import torch as tc


def check_gpu():
    """Returns if cuda is available to pytorch"""
    cuda_available = tc.cuda.is_available()
    print("Torch GPU available?", cuda_available)
    print(f"Torch v{tc.__version__}, torch cuda toolkit v{tc.version.cuda}")
    n_device = tc.cuda.device_count()
    print("Avaialable devices", n_device)
    for i in range(n_device):
        print(f"Device {i}: {tc.cuda.get_device_name(i)}")
    if cuda_available:
        curr_dev = tc.cuda.current_device()
    else:
        curr_dev = "cpu"
    print("Torch current device", curr_dev)
    print("Jax available devices:", jax.devices())
    return cuda_available


if __name__ == "__main__":
    check_gpu()