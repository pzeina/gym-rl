import torch

# ruff: noqa: T201
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")  # type: ignore # noqa: PGH003
    print(f"PyTorch is using GPU: {torch.cuda.current_device()}")
else:
    print("CUDA is NOT available.")
