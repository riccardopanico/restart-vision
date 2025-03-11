import torch
print("Torch path:", torch.__file__)
print("Cuda available?", torch.cuda.is_available())
print("Devices:", torch.cuda.device_count())
