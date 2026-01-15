import torch

def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
