import torch
from model import IeGenerator


def save_model(current_model, path):
    model_args = current_model.args_input_dict
    dict_save = {"model_weights": current_model.state_dict(), "model_args": model_args}
    torch.save(dict_save, path)


def load_model(path):
    dict_load = torch.load(path)
    model_args = dict_load["model_args"]
    loaded_model = IeGenerator(**model_args)
    loaded_model.load_state_dict(dict_load["model_weights"])
    return loaded_model
