from torch import nn


def calc_norm_dict(model, ord=1, errors="ignore", no_biases=True):
    """Calculate the norm of order `ord` for weights and gradients in
    each layer of model, returned as a dictionary.
    """
    _norm_dict = {}
    _mod_name = model.__class__.__name__
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if no_biases and "bias" in name:
            continue
        try:
            _norm_dict[f"{_mod_name}.{name}.weight_norm"] = parameter.data.norm(ord).item()
            _norm_dict[f"{_mod_name}.{name}.grad_norm"] = parameter.grad.norm(ord).item()
        except Exception as e:
            if errors != "ignore":
                raise e

    return _norm_dict


def weights_init(m):
    # TODO docstring
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def gather_object_params(obj, prefix="", clip=249):
    # TODO docstring
    return {prefix + k: str(v)[:clip] if len(str(v)) > clip else v for k, v in obj.__dict__.items()}
