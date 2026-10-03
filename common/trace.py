"""Print parameter counts and per-layer tensor shapes."""

import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trace_shapes(model, x):
    """Forward once and print each leaf module: input shape -> output shape."""
    hooks = []

    def make_hook(name):
        def hook(_module, inp, out):
            if not inp or not torch.is_tensor(inp[0]) or not torch.is_tensor(out):
                return
            print(f"{name:28s} {tuple(inp[0].shape)} -> {tuple(out.shape)}")

        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name or "output")))

    model.eval()
    with torch.no_grad():
        model(x)
    for hook in hooks:
        hook.remove()


def run_shapes(model, x):
    print(f"params={count_params(model):,}")
    trace_shapes(model, x)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"output {tuple(y.shape)}")
    return y
