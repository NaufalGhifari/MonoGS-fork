import rich
from rich.console import Console
from tqdm import tqdm
import os
import torch

# Fix: Initialize without the unexpected 'capture' argument
console = Console()

_log_styles = {
    "MonoGS": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
}

def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="MonoGS"):
    style = get_style(tag)
    message = f"[{style}]{tag}:[/{style}] " + " ".join(map(str, args))

    # This context manager captures the styled string cleanly
    with console.capture() as capture:
        console.print(message)

    # Hand it over to tqdm to print above the progress bar
    tqdm.write(capture.get().strip())

def log_gradients_to_csv(filepath, frame_idx, itr, loss_val, named_parameters):
    """
    Writes detailed gradient data to a background CSV file silently.
    """
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("frame_idx,iteration,loss,param_name,grad_norm_l2\n")
            
    log_lines = []
    for name, param in named_parameters:
        if param.grad is not None:
            grad_norm = torch.norm(param.grad, p=2).item()
            log_lines.append(f"{frame_idx},{itr},{loss_val:.6f},{name},{grad_norm:.6f}\n")
        else:
            log_lines.append(f"{frame_idx},{itr},{loss_val:.6f},{name},None\n")

    with open(filepath, "a") as f:
        f.writelines(log_lines)