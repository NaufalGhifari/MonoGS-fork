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
    Computes the L2 norm of the gradients for each parameter group,
    prints it out, and logs it cleanly to a CSV file.
    """
    # Create file and write header if it doesn't exist
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("frame_idx,iteration,loss,param_name,grad_norm_l2\n")
            
    # Calculate gradient norms
    log_lines = []
    print_strings = []
    
    for name, param in named_parameters:
        if param.grad is not None:
            # L2 Norm: square root of sum of squared gradients
            grad_norm = torch.norm(param.grad, p=2).item()
            log_lines.append(f"{frame_idx},{itr},{loss_val:.6f},{name},{grad_norm:.6f}\n")
            print_strings.append(f"      ↳ {name:15s} | Grad Norm: {grad_norm:.6f}")
        else:
            log_lines.append(f"{frame_idx},{itr},{loss_val:.6f},{name},None\n")

    # Append to file in one atomic operation
    with open(filepath, "a") as f:
        f.writelines(log_lines)
        
    # Print out summary to terminal
    if print_strings:
        print(f"📊 [Frame {frame_idx} | Itr {itr:02d}] Loss: {loss_val:.4f}")
        for s in print_strings:
            print(s)