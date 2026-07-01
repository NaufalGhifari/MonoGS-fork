import rich
from rich.console import Console
from tqdm import tqdm

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