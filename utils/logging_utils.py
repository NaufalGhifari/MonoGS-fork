import rich
from rich.console import Console

# Create a global Console instance
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

    # console.print automatically pushes logs to a new line above active tqdm bars
    console.print(message)