import rich
from rich.console import Console
from tqdm import tqdm

# Create an isolated console that captures output rather than printing it
console = Console(capture=True)

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

    # use rich to render the text with colors
    with console.capture() as capture:
        console.print(message)

    # print w tqdm
    tqdm.write(capture.get().strip())