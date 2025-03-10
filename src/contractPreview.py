from rich.console import Console
from rich.text import Text
from rich.panel import Panel

# Load the contract text
contract_file = "examples\contract_gpt.txt"
with open(contract_file, "r", encoding="utf-8") as f:
    contract_text = f.read()

# Create a console for rich output
console = Console()

# Define color themes
header_style = "bold cyan"
section_style = "bold yellow"
highlight_style = "bold magenta"
normal_style = "white"

# Process the contract text
formatted_lines = []
for line in contract_text.split("\n"):
    line = line.strip()
    if line.startswith("**") and line.endswith("**"):  # Bold Headers
        formatted_lines.append(Text(line.strip("**"), style=header_style))
    elif line.startswith("- "):  # List Items
        formatted_lines.append(Text(line, style=highlight_style))
    elif ":" in line and not line.startswith("-"):  # Key-Value Pairs
        key, value = line.split(":", 1)
        formatted_lines.append(Text(f"{key.strip()}:", style=section_style) + Text(value.strip(), style=normal_style))
    else:  # Regular Text
        formatted_lines.append(Text(line, style=normal_style))

# Display the formatted contract
console.print(Panel.fit("\n".join(str(line) for line in formatted_lines), title="📜 Contract Preview", border_style="green"))
