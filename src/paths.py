from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# '*' means make parts into a tuple of strings
# useful helper
def here(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)