import json
from pathlib import Path
from typing import Iterable, List, Dict, Any


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries.
      Each line in the file should be a JSON object, e.g.:
        {"input": "question text", "output": "answer text"}
    """

    # Make sure we have a Path object so we can call .open() on it
    path = Path(path)

    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            obj = json.loads(line)  # convert text line -> Python dict
            items.append(obj)

    return items


def iter_texts(samples: Iterable[Dict[str, Any]], field: str = "input") -> Iterable[str]:
    """Yield one text string from each sample.

    "field" says which key to read, e.g. "input" or "output".
    """

    for sample in samples:
        value = sample.get(field)
        if isinstance(value, str):
            yield value

