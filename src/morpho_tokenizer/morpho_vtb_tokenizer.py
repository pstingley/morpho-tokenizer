
from importlib import resources
from pathlib import Path

def _default_lexicon_paths():
    data_dir = resources.files("morpho_tokenizer") / "data"
    root = data_dir / "morph_lexicon_by_root.json.gz"
    proto = data_dir / "morph_lexicon_by_protoroot.json.gz"
    return str(root), str(proto)

# MorphoTokenizer implementation goes here

class MorphoTokenizer:
    def __init__(self):
        pass
