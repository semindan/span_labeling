from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

@dataclass
class Dataset(ABC):
    path: str
    key: str = field(default=None)
    name: str = field(default=None)
    description: str = field(default=None)
    instruction: str = field(default=None)
    examples: list[str] = field(default_factory=list)
    data: list[dict] = field(default_factory=list)

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def _preprocess_item(self, entry):
        pass

    def load_json(self):
        with open(self.path) as f:
            data = json.load(f)
        return data
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __iter__(self):
        for entry in self.data:
            yield entry


@dataclass
class SyntheticDataset(Dataset):
    key: str = field(default="synthetic")
    name: str = field(default="Synthetic regex dataset")
    description: str = field(default="Synthetic pattern finding task")
    examples: list[str] = field(default_factory=lambda: [
        """Task: Find all sequences matching 'ruhig wasser' that are not followed by 'groß'
Text: schiff flugzeug licht wasser schwarz laufen ruhig wasser flugzeug licht m\u00f6gen flugzeug licht flugzeug licht laufen geb\u00e4ude aufgeregt schwarz flugzeug licht wasser ruhig wasser wasser"
Output:
    [
        {"text": "ruhig wasser", "start": 44, "end": 56},
        { "text": "ruhig wasser", "label": "", "start": 163, "end": 175}
    ]
"""])


    def load(self):
        self.data = self.load_json()
        return self
    
    def _preprocess_item(self, entry):
        entry["key"] = self.key
        entry["examples"] = self.examples
        return entry
    
    def __iter__(self):
        for entry in self.data:
            yield self._preprocess_item(entry)
    

@dataclass
class ErrorDataset(Dataset):
    key: str = field(default="error")
    name: str = field(default="Error detection dataset")
    description: str = field(default="Identify and classify grammatical errors in text")
    instruction: str = field(default="Identify spans containing errors and classify them as: GRAMMAR, SPELLING, or PUNCTUATION. Extract any words that are grammatically incorrect. Just the individual words, not phrases.")

    def load(self):
        self.data = self.load_json()
        return self
    
    def _preprocess_item(self, entry):
        if not entry.get("instruction") and self.instruction:
            entry["instruction"] = self.instruction
        if not entry.get("examples") and self.examples:
            entry["examples"] = self.examples

        return entry
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for entry in self.data:
            yield self._preprocess_item(entry)


@dataclass
class NerDataset(Dataset):
    key: str = field(default="ner")
    name: str = field(default="Named Entity Recognition dataset")
    description: str = field(default="NER task to identify PERSON, ORG, and LOC entities")
    instruction: str = field(default="Extract PERSON, ORG, and LOC entities")
    



    def load(self):
        self.data = self.load_json()
        return self
    
    def _preprocess_item(self, entry):
        if not entry.get("instruction") and self.instruction:
            entry["instruction"] = self.instruction
        if not entry.get("examples") and self.examples:
            entry["examples"] = self.examples
        
        entry["instruction"] = self.instruction + " Use 0-indexed character positions."
        return entry
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for entry in self.data:
            yield self._preprocess_item(entry)


if __name__ == "__main__":
    path = "data/custom/synthetic_test.json"
    dataset = SyntheticDataset(name="Synthetic", description="Synthetic pattern finding task", path=path)
    # path = "data/custom/error_test.json"
    # dataset = ErrorDataset(name="Errors", description="Error detection task", path=path)
    dataset = dataset.load()
    print(f"Loaded {len(dataset.data)} examples from {dataset.path}")
    for i, ex in enumerate(dataset):
        print(f"Example {i}:")
        print(f"Entry: {ex}")
        # print(f"Text: {ex['text']}")
        # print(f"Instruction: {ex['instruction']}")
        # print(f"Spans: {ex['spans']}")
        if i >= 2:
            break