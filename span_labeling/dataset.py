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

    def load(self):
        self.data = self.load_json()
        return self
    
    def _preprocess_item(self, entry):
        entry["key"] = self.key
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
        entry["key"] = self.key
        entry["instruction"] = entry.get("instruction", self.instruction)
        return entry
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for entry in self.data:
            yield self._preprocess_item(entry)


@dataclass
class MultigecDataset(Dataset):
    key: str = field(default="multigec")
    name: str = field(default="MultiGEC dataset")
    description: str = field(default="Grammatical error correction task with multiple annotations")
    instruction: str = field(default="Identify spans containing errors and correct them. Use the following labels: R - replace, U - unnecessary, M - missing. Correct just the individual words, not phrases.")

    def load(self):
        self.data = self.load_json()
        return self
    
    def _preprocess_item(self, entry):
        entry["key"] = self.key
        entry["instruction"] = entry.get("instruction", self.instruction)
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
        entry["key"] = self.key
        entry["instruction"] = entry.get("instruction", self.instruction)
        return entry
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for entry in self.data:
            yield self._preprocess_item(entry)

@dataclass
class WMTDataset(Dataset):
    key: str = field(default="wmt")
    name: str = field(default="WMT dataset")
    description: str = field(default="WMT")
    instruction: str = field(default="Identify translation errors by comparing the translation to the source text. The error is usually a word or a short phrase that does not reflect the .")

    def load(self):
        self.data = self.load_json()
        return self
    
    def _preprocess_item(self, entry):
        entry["key"] = self.key
        entry["instruction"] = entry.get("instruction", self.instruction)
        entry["model_input"] = f'Source: "{entry["source"]}"\nTranslation: "{entry["text"]}"'
        spans = []
        for span in entry["spans"]:
            spans.append(
                {
                    "text": span["text"],
                    "start": span["start"],
                    "label" : "",
                    "end": span["end"]
                }
            )
        entry["spans"] = spans
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