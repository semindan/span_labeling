from dataclasses import dataclass, field
from abc import abstractmethod
import json
from span_labeling.base import DatasetBase


@dataclass
class Dataset(DatasetBase):
    path: str
    key: str = ""
    name: str = ""
    description: str = ""
    instruction: str = ""
    data: list[dict] = field(default_factory=list)

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def _preprocess_item(self, entry):
        pass

    def load_json(self):
        with open(self.path) as f:
            return json.load(f)

    def __getitem__(self, idx):
        return self._preprocess_item(self.data[idx])

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]


@dataclass
class SyntheticDataset(Dataset):
    key: str = "synthetic"
    name: str = "Synthetic regex dataset"
    description: str = "Synthetic pattern finding task"
    instruction: str = "Find patterns matching the queries"

    def load(self):
        self.data = self.load_json()
        return self

    def _preprocess_item(self, entry):
        entry["key"] = self.key
        # entry["instruction"] = f"Task: {entry.get('instruction', self.instruction)}"
        entry["model_input"] = f'Task: {entry["instruction"]}\nText: "{entry["text"]}"'
        return entry


@dataclass
class ErrorDataset(Dataset):
    key: str = "error"
    name: str = "Error detection dataset"
    description: str = "Identify and classify grammatical errors in text"
    instruction: str = "Identify spans containing errors and classify them as: GRAMMAR, SPELLING, or PUNCTUATION. Extract any words that are grammatically incorrect. Just the individual words, not phrases."

    def load(self):
        self.data = self.load_json()
        return self

    def _preprocess_item(self, entry):
        entry["key"] = self.key
        # entry["instruction"] = f"Task: {entry.get('instruction', self.instruction)}"
        entry["model_input"] = f'Text: "{entry["text"]}"'
        return entry


@dataclass
class MultigecDataset(Dataset):
    key: str = "multigec"
    name: str = "MultiGEC dataset"
    description: str = "Grammatical error correction task with multiple annotations"
    instruction: str = "Identify spans containing errors and correct them. Use the following labels: R - replace, U - unnecessary, M - missing. Correct just the individual words, not phrases."

    def load(self):
        self.data = self.load_json()
        return self

    def _preprocess_item(self, entry):
        entry["key"] = self.key
        # entry["instruction"] = f"Task: {entry.get('instruction', self.instruction)}"
        entry["model_input"] = f'Text: "{entry["text"]}"'
        return entry


@dataclass
class NerDataset(Dataset):
    key: str = "ner"
    name: str = "Named Entity Recognition dataset"
    description: str = "NER task to identify PERSON, ORG, and LOC entities"
    instruction: str = "Extract PERSON, ORG, and LOC entities"

    def load(self):
        self.data = self.load_json()
        return self

    def _preprocess_item(self, entry):
        entry["key"] = self.key
        # entry["instruction"] = f"Task: {entry.get('instruction', self.instruction)}"
        entry["model_input"] = f'Text: "{entry["text"]}"'
        return entry


@dataclass
class WMTDataset(Dataset):
    key: str = "wmt"
    name: str = "WMT dataset"
    description: str = "WMT translation quality estimation"
    instruction: str = "Identify translation errors by comparing the translation to the source text. Error severity: 0=minor, 1=major."

    def load(self):
        self.data = self.load_json()
        return self

    def _preprocess_item(self, entry):
        entry["key"] = self.key
        # entry["instruction"] = f"Task: {entry.get('instruction', self.instruction)}"
        entry["model_input"] = (
            f'Source: "{entry["source"]}"\nTranslation: "{entry["text"]}"'
        )

        spans = []
        for span in entry["spans"]:
            spans.append(
                {
                    "text": span["text"],
                    "start": span["start"],
                    "end": span["end"],
                    "label": span.get("label", ""),
                }
            )
        entry["spans"] = spans
        return entry


if __name__ == "__main__":
    # Example usage
    dataset = NerDataset(path="data/custom/ner_en_test.json").load()
    for item in dataset:
        print(item)
