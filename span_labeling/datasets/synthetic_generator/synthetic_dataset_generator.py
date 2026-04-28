"""
Synthetic Dataset Generator for conditional pattern lookup

This script generates examples for training models on conditional pattern lookup tasks.
It supports both character-level and word-level sequence generation with regex-like constraints.
"""

import random
import re
import string
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum
import json

random.seed(432)


class GenerationMode(Enum):
    CHARACTER = "character"
    WORD = "word"


@dataclass
class GenerationConfig:
    """Configuration for dataset generation"""

    mode: GenerationMode
    alphabet_size: int = 4
    sequence_length: int = 100
    num_examples: int = 10
    num_queries_per_example: int = 5
    language: str = "en"
    min_pattern_occurrences: int = 2
    max_pattern_occurrences: int = 8
    min_pattern_length: int = 2
    max_pattern_length: int = 3
    wildcard_probability: float = 0.1  # Probability of using wildcard in pattern


@dataclass
class Query:
    """Represents a regex-like query with constraints"""

    natural_language: str
    pattern: str
    constraint_type: (
        str  # "not_preceded_by", "not_followed_by", "preceded_by", "followed_by"
    )
    constraint_pattern: str


@dataclass
class Example:
    """Represents a single training example"""

    input_sequence: str
    queries: List[Query]
    answers: List[List[Tuple[int, int]]]  # positions for each query


class WordVocabularyManager:
    """Manages word vocabularies for different languages"""

    def __init__(self):
        self.vocabularies = {}
        self._initialize_vocabularies()

    def _initialize_vocabularies(self):
        """Initialize basic vocabularies for different languages"""
        # English vocabulary
        self.vocabularies["en"] = [
            "cat",
            "dog",
            "house",
            "tree",
            "book",
            "water",
            "fire",
            "earth",
            "air",
            "light",
            "dark",
            "big",
            "small",
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "grey",
            "run",
            "walk",
            "jump",
            "fly",
            "swim",
            "eat",
            "drink",
            "sleep",
            "wake",
            "think",
            "love",
            "hate",
            "like",
            "want",
            "need",
            "have",
            "get",
            "give",
            "take",
            "make",
            "hello",
            "goodbye",
            "yes",
            "no",
            "maybe",
            "always",
            "never",
            "sometimes",
            "often",
            "rarely",
            "building",
            "street",
            "car",
            "train",
            "plane",
            "ship",
            "mountain",
            "river",
            "ocean",
            "forest",
            "happy",
            "sad",
            "angry",
            "calm",
            "excited",
            "tired",
            "strong",
            "weak",
            "fast",
            "slow",
            "good",
            "bad",
            "new",
            "old",
            "hot",
            "cold",
            "warm",
            "cool",
            "dry",
            "wet",
            "music",
            "dance",
            "sing",
            "play",
            "work",
            "study",
            "learn",
            "teach",
            "write",
            "read",
            "family",
            "friend",
            "person",
            "child",
            "adult",
            "man",
            "woman",
            "boy",
            "girl",
            "baby",
        ]

        # German vocabulary
        self.vocabularies["de"] = [
            "katze",
            "hund",
            "haus",
            "baum",
            "buch",
            "wasser",
            "feuer",
            "erde",
            "luft",
            "licht",
            "dunkel",
            "groß",
            "klein",
            "rot",
            "blau",
            "grün",
            "gelb",
            "schwarz",
            "weiß",
            "grau",
            "laufen",
            "gehen",
            "springen",
            "fliegen",
            "schwimmen",
            "essen",
            "trinken",
            "schlafen",
            "aufwachen",
            "denken",
            "lieben",
            "hassen",
            "mögen",
            "wollen",
            "brauchen",
            "haben",
            "bekommen",
            "geben",
            "nehmen",
            "machen",
            "hallo",
            "auf_wiedersehen",
            "ja",
            "nein",
            "vielleicht",
            "immer",
            "nie",
            "manchmal",
            "oft",
            "selten",
            "gebäude",
            "straße",
            "auto",
            "zug",
            "flugzeug",
            "schiff",
            "berg",
            "fluss",
            "ozean",
            "wald",
            "glücklich",
            "traurig",
            "wütend",
            "ruhig",
            "aufgeregt",
            "müde",
            "stark",
            "schwach",
            "schnell",
            "langsam",
        ]

        # Spanish vocabulary
        self.vocabularies["es"] = [
            "gato",
            "perro",
            "casa",
            "árbol",
            "libro",
            "agua",
            "fuego",
            "tierra",
            "aire",
            "luz",
            "oscuro",
            "grande",
            "pequeño",
            "rojo",
            "azul",
            "verde",
            "amarillo",
            "negro",
            "blanco",
            "gris",
            "correr",
            "caminar",
            "saltar",
            "volar",
            "nadar",
            "comer",
            "beber",
            "dormir",
            "despertar",
            "pensar",
            "amar",
            "odiar",
            "gustar",
            "querer",
            "necesitar",
            "tener",
            "obtener",
            "dar",
            "tomar",
            "hacer",
            "hola",
            "adiós",
            "sí",
            "no",
            "tal_vez",
            "siempre",
            "nunca",
            "a_veces",
            "a_menudo",
            "raramente",
            "edificio",
            "calle",
            "coche",
            "tren",
            "avión",
            "barco",
            "montaña",
            "río",
            "océano",
            "bosque",
        ]

        # French vocabulary
        self.vocabularies["fr"] = [
            "chat",
            "chien",
            "maison",
            "arbre",
            "livre",
            "eau",
            "feu",
            "terre",
            "air",
            "lumière",
            "sombre",
            "grand",
            "petit",
            "rouge",
            "bleu",
            "vert",
            "jaune",
            "noir",
            "blanc",
            "gris",
            "courir",
            "marcher",
            "sauter",
            "voler",
            "nager",
            "manger",
            "boire",
            "dormir",
            "réveiller",
            "penser",
            "aimer",
            "détester",
            "aimer",
            "vouloir",
            "avoir_besoin",
            "avoir",
            "obtenir",
            "donner",
            "prendre",
            "faire",
            "bonjour",
            "au_revoir",
            "oui",
            "non",
            "peut_être",
            "toujours",
            "jamais",
            "parfois",
            "souvent",
            "rarement",
        ]

    def get_vocabulary(self, language: str) -> List[str]:
        """Get vocabulary for a specific language"""
        return self.vocabularies.get(language, self.vocabularies["en"])

    def get_random_words(self, language: str, count: int) -> List[str]:
        """Get random words from a language vocabulary"""
        vocab = self.get_vocabulary(language)
        return random.choices(vocab, k=count)


class CharacterSequenceGenerator:
    """Generates character-level sequences with constraints"""

    def __init__(self, alphabet_size: int = 4, config: GenerationConfig = None):
        self.alphabet = string.ascii_uppercase[:alphabet_size]
        self.config = config or GenerationConfig(mode=GenerationMode.CHARACTER)

    def generate_sequence(self, length: int) -> str:
        """Generate a random character sequence"""
        return "".join(random.choices(self.alphabet, k=length))

    def _generate_random_pattern(self, min_length: int, max_length: int) -> str:
        """Generate a random pattern with specified length range"""
        length = random.randint(min_length, max_length)
        pattern = ""

        for _ in range(length):
            if random.random() < self.config.wildcard_probability:
                pattern += "."
            else:
                pattern += random.choice(self.alphabet)

        return pattern

    def generate_queries(self) -> List[Query]:
        """Generate character-level queries with constraints"""
        queries = []

        # Constraint types to use
        constraint_types = [
            "not_preceded_by",
            "not_followed_by",
            "preceded_by",
            "followed_by",
        ]

        # Generate random patterns and queries
        for i in range(self.config.num_queries_per_example):
            # Generate random pattern
            pattern = self._generate_random_pattern(
                self.config.min_pattern_length, self.config.max_pattern_length
            )

            # Choose random constraint type
            constraint_type = random.choice(constraint_types)

            # Choose random constraint character
            constraint_char = random.choice(self.alphabet)

            # Generate natural language description
            if constraint_type == "not_preceded_by":
                nl = f"Find all sequences matching '{pattern}' that are not preceded by '{constraint_char}'"
            elif constraint_type == "not_followed_by":
                nl = f"Find all sequences matching '{pattern}' that are not followed by '{constraint_char}'"
            elif constraint_type == "preceded_by":
                nl = f"Find all sequences matching '{pattern}' that are preceded by '{constraint_char}'"
            elif constraint_type == "followed_by":
                nl = f"Find all sequences matching '{pattern}' that are followed by '{constraint_char}'"

            queries.append(
                Query(
                    natural_language=nl,
                    pattern=pattern,
                    constraint_type=constraint_type,
                    constraint_pattern=constraint_char,
                )
            )

        return queries

    def inject_patterns_batch(
        self,
        sequence: str,
        queries: List[Query],
        min_occurrences: int,
        max_occurrences: int,
    ) -> str:
        """Inject patterns for all queries at once to avoid conflicts"""
        sequence = list(sequence)
        sequence_length = len(sequence)

        # Pick a random number of patterns for each query in the range [min, max]
        patterns_per_query = [
            random.randint(min_occurrences, max_occurrences) for _ in queries
        ]
        total_patterns = sum(patterns_per_query)

        max_pattern_length = max(len(q.pattern) for q in queries)
        min_spacing_per_pattern = max_pattern_length + 2  # +2 for context

        required_length = total_patterns * min_spacing_per_pattern

        # Extend sequence if needed
        if required_length > sequence_length:
            additional_length = required_length - sequence_length
            sequence.extend(random.choices(self.alphabet, k=additional_length))
            sequence_length = len(sequence)

        spacing = max(min_spacing_per_pattern, sequence_length // (total_patterns + 1))

        # Track used positions to prevent overlaps
        used_positions = set()

        # Inject patterns for each query
        position_offset = 0
        for query_idx, query in enumerate(queries):
            pattern_length = len(query.pattern)
            num_patterns = patterns_per_query[query_idx]

            # Generate concrete pattern
            concrete_pattern = ""
            for char in query.pattern:
                if char == ".":
                    concrete_pattern += random.choice(self.alphabet)
                else:
                    concrete_pattern += char

            # Inject the exact number of patterns for this query
            for pattern_num in range(num_patterns):
                # Calculate position - distribute patterns across the sequence
                base_pos = position_offset * spacing + 1

                # Ensure we don't go out of bounds
                if base_pos + pattern_length + 1 >= sequence_length:
                    base_pos = sequence_length - pattern_length - 2
                    if base_pos < 1:
                        base_pos = 1

                # Inject the pattern
                for j, char in enumerate(concrete_pattern):
                    if base_pos + j < sequence_length:
                        sequence[base_pos + j] = char

                # Mark positions as used
                for j in range(base_pos - 1, base_pos + pattern_length + 1):
                    used_positions.add(j)

                # Ensure constraint is satisfied
                if query.constraint_type == "not_preceded_by":
                    if base_pos > 0:
                        sequence[base_pos - 1] = random.choice(
                            [c for c in self.alphabet if c != query.constraint_pattern]
                        )
                elif query.constraint_type == "not_followed_by":
                    if base_pos + pattern_length < sequence_length:
                        sequence[base_pos + pattern_length] = random.choice(
                            [c for c in self.alphabet if c != query.constraint_pattern]
                        )
                elif query.constraint_type == "preceded_by":
                    if base_pos > 0:
                        sequence[base_pos - 1] = query.constraint_pattern
                    else:
                        base_pos = 1
                        for j, char in enumerate(concrete_pattern):
                            if base_pos + j < sequence_length:
                                sequence[base_pos + j] = char
                        sequence[base_pos - 1] = query.constraint_pattern
                elif query.constraint_type == "followed_by":
                    if base_pos + pattern_length < sequence_length:
                        sequence[base_pos + pattern_length] = query.constraint_pattern
                    else:
                        base_pos = max(0, sequence_length - pattern_length - 2)
                        for j, char in enumerate(concrete_pattern):
                            if base_pos + j < sequence_length:
                                sequence[base_pos + j] = char
                        if base_pos + pattern_length < sequence_length:
                            sequence[base_pos + pattern_length] = (
                                query.constraint_pattern
                            )

                position_offset += 1

        # Fill unused positions with characters that don't create additional matches
        for i, char in enumerate(sequence):
            if i not in used_positions:
                # Use a character that's less likely to create unintended matches
                sequence[i] = random.choice(self.alphabet)

        return "".join(sequence)


class WordSequenceGenerator:
    """Generates word-level sequences with constraints"""

    def __init__(self, language: str = "en", config: GenerationConfig = None):
        self.language = language
        self.vocab_manager = WordVocabularyManager()
        self.config = config or GenerationConfig(mode=GenerationMode.WORD)

    def generate_sequence(self, length: int) -> str:
        """Generate a random word sequence"""
        words = self.vocab_manager.get_random_words(self.language, length)
        return " ".join(words)

    def _generate_random_word_pattern(self, min_length: int, max_length: int) -> str:
        """Generate a random word pattern with specified length range"""
        length = random.randint(min_length, max_length)
        vocab = self.vocab_manager.get_vocabulary(self.language)
        sample_words = random.sample(vocab, min(10, len(vocab)))

        pattern_parts = []
        for i in range(length):
            if random.random() < self.config.wildcard_probability:
                pattern_parts.append("\\w+")
            else:
                pattern_parts.append(random.choice(sample_words))

        return " ".join(pattern_parts)

    def generate_queries(self) -> List[Query]:
        """Generate word-level queries with constraints"""
        queries = []
        vocab = self.vocab_manager.get_vocabulary(self.language)

        # Sample words for constraints
        sample_words = random.sample(vocab, min(20, len(vocab)))

        # Constraint types to use
        constraint_types = [
            "not_preceded_by",
            "not_followed_by",
            "preceded_by",
            "followed_by",
        ]

        # Generate random patterns and queries
        for i in range(self.config.num_queries_per_example):
            # Generate random pattern
            pattern = self._generate_random_word_pattern(
                self.config.min_pattern_length, self.config.max_pattern_length
            )

            # Choose random constraint type
            constraint_type = random.choice(constraint_types)

            # Choose random constraint word
            constraint_word = random.choice(sample_words)

            # Generate natural language description
            if constraint_type == "not_preceded_by":
                nl = f"Find all sequences matching '{pattern}' that are not preceded by '{constraint_word}'"
            elif constraint_type == "not_followed_by":
                nl = f"Find all sequences matching '{pattern}' that are not followed by '{constraint_word}'"
            elif constraint_type == "preceded_by":
                nl = f"Find all sequences matching '{pattern}' that are preceded by '{constraint_word}'"
            elif constraint_type == "followed_by":
                nl = f"Find all sequences matching '{pattern}' that are followed by '{constraint_word}'"

            queries.append(
                Query(
                    natural_language=nl,
                    pattern=pattern,
                    constraint_type=constraint_type,
                    constraint_pattern=constraint_word,
                )
            )

        return queries

    def inject_patterns_batch(
        self,
        sequence: str,
        queries: List[Query],
        min_occurrences: int,
        max_occurrences: int,
    ) -> str:
        """Inject word patterns for all queries at once to avoid conflicts"""
        words = sequence.split()
        sequence_length = len(words)

        vocab = self.vocab_manager.get_vocabulary(self.language)

        # Pick a random number of patterns for each query in the range [min, max]
        patterns_per_query = [
            random.randint(min_occurrences, max_occurrences) for _ in queries
        ]
        total_patterns = sum(patterns_per_query)

        max_pattern_length = max(len(q.pattern.split()) for q in queries)
        min_spacing_per_pattern = max_pattern_length + 2  # +2 for context

        required_length = total_patterns * min_spacing_per_pattern

        # Extend sequence if needed
        if required_length > sequence_length:
            additional_words = required_length - sequence_length
            words.extend(
                self.vocab_manager.get_random_words(self.language, additional_words)
            )
            sequence_length = len(words)

        spacing = max(min_spacing_per_pattern, sequence_length // (total_patterns + 1))

        # Track used positions to prevent overlaps
        used_positions = set()

        # Generate concrete pattern function
        def generate_concrete_pattern(pattern_str):
            concrete_pattern = []
            for part in pattern_str.split():
                if part == "\\w+":
                    concrete_pattern.append(random.choice(vocab))
                else:
                    concrete_pattern.append(part)
            return concrete_pattern

        # Inject patterns for each query
        position_offset = 0
        for query_idx, query in enumerate(queries):
            pattern_parts = query.pattern.split()
            pattern_length = len(pattern_parts)
            num_patterns = patterns_per_query[query_idx]

            # Inject the exact number of patterns for this query
            for pattern_num in range(num_patterns):
                # Calculate position - distribute patterns across the sequence
                base_pos = position_offset * spacing + 1

                # Ensure we don't go out of bounds
                if base_pos + pattern_length + 1 >= sequence_length:
                    base_pos = sequence_length - pattern_length - 2
                    if base_pos < 1:
                        base_pos = 1

                # Generate and inject concrete pattern
                concrete_pattern = generate_concrete_pattern(query.pattern)
                for j, word in enumerate(concrete_pattern):
                    if base_pos + j < sequence_length:
                        words[base_pos + j] = word

                # Mark positions as used
                for j in range(base_pos - 1, base_pos + pattern_length + 1):
                    if 0 <= j < sequence_length:
                        used_positions.add(j)

                # Ensure constraint is satisfied
                if query.constraint_type == "not_preceded_by":
                    if base_pos > 0:
                        # Choose a word that's NOT the constraint pattern
                        words[base_pos - 1] = random.choice(
                            [w for w in vocab if w != query.constraint_pattern]
                        )
                elif query.constraint_type == "not_followed_by":
                    if base_pos + pattern_length < sequence_length:
                        words[base_pos + pattern_length] = random.choice(
                            [w for w in vocab if w != query.constraint_pattern]
                        )
                elif query.constraint_type == "preceded_by":
                    if base_pos > 0:
                        words[base_pos - 1] = query.constraint_pattern
                    else:
                        base_pos = 1
                        for j, word in enumerate(concrete_pattern):
                            if base_pos + j < sequence_length:
                                words[base_pos + j] = word
                        words[base_pos - 1] = query.constraint_pattern
                elif query.constraint_type == "followed_by":
                    if base_pos + pattern_length < sequence_length:
                        words[base_pos + pattern_length] = query.constraint_pattern
                    else:
                        base_pos = max(0, sequence_length - pattern_length - 2)
                        for j, word in enumerate(concrete_pattern):
                            if base_pos + j < sequence_length:
                                words[base_pos + j] = word
                        if base_pos + pattern_length < sequence_length:
                            words[base_pos + pattern_length] = query.constraint_pattern

                position_offset += 1

        # Fill unused positions with random words to prevent accidental matches
        for i in range(sequence_length):
            if i not in used_positions:
                words[i] = random.choice(vocab)

        return " ".join(words)


class SyntheticDatasetGenerator:
    """Main class for generating synthetic datasets"""

    def __init__(self, config: GenerationConfig):
        self.config = config

        if config.mode == GenerationMode.CHARACTER:
            self.sequence_generator = CharacterSequenceGenerator(
                config.alphabet_size, config
            )
        else:
            self.sequence_generator = WordSequenceGenerator(config.language, config)

    def find_matches(self, sequence: str, query: Query) -> List[Tuple[int, int]]:
        """Find all matches for a query in the sequence"""
        matches = []

        if self.config.mode == GenerationMode.CHARACTER:
            pattern_regex = query.pattern.replace(
                ".", f"[{self.sequence_generator.alphabet}]"
            )

            for match in re.finditer(pattern_regex, sequence):
                start, end = match.span()

                # Check constraint
                valid = True
                if query.constraint_type == "not_preceded_by":
                    if start > 0 and sequence[start - 1] == query.constraint_pattern:
                        valid = False
                elif query.constraint_type == "not_followed_by":
                    if (
                        end < len(sequence)
                        and sequence[end] == query.constraint_pattern
                    ):
                        valid = False
                elif query.constraint_type == "preceded_by":
                    if start == 0 or sequence[start - 1] != query.constraint_pattern:
                        valid = False
                elif query.constraint_type == "followed_by":
                    if (
                        end >= len(sequence)
                        or sequence[end] != query.constraint_pattern
                    ):
                        valid = False

                if valid:
                    matches.append((start, end))

        else:  # Word mode
            words = sequence.split()
            pattern_parts = query.pattern.split()

            for i in range(len(words) - len(pattern_parts) + 1):
                match = True
                for j, part in enumerate(pattern_parts):
                    if part == "\\w+":
                        continue  # Wildcard matches any word
                    elif words[i + j] != part:
                        match = False
                        break

                if match:
                    # Check constraint
                    valid = True
                    if query.constraint_type == "not_preceded_by":
                        if i > 0 and words[i - 1] == query.constraint_pattern:
                            valid = False
                    elif query.constraint_type == "not_followed_by":
                        if (
                            i + len(pattern_parts) < len(words)
                            and words[i + len(pattern_parts)]
                            == query.constraint_pattern
                        ):
                            valid = False
                    elif query.constraint_type == "preceded_by":
                        if i == 0 or words[i - 1] != query.constraint_pattern:
                            valid = False
                    elif query.constraint_type == "followed_by":
                        if (
                            i + len(pattern_parts) >= len(words)
                            or words[i + len(pattern_parts)] != query.constraint_pattern
                        ):
                            valid = False

                    if valid:
                        # Calculate character positions
                        start_char = sum(len(w) + 1 for w in words[:i]) - (
                            1 if i > 0 else 0
                        )
                        end_char = (
                            sum(len(w) + 1 for w in words[: i + len(pattern_parts)]) - 1
                        )
                        matches.append((start_char, end_char))

        return matches

    def generate_example(self) -> Example:
        """Generate a single example with queries and answers"""
        # Generate initial sequence
        sequence = self.sequence_generator.generate_sequence(
            self.config.sequence_length
        )

        # Generate queries
        queries = self.sequence_generator.generate_queries()

        # Inject patterns for ALL queries at once to avoid conflicts
        sequence = self.sequence_generator.inject_patterns_batch(
            sequence,
            queries,
            self.config.min_pattern_occurrences,
            self.config.max_pattern_occurrences,
        )

        # Find answers for each query
        answers = []
        for query in queries:
            matches = self.find_matches(sequence, query)
            answers.append(matches)

        return Example(input_sequence=sequence, queries=queries, answers=answers)

    def generate_dataset(self) -> List[Example]:
        """Generate a complete dataset"""
        dataset = []
        for _ in range(self.config.num_examples):
            example = self.generate_example()
            dataset.append(example)
        return dataset

    def save_dataset(self, dataset: List[Example], filename: str):
        """Save dataset to JSON file"""
        data = []
        for example in dataset:
            example_data = {
                "input_sequence": example.input_sequence,
                "queries": [
                    {
                        "natural_language": q.natural_language,
                        "pattern": q.pattern,
                        "constraint_type": q.constraint_type,
                        "constraint_pattern": q.constraint_pattern,
                    }
                    for q in example.queries
                ],
                "answers": example.answers,
            }
            data.append(example_data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Main function demonstrating the dataset generator"""

    # Character-level example
    print("Generating character-level dataset...")
    char_config = GenerationConfig(
        mode=GenerationMode.CHARACTER,
        alphabet_size=4,
        sequence_length=100,
        num_examples=100,
        num_queries_per_example=5,
        min_pattern_length=2,
        max_pattern_length=3,
        wildcard_probability=0.4,
        language="en",
    )

    char_generator = SyntheticDatasetGenerator(char_config)
    char_dataset = char_generator.generate_dataset()
    char_generator.save_dataset(char_dataset, "character_dataset.json")

    # Print example
    example = char_dataset[0]
    print("\nCharacter Example:")
    print(f"Input: {example.input_sequence}")
    for i, (query, answer) in enumerate(zip(example.queries, example.answers)):
        print(f"Query {i + 1}: {query.natural_language}")
        print(f"Matches ({len(answer)}): {answer}")

    # Word-level example
    print("\n" + "=" * 50)
    print("Generating word-level dataset...")
    word_config = GenerationConfig(
        mode=GenerationMode.WORD,
        sequence_length=60,
        num_examples=100,
        language="en",
        min_pattern_length=1,
        max_pattern_length=2,
        wildcard_probability=0.3,
    )

    word_generator = SyntheticDatasetGenerator(word_config)
    word_dataset = word_generator.generate_dataset()
    print(len(word_dataset))
    word_generator.save_dataset(word_dataset, "english_word_dataset.json")

    # Print example
    example = word_dataset[0]
    print("\nWord Example:")
    print(f"Input: {example.input_sequence}")
    for i, (query, answer) in enumerate(zip(example.queries, example.answers)):
        print(f"Query {i + 1}: {query.natural_language}")
        print(f"Matches ({len(answer)}): {answer}")

    # Multi-language example
    print("\n" + "=" * 50)
    print("Generating German word-level dataset...")
    german_config = GenerationConfig(
        mode=GenerationMode.WORD,
        sequence_length=60,
        num_examples=100,
        language="de",
        min_pattern_length=1,
        max_pattern_length=2,
        wildcard_probability=0.3,
    )

    german_generator = SyntheticDatasetGenerator(german_config)
    german_dataset = german_generator.generate_dataset()
    german_generator.save_dataset(german_dataset, "german_word_dataset.json")

    example = german_dataset[0]
    print("\nGerman Word Example:")
    print(f"Input: {example.input_sequence}")
    for i, (query, answer) in enumerate(zip(example.queries, example.answers)):
        print(f"Query {i + 1}: {query.natural_language}")
        print(f"Matches ({len(answer)}): {answer}")

    print("\n" + "=" * 50)
    print("Generating examples")

    example_config = GenerationConfig(
        mode=GenerationMode.WORD,
        sequence_length=20,
        num_examples=5,
        language="en",
        min_pattern_length=1,
        max_pattern_length=2,
        min_pattern_occurrences=1,
        max_pattern_occurrences=3,
        wildcard_probability=0.3,
    )

    example_generator = SyntheticDatasetGenerator(example_config)
    example_dataset = example_generator.generate_dataset()
    example_generator.save_dataset(example_dataset, "example_word_dataset.json")


if __name__ == "__main__":
    main()
