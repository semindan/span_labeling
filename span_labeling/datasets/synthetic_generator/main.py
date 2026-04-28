from synthetic_dataset_generator import (
    SyntheticDatasetGenerator,
    GenerationConfig,
    GenerationMode,
)
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for conditional pattern lookup"
    )
    parser.add_argument(
        "--mode",
        choices=["character", "word"],
        default="character",
        help="Generation mode: character or word level",
    )
    parser.add_argument(
        "--alphabet-size",
        type=int,
        default=4,
        help="Size of character alphabet (for character mode)",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=100, help="Length of generated sequences"
    )
    parser.add_argument(
        "--num-examples", type=int, default=10, help="Number of examples to generate"
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "de", "es", "fr"],
        help="Language for word vocabularies (for word mode)",
    )
    parser.add_argument(
        "--min-pattern-length",
        type=int,
        default=2,
        help="Minimum length of generated patterns",
    )
    parser.add_argument(
        "--max-pattern-length",
        type=int,
        default=3,
        help="Maximum length of generated patterns",
    )
    parser.add_argument(
        "--wildcard-probability",
        type=float,
        default=0.3,
        help="Probability of using wildcards in patterns (0.0-1.0)",
    )
    parser.add_argument("--output", default="dataset.json", help="Output filename")
    parser.add_argument(
        "--demo", action="store_true", help="Run demonstration with sample outputs"
    )

    args = parser.parse_args()

    if args.demo:
        # Run the demonstration from synthetic_dataset_generator
        from synthetic_dataset_generator import main as demo_main

        demo_main()
        return

    # Configure generation
    config = GenerationConfig(
        mode=GenerationMode.CHARACTER
        if args.mode == "character"
        else GenerationMode.WORD,
        alphabet_size=args.alphabet_size,
        sequence_length=args.sequence_length,
        num_examples=args.num_examples,
        language=args.language,
        min_pattern_length=args.min_pattern_length,
        max_pattern_length=args.max_pattern_length,
        wildcard_probability=args.wildcard_probability,
    )

    # Generate dataset
    generator = SyntheticDatasetGenerator(config)
    dataset = generator.generate_dataset()

    # Save to file
    generator.save_dataset(dataset, args.output)
    print(f"Generated {len(dataset)} examples and saved to {args.output}")

    # Print first example as preview
    if dataset:
        example = dataset[0]
        print("\nPreview of first example:")
        print(
            f"Input: {example.input_sequence[:100]}{'...' if len(example.input_sequence) > 100 else ''}"
        )
        print(f"Number of queries: {len(example.queries)}")
        for i, query in enumerate(example.queries[:2]):  # Show first 2 queries
            print(f"  Query {i + 1}: {query.natural_language}")


if __name__ == "__main__":
    main()
