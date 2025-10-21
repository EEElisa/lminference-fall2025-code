#!/usr/bin/env python3

import sys
import traceback

def debug_dataset():
    try:
        from datasets import load_dataset
        print("Loading dataset...")
        raw_dataset = load_dataset("vashistht/11763_datasets", "graph_dev")
        print("Dataset loaded successfully")

        print(f"Dataset keys: {raw_dataset.keys()}")
        dev_test = raw_dataset["dev_test"]
        print(f"dev_test type: {type(dev_test)}")

        # Try to get first example safely
        examples = []
        for i, example in enumerate(dev_test):
            examples.append(example)
            if i >= 2:  # Get first 3 examples
                break

        print(f"Number of examples collected: {len(examples)}")

        if examples:
            example = examples[0]
            print(f"Example keys: {list(example.keys())}")

            if 'edges' in example:
                edges = example['edges']
                print(f"edges type: {type(edges)}")
                print(f"edges: {edges}")

                if isinstance(edges, list):
                    print(f"edges is a list with {len(edges)} items")
                    if len(edges) > 0:
                        print(f"first edge: {edges[0]}")
                        print(f"first edge type: {type(edges[0])}")
                elif isinstance(edges, dict):
                    print(f"edges is a dict with keys: {list(edges.keys())}")
                    for key in ['src', 'dst', 'weight']:
                        if key in edges:
                            print(f"edges['{key}']: {edges[key]} (type: {type(edges[key])})")

            # Print full example structure for first example
            print(f"Full example: {example}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset()