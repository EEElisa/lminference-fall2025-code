#!/usr/bin/env python3
"""Quick script to check dataset format"""

from datasets import load_dataset

print("="*60)
print("MMLU_Med Dataset Sample")
print("="*60)

# Load MMLU dataset
ds = load_dataset("vashistht/11763_datasets", "mmlu_med", split="dev_test")

# Print first 3 examples
for i in range(3):
    print(f"\n--- Example {i} ---")
    example = ds[i]
    print(f"Question: {example['question']}")
    print(f"Choices: {example['choices']}")
    print(f"Answer (raw): {example['answer']}")
    print(f"Answer type: {type(example['answer'])}")
    if isinstance(example['answer'], int):
        print(f"Answer as letter: {chr(65 + example['answer'])}")

print("\n" + "="*60)
print("Graph Dataset Sample")
print("="*60)

# Load Graph dataset
ds_graph = load_dataset("vashistht/11763_datasets", "graph_dev", split="dev_test")

print(f"\n--- Example 0 ---")
example = ds_graph[0]
print("All keys:", example.keys())
for key, value in example.items():
    if isinstance(value, list) and len(str(value)) > 150:
        print(f"{key}: {str(value)[:150]}...")
    else:
        print(f"{key}: {value}")
