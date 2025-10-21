import json
import re
import os
import time
import random
import heapq
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from datasets import load_dataset


choices = ["A", "B", "C", "D"]

SYS_MSG ="""Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?” and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"""


@dataclass
class DecodingStrategy:
    name: str
    do_sample: bool = False
    temperature: float = 1.0
    typical_p: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_beams: int = 1


@dataclass
class PathInfo:
    """Information about a single path"""
    path: List[int]
    weight: int


@dataclass
class GraphPathSolution:
    """Solution containing top-P paths with their weights"""
    paths: List[PathInfo]


def load_custom_dataset(dataset_name: str):
    # load datasets from Hugging Face
    if dataset_name == "MMLU":
        raw_dataset = load_dataset("vashistht/11763_datasets", "mmlu_med")
        return [example for example in raw_dataset["dev_test"]]
    elif dataset_name == "InfoBench":
        raw_dataset = load_dataset("vashistht/11763_datasets", "infobench")
        return [example for example in raw_dataset["dev_test"]]
    elif dataset_name == "N-best":
        raw_dataset = load_dataset("vashistht/11763_datasets", "graph_dev")
        return [example for example in raw_dataset["dev_test"]]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_mmlu_example(example: Dict[str, Any], include_answer: bool = False) -> str:
    prompt = f"The following is a multiple choice question (with answers) about {format_subject(example['subject'])}. Output the answer in the format of \"The answer is (X)\" at the end.\n\n"
    prompt += f"Question: {example['question']}\n Options:"
    these_choices = example["choices"]
    for i in range(len(these_choices)):
        prompt += f"\n{choices[i]}. {these_choices[i]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {choices[example['answer']]}\n\n"
    return prompt

def format_infobench_example(example: Dict[str, Any]) -> str:
    instruction = example['instruction']
    generated_text = example['generated_text']
    input_text = example['input']
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nGenerated Text: {generated_text}"
    return prompt

def generate_problem_prompt(task_name: str, example: Dict[str, Any]) -> str:
    if task_name == "MMLU":
        return format_mmlu_example(example, include_answer=False)
    elif task_name == "InfoBench":
        return f"Instruction: {example['instruction']}\nQuestion: {example['input']}\nGeneration:"
    elif task_name == "N-best":
        # edges is a list of [src, dst, weight] or (src, dst, weight)
        edges = [(edge[0], edge[1], edge[2]) for edge in example['edges']]
        N = example['graph_params']['N']
        P = example['graph_params']['P']
        prompt = f"""You are given a directed graph with {N} nodes (numbered 0 to {N-1}) and the following edges:

Edges (source -> target, weight):
"""
        for src, dst, weight in edges:
            prompt += f"{src} -> {dst}, weight: {weight}\n"
        prompt += f"""
Find the top {P} shortest path{'s' if P > 1 else ''} from node 0 to node {N-1}.

Return a JSON object with the following structure:
{{
  "paths": [
    [0, ..., {N-1}],
    ...
  ],
  "weights": [
    10,
    ...
  ]
}}
"""
        return prompt
    else:
        raise ValueError(f"Unknown dataset name: {task_name}")


def extract_answer(text):
    text = re.sub(r'\$\\boxed\{([A-Za-z])\}\$', r'\1', text)
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        pattern = r"option \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None

def find_top_p_paths(edges: List[Tuple[int, int, int]], N: int, P: int) -> GraphPathSolution:
    graph = {i: [] for i in range(N)}
    for src, dst, weight in edges:
        graph[src].append((dst, weight))

    pq = [(0, [0])]
    paths_found = []
    
    while pq and len(paths_found) < P:
        cost, path = heapq.heappop(pq)
        current_node = path[-1]
        
        if current_node == N - 1:
            paths_found.append(PathInfo(path=path, weight=cost))
            continue
        
        for neighbor, edge_weight in graph.get(current_node, []):
            if neighbor not in path:
                new_cost = cost + edge_weight
                new_path = path + [neighbor]
                heapq.heappush(pq, (new_cost, new_path))
    
    return GraphPathSolution(paths=paths_found)

def convert_llm_response_to_solution(llm_response: str, task_name: str) -> Any:
    if task_name == "MMLU":
        return extract_answer(llm_response.replace('**', ''))
    elif task_name == "InfoBench":
        return llm_response
    else:
        return llm_response

@dataclass
class HuggingFaceModel:
    tokenizer: Any
    model: Any
    generation_args: Dict[str, Any]


def load_hf_model(model_name: str, device: str) -> HuggingFaceModel:
    if device == "auto" and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Loading model on CPU, which will be much slower.")
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    generation_args = {
        "max_new_tokens": 512,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    return HuggingFaceModel(tokenizer=tokenizer, model=model, generation_args=generation_args)


def query_llm(
    hf_model: HuggingFaceModel,
    prompts: List[str],
    decoding_strategy: DecodingStrategy
) -> List[str]:
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

    tokenized = hf_model.tokenizer.apply_chat_template(
        messages_list,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    )

    input_ids = tokenized.to(hf_model.model.device)
    attention_mask = (input_ids != hf_model.tokenizer.pad_token_id).to(hf_model.model.device)

    gen_args = hf_model.generation_args.copy()
    gen_args.update({
        "temperature": decoding_strategy.temperature,
        "do_sample": decoding_strategy.do_sample,
        "top_p": decoding_strategy.top_p,
        "top_k": decoding_strategy.top_k,
        "typical_p": decoding_strategy.typical_p,
        "num_beams": decoding_strategy.num_beams,
        "attention_mask": attention_mask,
    })

    # Handle greedy and beam search specific parameters
    if decoding_strategy.name in ["Greedy decoding", "Beam search"]:
        gen_args["do_sample"] = False
        if decoding_strategy.name == "Greedy decoding":
            gen_args["num_beams"] = 1
        elif decoding_strategy.name == "Beam search":
            gen_args["early_stopping"] = True
    else:
        gen_args["num_beams"] = 1

    output = hf_model.model.generate(
        input_ids,
        **gen_args,
    )
    
    # Decode the responses, handling the different output lengths from the batch
    responses = []
    for i in range(output.shape[0]):
        # Find the start of the generated text
        start_index = input_ids.shape[1]
        response_tensor = output[i, start_index:]
        response_text = hf_model.tokenizer.decode(response_tensor, skip_special_tokens=True)
        responses.append(response_text)
        
    return responses


def query_llm_with_function_call(
    hf_model: HuggingFaceModel,
    prompt: str,
) -> Dict[str, Any]:
    llm_response = query_llm(hf_model, [prompt], DecodingStrategy(name="Function Call", num_beams=4))
    llm_response = llm_response[0]
    
    json_match = re.search(r"```json\s*(\{.*\})\s*```", llm_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            print("Failed to parse JSON from LLM response.")
            return {"paths": [], "weights": []}
    
    print("No JSON object found in LLM response.")
    return {"paths": [], "weights": []}


def info_bench_eval(example: Dict[str, Any], predicted_solution: str, openai_client: OpenAI) -> float:
    message = []
    answer = ""
    input_task = example['input']
    output = predicted_solution
    eval_model = "gpt-4o-mini"
    
    for question in example["decomposed_questions"]:
        if len(message) == 0:
            if input_task:
                content =  f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
            else:
                content =  f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
        else:
            content = f"{question}\n"
        message.append({"role": "user", "content": content})
        success = False
        while not success:
            try:
                completion = openai_client.chat.completions.create(
                        model=eval_model,
                        messages=message,
                        temperature=1.0,
                    )
                generation = completion.choices[0].message.content
                message.append(
                        {"role": "assistant", "content": generation})
                if generation.lower().startswith("yes"):
                    answer += "Yes\n"
                elif generation.lower().startswith("no"):
                    answer += "No\n"
                else:
                    answer += "None\n"
                success = True
            except Exception as e:
                print("ERROR!", e)
                print("Retry!")
                time.sleep(5)
    
    bool_results = []
    for i in answer.strip().split('\n'):
        if i == "Yes":
            bool_results.append(True)
        elif i == "No":
            bool_results.append(False)
        else:
            bool_results.append(None)

    count_true = bool_results.count(True)
    count_valid = len([x for x in bool_results if x is not None])
    return count_true / count_valid if count_valid > 0 else 0.0

def evaluate_solution(example: Dict[str, Any], predicted_solution: Any, task_name: str, openai_client: OpenAI) -> float:
    if task_name == "MMLU":
        correct_solution = choices[example["answer"]]
        return 1.0 if predicted_solution == correct_solution else 0.0
    elif task_name == "InfoBench":
        return info_bench_eval(example, predicted_solution, openai_client)
    elif task_name == "N-best":
        # edges is a list of [src, dst, weight] or (src, dst, weight)
        edges = [(edge[0], edge[1], edge[2]) for edge in example['edges']]
        P = example['graph_params']['P']
        correct_solution = find_top_p_paths(edges, example['graph_params']['N'], P)
        
        correct_paths = {(tuple(path.path), path.weight) for path in correct_solution.paths}
        predicted_paths = {(tuple(path.path), path.weight) for path in predicted_solution.paths}
        
        matches = len(correct_paths.intersection(predicted_paths))
        return matches / P if P > 0 else 0.0
    else:
        raise ValueError(f"Unknown task name: {task_name}")

def run_evaluation(
    examples: List[Dict[str, Any]],
    hf_model: HuggingFaceModel,
    openai_client: OpenAI,
    task: str,
    decoding_strategy: DecodingStrategy = None,
    batch_size: int = 1
) -> Dict[str, Any]:
    print(f"Running evaluation for model: {hf_model.model.name_or_path} with strategy: {decoding_strategy.name}, batch size: {batch_size}")
    total_score = 0.0
    results = []
    
    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i + batch_size]
        prompts = [generate_problem_prompt(task, example) for example in batch]

        if task == "N-best":
            # N-best is a function-call like task, so we still process one by one
            # as it doesn't fit into the batched decoding paradigm easily.
            for example in batch:
                prompt = generate_problem_prompt(task, example)
                llm_response = query_llm_with_function_call(hf_model, prompt)
                predicted_solution = GraphPathSolution(paths=[PathInfo(path=p, weight=w) for p, w in zip(llm_response.get("paths", []), llm_response.get("weights", []))])
                score = evaluate_solution(example, predicted_solution, task, openai_client)
                total_score += score
                results.append({"example_id": i, "score": score})
        else:
            llm_responses = query_llm(
                hf_model,
                prompts,
                decoding_strategy,
            )

            for j, example in enumerate(batch):
                predicted_solution = convert_llm_response_to_solution(llm_responses[j], task)
                score = evaluate_solution(example, predicted_solution, task, openai_client)
                total_score += score
                results.append({"example_id": i+j, "score": score})
    
    average_score = total_score / len(examples) if examples else 0.0
    
    return {
        "model": hf_model.model.name_or_path,
        "average_score": average_score,
        "total_examples": len(examples),
        "decoding_strategy": asdict(decoding_strategy),
        "results": results
    }

def run_benchmarking(
    strategies: List[DecodingStrategy],
    models: List[str],
    openai_client: OpenAI,
    task: str,
    batch_size: int
):
    print("Generating test examples...")
    examples = load_custom_dataset(task)
    
    for model_name in models:
        print(f"\n==========================================")
        print(f"     Starting benchmark for model: {model_name}     ")
        print(f"==========================================\n")
        
        try:
            # Use 'auto' to let Hugging Face choose the device, or 'cpu'
            hf_model = load_hf_model(model_name, "auto") 
        except Exception as e:
            print(f"Failed to load Hugging Face model {model_name}: {e}")
            print("Attempting to load model on CPU as a fallback.")
            try:
                hf_model = load_hf_model(model_name, "cpu")
            except Exception as e_cpu:
                print(f"Failed to load Hugging Face model on CPU as well: {e_cpu}")
                continue
            
        benchmark_results = {}
        
        # N-best task does not use different decoding strategies, so we run it only once
        if task == "N-best":
            strategy = DecodingStrategy(name="Function Call", num_beams=4)
            results = run_evaluation(
                examples=examples,
                hf_model=hf_model,
                openai_client=openai_client,
                task=task,
                decoding_strategy=strategy,
                batch_size=1 # N-best is processed one by one
            )
            benchmark_results[strategy.name] = results
            print(f"Average score for {strategy.name}: {results['average_score']:.2f}")
        else:
            for strategy in strategies:
                print(f"\n--- Running benchmark for strategy: {strategy.name} ---")
                results = run_evaluation(
                    examples=examples,
                    hf_model=hf_model,
                    openai_client=openai_client,
                    task=task,
                    decoding_strategy=strategy,
                    batch_size=batch_size
                )
                benchmark_results[strategy.name] = results
                print(f"Average score for {strategy.name}: {results['average_score']:.2f}")

        print("\n\n--- Benchmark Summary for all strategies on model: {model_name} ---")
        for name, result in benchmark_results.items():
            print(f"Strategy: {name:<20} | Average Score: {result['average_score']:.2f}")
        
        model_filename = model_name.replace('/', '-')
        with open(f"benchmark_summary_{task}_{model_filename}.json", "w") as f:
            json.dump(
                {name: {"average_score": result["average_score"]} for name, result in benchmark_results.items()}, 
                f, 
                indent=4
            )
        print(f"\nBenchmark summary for {model_name} saved to file.")


if __name__ == "__main__":
    # hf_token = os.getenv("HF_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # if not hf_token:
    #     print("Please set HF_TOKEN environment variable. You can get a token from https://huggingface.co/settings/tokens")
    #     exit(1)
    
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not set. InfoBench evaluation will not work.")
        openai_client = None
    else:
        openai_client = OpenAI(api_key=openai_api_key)

    task = os.getenv("TASK", "MMLU")
    batch_size = int(os.getenv("BATCH_SIZE", 8))

    models_to_test = [
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B"
    ]
    
    decoding_strategies_to_test = [
        DecodingStrategy(name="Default generation settings", do_sample=True, temperature=0.7, top_p=0.95),
        DecodingStrategy(name="Greedy decoding", do_sample=False, num_beams=1),
        DecodingStrategy(name="Temperature sampling (tau=0.25)", do_sample=True, temperature=0.25, top_p=1.0),
        DecodingStrategy(name="Temperature sampling (tau=1.5)", do_sample=True, temperature=1.5, top_p=1.0),
        DecodingStrategy(name="Beam search (beam width=3)", do_sample=False, num_beams=3),
        DecodingStrategy(name="Beam search (beam width=25)", do_sample=False, num_beams=25),
        DecodingStrategy(name="Locally typical sampling (typical_p=0.95)", do_sample=True, temperature=1.0, typical_p=0.95),
    ]

    run_benchmarking(
        strategies=decoding_strategies_to_test,
        models=models_to_test,
        openai_client=openai_client,
        task=task,
        batch_size=batch_size
    )