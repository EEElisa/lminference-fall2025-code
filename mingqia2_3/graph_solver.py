#!/usr/bin/env python3
"""
Graph Path Solver - Direct computation without LLM reasoning.

This module parses graph shortest path problems from text prompts and
computes the answer using Dijkstra's algorithm instead of calling an LLM.
"""

import re
import heapq
from typing import List, Tuple, Dict, Any, Optional


def parse_graph_prompt(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Parse a graph problem prompt and extract edges, node count, source/target, and top-P.
    More robust to minor noise/formatting differences.
    """
    # Extract number of nodes: "directed graph with N nodes"
    nodes_match = re.search(r'directed graph with\s+(\d+)\s+nodes', prompt, re.IGNORECASE)
    N = int(nodes_match.group(1)) if nodes_match else None

    # Extract source/target: "from node X to node Y"
    st_match = re.search(r'from node\s+(\d+)\s+to node\s+(\d+)', prompt, re.IGNORECASE)
    src = int(st_match.group(1)) if st_match else 0
    tgt = int(st_match.group(2)) if st_match else (N - 1 if N is not None else None)

    # Extract number of paths: "Find the top P shortest path(s)".
    # If not specified, default to 1.
    paths_match = re.search(r'Find the top\s+(\d+)\s+shortest path', prompt, re.IGNORECASE)
    P = int(paths_match.group(1)) if paths_match else 1

    # Extract edges: "source -> target, weight: X" (allow spaces)
    edges = []
    edge_pattern = r'(\d+)\s*->\s*(\d+)\s*,\s*weight:\s*([-\d]+)'
    for match in re.finditer(edge_pattern, prompt, re.IGNORECASE):
        src_e = int(match.group(1))
        dst_e = int(match.group(2))
        weight = int(match.group(3))
        edges.append((src_e, dst_e, weight))

    if not edges:
        return None

    # Infer N if missing from prompt: 1 + max node id seen
    if N is None:
        N = max(max(s, d) for s, d, _ in edges) + 1
    if tgt is None:
        tgt = N - 1

    return {'edges': edges, 'N': N, 'P': P, 'src': src, 'tgt': tgt}


def find_top_p_shortest_paths(edges: List[Tuple[int, int, int]], N: int, P: int, src: int = 0, tgt: Optional[int] = None) -> Dict[str, Any]:
    """
    Find the top P shortest paths from node `src` to node `tgt` (default N-1)
    using a modified Dijkstra's algorithm.

    Args:
        edges: List of (source, target, weight) tuples
        N: Number of nodes
        P: Number of paths to find

    Returns:
        Dictionary with 'paths' (list of lists) and 'weights' (list of ints)
    """
    if tgt is None:
        tgt = N - 1

    # Build adjacency list
    graph = {i: [] for i in range(N)}
    for s, dst, weight in edges:
        graph[s].append((dst, weight))

    # Use modified Dijkstra's algorithm to find top P paths
    # Each state is (cost, path)
    start = src  # preserve caller-provided start
    pq = [(0, [start])]  # (cost, path)
    paths_found = []
    visited_states = set()

    while pq and len(paths_found) < P:
        cost, path = heapq.heappop(pq)
        current_node = path[-1]

        # Create a state key to avoid revisiting the same (node, path_length) combination
        state_key = (current_node, len(path))
        if state_key in visited_states:
            continue
        visited_states.add(state_key)

        # If we reached the target node, add this path to results
        if current_node == tgt:
            paths_found.append({
                'path': path,
                'weight': cost
            })
            continue

        # Explore neighbors
        for neighbor, edge_weight in graph[current_node]:
            if neighbor not in path:  # Avoid cycles
                new_cost = cost + edge_weight
                new_path = path + [neighbor]
                heapq.heappush(pq, (new_cost, new_path))

    # Convert to the format expected by the grader
    paths = [p['path'] for p in paths_found]
    weights = [p['weight'] for p in paths_found]

    return {
        'paths': paths,
        'weights': weights
    }


def solve_graph_problem(prompt: str) -> str:
    """
    Solve a graph shortest path problem by parsing the prompt and computing the answer.

    Args:
        prompt: The graph problem prompt text

    Returns:
        String in the format expected by the grader (function call format)
    """
    # Parse the prompt
    parsed = parse_graph_prompt(prompt)
    if parsed is None:
        return "Error: Could not parse graph problem"

    edges = parsed['edges']
    N = parsed['N']
    P = parsed['P']
    src = parsed.get('src', 0)
    tgt = parsed.get('tgt', N - 1)

    # Compute the shortest paths
    result = find_top_p_shortest_paths(edges, N, P, src=src, tgt=tgt)
    paths = result['paths']
    weights = result['weights']

    # Format as function call (this is what the grader expects)
    return f"submit_paths(paths={paths}, weights={weights})"


def solve_graph_problem_dict(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Solve a graph shortest path problem and return the result as a dictionary.

    Args:
        prompt: The graph problem prompt text

    Returns:
        Dictionary with 'paths' and 'weights' keys, or None if parsing fails
    """
    # Parse the prompt
    parsed = parse_graph_prompt(prompt)
    if parsed is None:
        return None

    edges = parsed['edges']
    N = parsed['N']
    P = parsed['P']
    src = parsed.get('src', 0)
    tgt = parsed.get('tgt', N - 1)

    # Compute the shortest paths
    result = find_top_p_shortest_paths(edges, N, P, src=src, tgt=tgt)
    return result


if __name__ == "__main__":
    # Test with a simple example
    test_prompt = """You are given a directed graph with 5 nodes (numbered 0 to 4) and the following edges:

Edges (source -> target, weight):
0 -> 1, weight: 10
0 -> 2, weight: 20
1 -> 3, weight: 5
2 -> 3, weight: 3
3 -> 4, weight: 2
1 -> 4, weight: 50

Find the top 2 shortest paths from node 0 to node 4.

Return your answer by calling the submit_paths function with:
- paths: A list of paths, where each path is a list of node indices
- weights: A list of corresponding path weights

For example, if the shortest path is [0, 2, 4] with weight 10, call:
submit_paths(paths=[[0, 2, 4]], weights=[10])
"""

    result = solve_graph_problem(test_prompt)
    print("Result:", result)

    result_dict = solve_graph_problem_dict(test_prompt)
    print("Result dict:", result_dict)
