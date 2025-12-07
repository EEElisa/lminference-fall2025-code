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
    Parse a graph problem prompt and extract edges, N (number of nodes), and P (number of paths).

    Args:
        prompt: The graph problem prompt text

    Returns:
        Dictionary with 'edges', 'N', and 'P' keys, or None if parsing fails
    """
    # Extract number of nodes: "You are given a directed graph with N nodes"
    nodes_match = re.search(r'directed graph with (\d+) nodes', prompt)
    if not nodes_match:
        return None
    N = int(nodes_match.group(1))

    # Extract number of paths to find: "Find the top P shortest path"
    paths_match = re.search(r'Find the top (\d+) shortest path', prompt)
    if not paths_match:
        return None
    P = int(paths_match.group(1))

    # Extract edges: "source -> target, weight: X"
    edges = []
    edge_pattern = r'(\d+) -> (\d+), weight: (\d+)'
    for match in re.finditer(edge_pattern, prompt):
        src = int(match.group(1))
        dst = int(match.group(2))
        weight = int(match.group(3))
        edges.append((src, dst, weight))

    if not edges:
        return None

    return {
        'edges': edges,
        'N': N,
        'P': P
    }


def find_top_p_shortest_paths(edges: List[Tuple[int, int, int]], N: int, P: int) -> Dict[str, Any]:
    """
    Find the top P shortest paths from node 0 to node N-1 using modified Dijkstra's algorithm.

    Args:
        edges: List of (source, target, weight) tuples
        N: Number of nodes
        P: Number of paths to find

    Returns:
        Dictionary with 'paths' (list of lists) and 'weights' (list of ints)
    """
    # Build adjacency list
    graph = {i: [] for i in range(N)}
    for src, dst, weight in edges:
        graph[src].append((dst, weight))

    # Use modified Dijkstra's algorithm to find top P paths
    # Each state is (cost, path)
    pq = [(0, [0])]  # (cost, path)
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
        if current_node == N - 1:
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

    # Compute the shortest paths
    result = find_top_p_shortest_paths(edges, N, P)
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

    # Compute the shortest paths
    result = find_top_p_shortest_paths(edges, N, P)
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
