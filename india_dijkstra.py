import csv
import heapq
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


Graph = Dict[str, List[Tuple[str, float]]]


@dataclass
class DijkstraResult:
    source: str
    distances: Dict[str, float]
    previous: Dict[str, str]


def load_graph_from_edge_csv(edge_csv: Path) -> Graph:
    graph: Graph = defaultdict(list)
    with edge_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"city_a", "city_b", "distance_km"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                "Edge CSV must contain columns: city_a, city_b, distance_km"
            )

        for row in reader:
            a = row["city_a"].strip()
            b = row["city_b"].strip()
            w = float(row["distance_km"])
            graph[a].append((b, w))
            graph[b].append((a, w))

    return dict(graph)


def dijkstra_all_destinations(graph: Graph, source: str) -> DijkstraResult:
    """
    Executes Dijkstra's algorithm (Uniform-Cost Search).

    Since all edge weights (road distances) are positive and no heuristic is
    used (h(n)=0), this is equivalent to Uniform-Cost Search in AI literature.
    """
    if source not in graph:
        raise ValueError(f"Source city '{source}' does not exist in graph.")

    distances = {node: float("inf") for node in graph}
    previous: Dict[str, str] = {}
    distances[source] = 0.0

    pq: List[Tuple[float, str]] = [(0.0, source)]

    while pq:
        curr_dist, node = heapq.heappop(pq)
        if curr_dist > distances[node]:
            continue

        for neighbor, weight in graph[node]:
            alt = curr_dist + weight
            if alt < distances[neighbor]:
                distances[neighbor] = alt
                previous[neighbor] = node
                heapq.heappush(pq, (alt, neighbor))

    return DijkstraResult(source=source, distances=distances, previous=previous)


def reconstruct_path(previous: Dict[str, str], source: str, destination: str) -> List[str]:
    if source == destination:
        return [source]

    if destination not in previous:
        return []

    path = [destination]
    current = destination
    while current != source:
        current = previous.get(current)
        if current is None:
            return []
        path.append(current)

    path.reverse()
    return path


def print_shortest_paths(result: DijkstraResult) -> None:
    print(f"\nDijkstra shortest path tree from: {result.source}")
    print("-" * 72)
    print(f"{'Destination':<20} {'Distance (km)':<16} Path")
    print("-" * 72)

    for destination in sorted(result.distances):
        dist = result.distances[destination]
        if dist == float("inf"):
            print(f"{destination:<20} {'UNREACHABLE':<16} -")
            continue

        path = reconstruct_path(result.previous, result.source, destination)
        path_text = " -> ".join(path) if path else "-"
        print(f"{destination:<20} {dist:<16.1f} {path_text}")


def read_source_city(graph: Graph) -> str:
    available = sorted(graph.keys())
    lookup = {name.lower(): name for name in available}
    base_name_map: Dict[str, List[str]] = {}

    for name in available:
        base = name.split("(", 1)[0].strip().lower()
        base_name_map.setdefault(base, []).append(name)

    print("Available cities:")
    print(", ".join(available))

    while True:
        source_input = input("\nEnter source city: ").strip()
        if not source_input:
            print("Source city cannot be empty.")
            continue

        source_city = lookup.get(source_input.lower())
        if source_city is not None:
            return source_city

        base_matches = base_name_map.get(source_input.lower(), [])
        if len(base_matches) == 1:
            return base_matches[0]

        if len(base_matches) > 1:
            print("Multiple matches found. Please enter one of:")
            print(", ".join(sorted(base_matches)))
            continue

        if source_city is None:
            print("City not found in graph. Please enter a city from the list above.")
            continue


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    edge_csv = base_dir / "data" / "india_road_edges_osm.csv"

    if not edge_csv.exists():
        raise FileNotFoundError(
            "Missing data/india_road_edges_osm.csv. Build it first with "
            "build_india_road_graph.py"
        )

    graph = load_graph_from_edge_csv(edge_csv)

    source_city = read_source_city(graph)

    result = dijkstra_all_destinations(graph, source_city)
    print_shortest_paths(result)


if __name__ == "__main__":
    main()
