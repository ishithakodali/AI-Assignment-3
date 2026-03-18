import csv
import heapq
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


ROAD_FACTOR_DEFAULT = 1.30


@dataclass
class City:
    name: str
    lat: float
    lon: float


def read_cities(city_csv: Path) -> List[City]:
    cities: List[City] = []
    with city_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append(
                City(
                    name=row["city"].strip(),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                )
            )
    return cities


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def k_nearest_neighbors(cities: List[City], k: int) -> Dict[str, List[City]]:
    neighbors: Dict[str, List[City]] = {}
    for city in cities:
        dists: List[Tuple[float, City]] = []
        for other in cities:
            if city.name == other.name:
                continue
            d = haversine_km(city.lat, city.lon, other.lat, other.lon)
            dists.append((d, other))

        nearest = [c for _, c in heapq.nsmallest(k, dists, key=lambda x: x[0])]
        neighbors[city.name] = nearest

    return neighbors


def estimated_road_distance_km(city_a: City, city_b: City, road_factor: float) -> float:
    # Approximate road travel distance from straight-line geodesic distance.
    base = haversine_km(city_a.lat, city_a.lon, city_b.lat, city_b.lon)
    return base * road_factor


def build_edge_list(cities: List[City], k: int, road_factor: float) -> List[Tuple[str, str, float]]:
    name_to_city = {c.name: c for c in cities}
    neighbors = k_nearest_neighbors(cities, k)

    visited_pairs = set()
    edges: List[Tuple[str, str, float]] = []

    for city_name, nearby in neighbors.items():
        city = name_to_city[city_name]
        for nb in nearby:
            pair = tuple(sorted((city.name, nb.name)))
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)

            road_km = estimated_road_distance_km(city, nb, road_factor=road_factor)
            edges.append((pair[0], pair[1], round(road_km, 2)))
            print(f"OK: {pair[0]} <-> {pair[1]} = {road_km:.1f} km (offline estimate)")

    return edges


def write_edges(edge_csv: Path, edges: List[Tuple[str, str, float]]) -> None:
    edge_csv.parent.mkdir(parents=True, exist_ok=True)
    with edge_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["city_a", "city_b", "distance_km"])
        writer.writerows(edges)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build India road-edge graph from city coordinates fully offline."
    )
    parser.add_argument(
        "--city-csv",
        default="data/india_cities_sample.csv",
        help="Input city CSV with columns city,lat,lon",
    )
    parser.add_argument(
        "--edge-csv",
        default="data/india_road_edges_osm.csv",
        help="Output edge CSV path",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of nearest-neighbor connections per city",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only first N cities from input (0 = all)",
    )
    parser.add_argument(
        "--road-factor",
        type=float,
        default=ROAD_FACTOR_DEFAULT,
        help="Multiplier to approximate road distance from geodesic distance",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    city_csv = (base_dir / args.city_csv).resolve()
    edge_csv = (base_dir / args.edge_csv).resolve()

    if not city_csv.exists():
        raise FileNotFoundError(f"Missing city CSV: {city_csv}")

    cities = read_cities(city_csv)
    if args.limit > 0:
        cities = cities[: args.limit]

    if len(cities) < 2:
        raise ValueError("Need at least two cities to build a road graph.")

    k = max(1, min(args.k, len(cities) - 1))
    if args.road_factor <= 0:
        raise ValueError("--road-factor must be > 0")

    edges = build_edge_list(cities, k=k, road_factor=args.road_factor)
    write_edges(edge_csv, edges)

    print(f"\nSaved {len(edges)} edges to: {edge_csv}")
    print("Data source: local city coordinates + haversine-based offline road estimate.")


if __name__ == "__main__":
    main()
