import csv
from pathlib import Path
from typing import Dict, List, Tuple


def load_local_cities(input_csv: Path) -> List[Tuple[str, float, float, int, str]]:
    rows: List[Tuple[str, float, float, int, str]] = []
    seen: Dict[str, int] = {}

    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"city", "lat", "lon"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Input CSV must contain columns: city, lat, lon")

        for row in reader:
            city_name = row["city"].strip()
            if not city_name:
                continue

            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (TypeError, ValueError):
                continue

            state_code = (row.get("state_code") or "NA").strip() or "NA"

            pop_text = (row.get("population") or "0").strip()
            try:
                population = int(pop_text) if pop_text else 0
            except ValueError:
                population = 0

            # Normalize naming to reduce collisions while staying backwards-compatible.
            label = city_name if "(" in city_name and ")" in city_name else f"{city_name} ({state_code})"
            key = label.lower()
            if key in seen:
                continue
            seen[key] = 1

            rows.append((label, lat, lon, population, state_code))

    rows.sort(key=lambda x: (-x[3], x[0]))
    return rows


def write_csv(rows: List[Tuple[str, float, float, int, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["city", "lat", "lon", "population", "state_code"])
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare India city coordinates CSV fully offline from a local input CSV."
    )
    parser.add_argument(
        "--input-csv",
        default="data/india_cities_sample.csv",
        help="Local input CSV path with at least city,lat,lon columns",
    )
    parser.add_argument(
        "--output-csv",
        default="data/india_cities_geonames.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only first N rows after sorting (0 = all)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    in_csv = (base_dir / args.input_csv).resolve()
    out_csv = (base_dir / args.output_csv).resolve()

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing local input CSV: {in_csv}")

    print(f"Reading local city dataset: {in_csv}")
    rows = load_local_cities(in_csv)

    if args.limit > 0:
        rows = rows[: args.limit]

    write_csv(rows, out_csv)
    print(f"Saved {len(rows)} India cities to: {out_csv}")
    print("Source: local offline CSV input")


if __name__ == "__main__":
    main()
