# AI Assignment: Dijkstra + UGV Path Planning

This workspace contains solutions for all 3 requested points.

## Offline-Only Setup

This project now runs fully offline. No live APIs or web datasets are required.

## 1) Dijkstra/Uniform-Cost Search for India cities

### Files
- `build_india_road_graph.py`: builds a city-road graph fully offline using haversine-based road-distance estimates.
- `prepare_india_cities_geonames.py`: prepares/normalizes India city coordinates from a local CSV.
- `india_dijkstra.py`: runs Dijkstra from a source city to all other cities in the graph.
- `data/india_cities_sample.csv`: sample city list (editable; can be expanded toward all Indian cities).
- `data/india_cities_geonames.csv`: prepared city list generated from local input CSV.
- `data/india_road_edges_osm.csv`: generated edge list (created after running builder script).

### Data source
- Local CSV files inside `data/`.
- Road distances are offline estimates derived from geodesic distance using a configurable multiplier (`--road-factor`).

### Run
```powershell
pip install -r requirements.txt
python prepare_india_cities_geonames.py --input-csv data/india_cities_sample.csv
python build_india_road_graph.py --city-csv data/india_cities_geonames.csv --limit 120 --k 3
python india_dijkstra.py
```

Notes:
- For very large city lists, use `--limit` to control runtime.
- To make graph distances more conservative, increase `--road-factor` (default: `1.30`).

## 2) UGV shortest path in static obstacle grid (70x70)

### File
- `ugv_navigation.py`

### Method
- A* search on 4-connected grid.
- Three random obstacle density levels: LOW (10%), MEDIUM (20%), HIGH (30%).

### Measures of Effectiveness (MoE)
- Path cost (steps)
- Nodes expanded
- Runtime (ms)
- Turns (path smoothness)
- Success/Failure

## 3) UGV navigation with dynamic unknown obstacles

### Method in `ugv_navigation.py`
- Online replanning with Repeated A*.
- Obstacles move each cycle; planner replans from current UGV position to goal.

### Brief theory note: why D* Lite is often preferred in real deployments
- Repeated A* replans from scratch each cycle, which is simple but can re-expand many of the same states.
- D* Lite is an incremental heuristic search that reuses previous search effort after map changes.
- It maintains `g` and one-step lookahead `rhs` values, and only repairs inconsistent nodes when costs/obstacles change.
- Result: for frequent local changes in dynamic environments, D* Lite typically replans faster than full-from-scratch A* while preserving shortest-path optimality under the same cost model.

### Dynamic MoE
- Success
- Total traversed steps
- Ideal steps (no obstacles)
- Detour ratio
- Number of replans
- Total nodes expanded
- Total planning runtime

## Quick Run (Problems 2 and 3)
```powershell
python ugv_navigation.py
```
