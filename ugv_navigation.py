import heapq
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


GridPos = Tuple[int, int]


@dataclass
class SearchResult:
    path: List[GridPos]
    cost: float
    nodes_expanded: int
    runtime_ms: float


def manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors_4(grid_size: int, node: GridPos) -> Iterable[GridPos]:
    r, c = node
    cand = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for nr, nc in cand:
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            yield nr, nc


def reconstruct(came_from: Dict[GridPos, GridPos], start: GridPos, goal: GridPos) -> List[GridPos]:
    if goal == start:
        return [start]
    if goal not in came_from:
        return []

    path = [goal]
    cur = goal
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def a_star(
    grid_size: int,
    start: GridPos,
    goal: GridPos,
    obstacles: Set[GridPos],
) -> SearchResult:
    t0 = time.perf_counter()

    if start in obstacles or goal in obstacles:
        return SearchResult([], float("inf"), 0, 0.0)

    open_heap: List[Tuple[int, int, GridPos]] = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0, start))

    g_score: Dict[GridPos, int] = {start: 0}
    came_from: Dict[GridPos, GridPos] = {}
    visited: Set[GridPos] = set()
    expanded = 0

    while open_heap:
        _, g_curr, current = heapq.heappop(open_heap)
        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current == goal:
            path = reconstruct(came_from, start, goal)
            t1 = time.perf_counter()
            return SearchResult(path, float(g_curr), expanded, (t1 - t0) * 1000)

        for nb in neighbors_4(grid_size, current):
            if nb in obstacles:
                continue

            tentative_g = g_curr + 1
            if tentative_g < g_score.get(nb, 10**9):
                g_score[nb] = tentative_g
                came_from[nb] = current
                f = tentative_g + manhattan(nb, goal)
                heapq.heappush(open_heap, (f, tentative_g, nb))

    t1 = time.perf_counter()
    return SearchResult([], float("inf"), expanded, (t1 - t0) * 1000)


def generate_obstacles(
    grid_size: int,
    density: float,
    start: GridPos,
    goal: GridPos,
    rng: random.Random,
) -> Set[GridPos]:
    obstacles: Set[GridPos] = set()
    for r in range(grid_size):
        for c in range(grid_size):
            p = (r, c)
            if p in (start, goal):
                continue
            if rng.random() < density:
                obstacles.add(p)
    return obstacles


def turns_in_path(path: List[GridPos]) -> int:
    if len(path) < 3:
        return 0

    turns = 0
    for i in range(2, len(path)):
        d1 = (path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1])
        d2 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if d1 != d2:
            turns += 1
    return turns


def print_grid_snapshot(
    grid_size: int,
    obstacles: Set[GridPos],
    path: List[GridPos],
    start: GridPos,
    goal: GridPos,
    max_size: int = 25,
) -> None:
    # For readability, print only a centered crop when grid is large.
    if grid_size > max_size:
        print(f"Grid {grid_size}x{grid_size} too large to print fully. Showing top-left {max_size}x{max_size}.")
        rows = cols = max_size
    else:
        rows = cols = grid_size

    path_set = set(path)

    for r in range(rows):
        row_chars = []
        for c in range(cols):
            p = (r, c)
            if p == start:
                row_chars.append("S")
            elif p == goal:
                row_chars.append("G")
            elif p in path_set:
                row_chars.append("*")
            elif p in obstacles:
                row_chars.append("#")
            else:
                row_chars.append(".")
        print("".join(row_chars))


def read_user_start_goal(grid_size: int) -> Tuple[GridPos, GridPos]:
    print("\nEnter user-specified coordinates (0 to 69 for a 70x70 grid).")

    while True:
        try:
            start_x = int(input("Enter start X: "))
            start_y = int(input("Enter start Y: "))
            goal_x = int(input("Enter goal X: "))
            goal_y = int(input("Enter goal Y: "))
        except ValueError:
            print("Invalid input. Please enter integer values.")
            continue

        start = (start_x, start_y)
        goal = (goal_x, goal_y)

        in_bounds = all(0 <= v < grid_size for v in (start_x, start_y, goal_x, goal_y))
        if not in_bounds:
            print(f"Coordinates must be between 0 and {grid_size - 1}.")
            continue

        if start == goal:
            print("Start and goal must be different nodes.")
            continue

        return start, goal


def run_static_obstacle_experiment(start: GridPos, goal: GridPos) -> None:
    print("\n=== Problem 2: UGV shortest path with static obstacles ===")

    grid_size = 70
    density_levels = {
        "LOW": 0.10,
        "MEDIUM": 0.20,
        "HIGH": 0.30,
    }

    trials_per_level = 10

    print(f"Grid size: {grid_size}x{grid_size}, start={start}, goal={goal}")
    print("\nMeasures of Effectiveness (MoE):")
    print("1) Path cost (shortest distance in grid steps)")
    print("2) Nodes expanded")
    print("3) Runtime in milliseconds")
    print("4) Path smoothness (number of turns)")
    print("5) Success/Failure")

    for idx, (label, density) in enumerate(density_levels.items(), start=1):
        successes = 0
        total_cost = 0.0
        total_expanded = 0
        total_runtime = 0.0
        total_turns = 0
        snapshot = None

        for t in range(trials_per_level):
            rng = random.Random(1000 * idx + t)
            obstacles = generate_obstacles(grid_size, density, start, goal, rng)
            result = a_star(grid_size, start, goal, obstacles)
            success = bool(result.path)

            if success:
                successes += 1
                total_cost += result.cost
                total_expanded += result.nodes_expanded
                total_runtime += result.runtime_ms
                total_turns += turns_in_path(result.path)

                if snapshot is None:
                    snapshot = (obstacles, result.path)

        print(f"\n--- Density: {label} ({density:.0%}) ---")
        print(f"Trials: {trials_per_level}")
        print(f"Success rate: {successes}/{trials_per_level} ({(100 * successes / trials_per_level):.1f}%)")

        if successes > 0:
            print(f"Avg path cost: {total_cost / successes:.1f}")
            print(f"Avg nodes expanded: {total_expanded / successes:.1f}")
            print(f"Avg runtime: {total_runtime / successes:.2f} ms")
            print(f"Avg turns: {total_turns / successes:.1f}")
        else:
            print("No valid path found.")
            print("(All trials failed at this density.)")

        if label == "LOW" and snapshot is not None:
            print("\nPath trace snapshot (LOW density):")
            print_grid_snapshot(grid_size, snapshot[0], snapshot[1], start, goal)


# -------------------- Problem 3: Dynamic obstacles --------------------

def move_dynamic_obstacles(
    grid_size: int,
    obstacles: Set[GridPos],
    start: GridPos,
    goal: GridPos,
    rng: random.Random,
    move_fraction: float = 0.08,
) -> Set[GridPos]:
    if not obstacles:
        return obstacles

    obstacles = set(obstacles)
    count_to_move = max(1, int(len(obstacles) * move_fraction))
    moving = rng.sample(list(obstacles), k=min(count_to_move, len(obstacles)))

    for old in moving:
        obstacles.discard(old)

        candidates = list(neighbors_4(grid_size, old))
        rng.shuffle(candidates)

        placed = False
        for nb in candidates:
            if nb in obstacles or nb in (start, goal):
                continue
            obstacles.add(nb)
            placed = True
            break

        if not placed:
            if old not in (start, goal):
                obstacles.add(old)

    return obstacles


def sense_obstacles(
    current: GridPos,
    true_obstacles: Set[GridPos],
    sensor_range: int = 5,
) -> Set[GridPos]:
    """Returns obstacles within the UGV sensor range using Manhattan distance."""
    sensed: Set[GridPos] = set()
    for obs in true_obstacles:
        if manhattan(current, obs) <= sensor_range:
            sensed.add(obs)
    return sensed


def run_dynamic_obstacle_navigation(start: GridPos, goal: GridPos) -> None:
    print("\n=== Problem 3: UGV navigation with dynamic unknown obstacles ===")
    print("Approach: online replanning (Repeated A*) whenever path becomes invalid.")

    grid_size = 70
    found_success = False

    for scenario_seed in range(1, 21):
        rng = random.Random(700 + scenario_seed)

        # Moderate density with guaranteed initial reachability.
        for _ in range(100):
            obstacles = generate_obstacles(grid_size, 0.12, start, goal, rng)
            if a_star(grid_size, start, goal, obstacles).path:
                break
        else:
            continue

        current = start
        total_path = [current]
        replans = 0
        total_expanded = 0
        total_runtime_ms = 0.0
        steps_taken = 0
        max_steps = 7000
        blocked_replans = 0
        known_obstacles: Set[GridPos] = set()
        sensor_range = 5

        while current != goal and steps_taken < max_steps:
            # Environment changes before each planning cycle.
            obstacles = move_dynamic_obstacles(
                grid_size,
                obstacles,
                start,
                goal,
                rng,
                move_fraction=0.02,
            )

            # UGV updates only the portion of map it can sense right now.
            newly_sensed = sense_obstacles(current, obstacles, sensor_range)
            known_obstacles.update(newly_sensed)

            # Replan using only known obstacles, not the full true map.
            plan = a_star(grid_size, current, goal, known_obstacles)
            replans += 1
            total_expanded += plan.nodes_expanded
            total_runtime_ms += plan.runtime_ms

            if not plan.path:
                blocked_replans += 1
                if blocked_replans > 25:
                    break
                continue

            blocked_replans = 0

            # Move one step, then re-observe the environment and repeat.
            if len(plan.path) < 2:
                break

            nxt = plan.path[1]
            if nxt in obstacles:
                # Obstacle may have moved into next step after sensing.
                known_obstacles.add(nxt)
                continue

            current = nxt
            total_path.append(current)
            steps_taken += 1

        if current != goal:
            continue

        found_success = True
        ideal = manhattan(start, goal)
        actual = max(0, len(total_path) - 1)
        detour_ratio = (actual / ideal) if ideal > 0 else 1.0

        print("\nDynamic navigation MoE:")
        print("Success: True")
        print(f"Scenario seed: {scenario_seed}")
        print(f"Total traversed steps: {actual}")
        print(f"Ideal steps (no obstacles): {ideal}")
        print(f"Detour ratio: {detour_ratio:.3f}")
        print(f"Replans: {replans}")
        print(f"Total nodes expanded: {total_expanded}")
        print(f"Total planning runtime: {total_runtime_ms:.2f} ms")
        print(f"Turns in executed path: {turns_in_path(total_path)}")

        print("\nExecuted path snapshot:")
        print_grid_snapshot(grid_size, obstacles, total_path, start, goal)
        break

    if not found_success:
        print("\nDynamic navigation MoE:")
        print("Success: False")
        print("No successful mission found within 20 randomized scenarios.")


def main() -> None:
    grid_size = 70
    start, goal = read_user_start_goal(grid_size)
    run_static_obstacle_experiment(start, goal)
    run_dynamic_obstacle_navigation(start, goal)


if __name__ == "__main__":
    main()
