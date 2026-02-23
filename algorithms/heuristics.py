import math
from typing import Any, Tuple
from algorithms import utils
from algorithms.problems import MultiSurvivorProblem, SearchProblem



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def manhattanHeuristic(state, problem):
    """
    The Manhattan distance heuristic.
    """
    #L= |x2-x1| + |y2-y1|
    goal = problem.goal
    x1, y1 = state
    x2, y2 = goal
    return abs(x2 - x1) + abs(y2 - y1)
    


def euclideanHeuristic(state, problem):
    """
    The Euclidean distance heuristic.
    """
    # L= sqrt((x2-x1)^2 + (y2-y1)^2)
    goal = problem.goal
    x1, y1 = state
    x2, y2 = goal
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def survivorHeuristic(state: Tuple[Tuple, Any], problem: MultiSurvivorProblem):
    """
    # Versión inicial del código, haciendo uso de mst con KRUSKALL y distancia Manhattan entre sobrevivientes y posición actual. 
    # Se incluyen otras versiones más abajo, con diferentes combinaciones de técnicas apoyandose de la IAG.

    Your heuristic for the MultiSurvivorProblem.

    state: (position, survivors_grid)
    problem: MultiSurvivorProblem instance

    This must be admissible and preferably consistent.

    Hints:
    - Use problem.heuristicInfo to cache expensive computations
    - Go with some simple heuristics first, then build up to more complex ones
    - Consider: distance to nearest survivor + MST of remaining survivors
    - Balance heuristic strength vs. computation time (do experiments!)
    """
    """
    position, survivors_grid = state
    survivor_positions = survivors_grid.asList()

    if not survivor_positions:
        return 0

    # Distancia Manhattan simple entre dos puntos
    def mdist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Distancia al sobreviviente más cercano
    nearest = min(mdist(position, s) for s in survivor_positions)

    if len(survivor_positions) == 1:
        return nearest

    # MST con Kruskal
    cache_key = str(survivor_positions)

    if cache_key not in problem.heuristicInfo:
        edges = []
        for i in range(len(survivor_positions)):
            for j in range(i + 1, len(survivor_positions)):
                d = mdist(survivor_positions[i], survivor_positions[j])
                edges.append((d, i, j))

        edges.sort()

        parent = list(range(len(survivor_positions)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[px] = py
            return True

        mst_cost = 0
        edges_used = 0
        for cost, i, j in edges:
            if union(i, j):
                mst_cost += cost
                edges_used += 1
                if edges_used == len(survivor_positions) - 1:
                    break

        problem.heuristicInfo[cache_key] = mst_cost

    return nearest + problem.heuristicInfo[cache_key]
    utils.raiseNotDefined()
    ------------------------------------------------------------------------------------------------
    """
    """
    #KRUSKALL con BFS
    #PROMPTS: 1. Qué le podríamos hacer para que mejorara la ejecución y no se demore tanto?
    # 2. Pues implementemos bfs, pero como lo hacemos?
    #en el archivo de search.py que es donde tambien está el de A* está de bfs, pero se diseñó como algoritmo para searchProblem:

    position, survivors_grid = state
    survivor_positions = survivors_grid.asList()

    if not survivor_positions:
        return 0

    # Precalcular distancias reales entre todos los sobrevivientes una sola vez
    if "dist_matrix" not in problem.heuristicInfo:
        all_survivors = problem.start[1].asList()  # todos los sobrevivientes originales
        dist_matrix = {}
        for i, a in enumerate(all_survivors):
            for j, b in enumerate(all_survivors):
                if i != j:
                    dist_matrix[(a, b)] = _bfsDistance(a, b, problem.walls)
        problem.heuristicInfo["dist_matrix"] = dist_matrix
        problem.heuristicInfo["all_survivors"] = all_survivors

    dist_matrix = problem.heuristicInfo["dist_matrix"]

    # 1. Distancia real al sobreviviente más cercano
    nearest = min(_bfsDistance(position, s, problem.walls) for s in survivor_positions)

    if len(survivor_positions) == 1:
        return nearest

    # 2. MST con Kruskal usando distancias reales
    cache_key = str(survivor_positions)
    if cache_key not in problem.heuristicInfo:
        edges = []
        for i in range(len(survivor_positions)):
            for j in range(i+1, len(survivor_positions)):
                a, b = survivor_positions[i], survivor_positions[j]
                d = dist_matrix.get((a, b)) or _bfsDistance(a, b, problem.walls)
                edges.append((d, i, j))

        edges.sort()

        parent = list(range(len(survivor_positions)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[px] = py
            return True

        mst_cost = 0
        edges_used = 0
        for cost, i, j in edges:
            if union(i, j):
                mst_cost += cost
                edges_used += 1
                if edges_used == len(survivor_positions) - 1:
                    break

        problem.heuristicInfo[cache_key] = mst_cost

    return nearest + problem.heuristicInfo[cache_key]
    -------------------------------------------------------------------------------------------------

    #solo PRIM con Manhattan, distancias reales entre sobrevivientes
    #PROMPTS: 1. Okay, dame entonces la versión más óptima pero ahora con prim que al final es otra opción y no lo probamos
    
    position, survivors_grid = state
    survivor_positions = survivors_grid.asList()

    if not survivor_positions:
        return 0

    def mdist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # 1. Distancia al sobreviviente más cercano
    nearest = min(mdist(position, s) for s in survivor_positions)

    if len(survivor_positions) == 1:
        return nearest

    # 2. MST con Prim
    cache_key = str(survivor_positions)
    if cache_key not in problem.heuristicInfo:
        n = len(survivor_positions)
        in_mst = [False] * n
        min_edge = [float('inf')] * n
        min_edge[0] = 0
        mst_cost = 0

        for _ in range(n):
            # Nodo no visitado con menor arista
            u = min((i for i in range(n) if not in_mst[i]), key=lambda i: min_edge[i])
            in_mst[u] = True
            mst_cost += min_edge[u]

            # Actualizar aristas mínimas de vecinos
            for v in range(n):
                if not in_mst[v]:
                    d = mdist(survivor_positions[u], survivor_positions[v])
                    if d < min_edge[v]:
                        min_edge[v] = d

        problem.heuristicInfo[cache_key] = mst_cost

    return nearest + problem.heuristicInfo[cache_key]
    ---------------------------------------------------------------------------------
    """
    """
    #PRIM con BFS precalculado
    #PROMPTS: 1. Pero analicemos, si lo único que en verdad está cambiando que afecta es el tamaño del layout, 
    # cómo podríamos mejorar a una versión absolutamente mejor?

    position, survivors_grid = state
    survivor_positions = survivors_grid.asList()

    if not survivor_positions:
        return 0

    def mdist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Precalcular UNA sola vez distancias reales entre todos los sobrevivientes
    if "dist_matrix" not in problem.heuristicInfo:
        all_survivors = problem.start[1].asList()
        dist_matrix = {}
        for s in all_survivors:
            # BFS desde cada sobreviviente hacia todos los demás
            dist_from_s = {s: 0}
            queue = utils.Queue()
            queue.push((s, 0))
            while not queue.isEmpty():
                pos, dist = queue.pop()
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    neighbor = (pos[0]+dx, pos[1]+dy)
                    if not problem.walls[neighbor[0]][neighbor[1]] and neighbor not in dist_from_s:
                        dist_from_s[neighbor] = dist + 1
                        queue.push((neighbor, dist+1))
            # Guardar solo distancias hacia otros sobrevivientes
            for t in all_survivors:
                if s != t:
                    dist_matrix[(s, t)] = dist_from_s.get(t, float('inf'))

        problem.heuristicInfo["dist_matrix"] = dist_matrix

    dist_matrix = problem.heuristicInfo["dist_matrix"]

    # 1. Nearest con Manhattan (robot cambia constantemente, BFS sería muy caro)
    nearest = min(mdist(position, s) for s in survivor_positions)

    if len(survivor_positions) == 1:
        return nearest

    # 2. MST con Prim usando distancias reales entre sobrevivientes
    cache_key = str(survivor_positions)
    if cache_key not in problem.heuristicInfo:
        n = len(survivor_positions)
        in_mst = [False] * n
        min_edge = [float('inf')] * n
        min_edge[0] = 0
        mst_cost = 0

        for _ in range(n):
            u = min((i for i in range(n) if not in_mst[i]), key=lambda i: min_edge[i])
            in_mst[u] = True
            mst_cost += min_edge[u]

            for v in range(n):
                if not in_mst[v]:
                    a, b = survivor_positions[u], survivor_positions[v]
                    d = dist_matrix.get((a, b), mdist(a, b))
                    if d < min_edge[v]:
                        min_edge[v] = d

        problem.heuristicInfo[cache_key] = mst_cost

    return nearest + problem.heuristicInfo[cache_key]
    
    """

    #BFS precalculado con KRUSKAL
    #PROMPTS: 1. Hagamos una última versión, con este bfs precalculado pero con Kruskal
    position, survivors_grid = state
    survivor_positions = survivors_grid.asList()

    if not survivor_positions:
        return 0

    def mdist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Precalcular UNA sola vez distancias reales entre todos los sobrevivientes
    if "dist_matrix" not in problem.heuristicInfo:
        all_survivors = problem.start[1].asList()
        dist_matrix = {}
        for s in all_survivors:
            dist_from_s = {s: 0}
            queue = utils.Queue()
            queue.push((s, 0))
            while not queue.isEmpty():
                pos, dist = queue.pop()
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    neighbor = (pos[0]+dx, pos[1]+dy)
                    if not problem.walls[neighbor[0]][neighbor[1]] and neighbor not in dist_from_s:
                        dist_from_s[neighbor] = dist + 1
                        queue.push((neighbor, dist+1))
            for t in all_survivors:
                if s != t:
                    dist_matrix[(s, t)] = dist_from_s.get(t, float('inf'))

        problem.heuristicInfo["dist_matrix"] = dist_matrix

    dist_matrix = problem.heuristicInfo["dist_matrix"]

    # 1. Nearest con Manhattan
    nearest = min(mdist(position, s) for s in survivor_positions)

    if len(survivor_positions) == 1:
        return nearest

    # 2. MST con Kruskal usando distancias reales
    cache_key = str(survivor_positions)
    if cache_key not in problem.heuristicInfo:
        edges = []
        for i in range(len(survivor_positions)):
            for j in range(i+1, len(survivor_positions)):
                a, b = survivor_positions[i], survivor_positions[j]
                d = dist_matrix.get((a, b), mdist(a, b))
                edges.append((d, i, j))

        edges.sort()

        parent = list(range(len(survivor_positions)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[px] = py
            return True

        mst_cost = 0
        edges_used = 0
        for cost, i, j in edges:
            if union(i, j):
                mst_cost += cost
                edges_used += 1
                if edges_used == len(survivor_positions) - 1:
                    break

        problem.heuristicInfo[cache_key] = mst_cost

    return nearest + problem.heuristicInfo[cache_key]
    

#Función auxiliar para calcular distancia en el algoritmo de BFS + Kruskal
def _bfsDistance(start, goal, walls):
    """Distancia real entre dos puntos evitando paredes."""
    if start == goal:
        return 0
    
    from algorithms import utils
    queue = utils.Queue()
    queue.push((start, 0))
    visited = {start}
    
    while not queue.isEmpty():
        pos, dist = queue.pop()
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = pos[0]+dx, pos[1]+dy
            neighbor = (nx, ny)
            if neighbor == goal:
                return dist + 1
            if not walls[nx][ny] and neighbor not in visited:
                visited.add(neighbor)
                queue.push((neighbor, dist+1))
    
    return float('inf')
