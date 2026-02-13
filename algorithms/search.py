from algorithms.problems import SearchProblem
import algorithms.utils as utils
from world.game import Directions
from algorithms.heuristics import nullHeuristic


def tinyHouseSearch(problem: SearchProblem):
    """
    Returns a sequence of moves that solves tinyHouse. For any other building, the
    sequence of moves will be incorrect, so only use this for tinyHouse.
    """
    print("Start:", problem.getState)
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))  # Posibles movimientos
    s = Directions.SOUTH
    w = Directions.WEST
    print("Cost of these moves:", problem.getCostOfActions([s, s, w, s, w, w, s, w]))
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    stack = utils.Stack()
    actions = utils.Stack()
    visited = set()
    
    stack.push(problem.getStartState())
    actions.push([])
    
    
    while not stack.isEmpty():
        node = stack.pop()
        action = actions.pop()
        
        if node not in visited:
            visited.add(node)
        
        if problem.isGoalState(node):
            # Llegamos al objetivo
            return action
        
        for element in problem.getSuccessors(node):
            if element[0] not in visited:
                stack.push(element[0])
                actions.push( action + [element[1]])
        
    
    return []


def breadthFirstSearch(problem: SearchProblem):
    """
    Search the shallowest nodes in the search tree first.
    """
    # TODO: Add your code here
    utils.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """
    Search the node of least total cost first.
    """

    # TODO: Add your code here
    utils.raiseNotDefined()


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    # TODO: Add your code here
    utils.raiseNotDefined()


# Abbreviations (you can use them for the -f option in main.py)
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
