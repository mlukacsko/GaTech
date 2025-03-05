from collections import deque

class SemanticNetsAgent:
    def __init__(self):
        # If you want to do any initial processing, add it here.
        pass

    def solve(self, starting_sheep, starting_wolves):
        starting_state = (starting_sheep, starting_wolves, 0)  # (sheep and wolves on left along with boat position)
        goal_state = (0, 0, 1) # no sheep or wolves on left and boat on right

        # BFS
        bfs_queue = deque([(starting_state, [])])
        visited_states = set()

        while bfs_queue:
            current_state, path = bfs_queue.popleft()

            # Are we done?
            if current_state == goal_state:
                return path

            # Mark the current state as visited
            if current_state in visited_states:
                continue
            visited_states.add(current_state)

            # Generate all valid moves
            possible_moves = self.generate_moves(current_state, starting_sheep, starting_wolves)
            for move in possible_moves:
                next_state = self.apply_move(current_state, move)

                # Check if the new state is valid
                if self.is_move_valid(next_state, starting_sheep, starting_wolves):
                    bfs_queue.append((next_state, path + [move]))

        # Return nothing if not solvable 
        return []

    def generate_moves(self, state, total_sheep, total_wolves):
        sheep, wolves, boat = state
        possible_moves = []

        if boat == 0:  # Boat is on left, (Sheep on left, Wolf on right)
            if sheep > 0:
                possible_moves.append((1, 0))
            if wolves > 0:
                possible_moves.append((0, 1))
            if sheep > 1:
                possible_moves.append((2, 0))
            if wolves > 1:
                possible_moves.append((0, 2))
            if sheep > 0 and wolves > 0:
                possible_moves.append((1, 1))
        else:  # Boat is on the right
            if total_sheep - sheep > 0:
                possible_moves.append((1, 0))
            if total_wolves - wolves > 0:
                possible_moves.append((0, 1))
            if total_sheep - sheep > 1:
                possible_moves.append((2, 0))
            if total_wolves - wolves > 1:
                possible_moves.append((0, 2))
            if (total_sheep - sheep > 0 and total_wolves - wolves > 0):
                possible_moves.append((1, 1))
        return possible_moves

    def apply_move(self, state, move):
        sheep, wolves, boat = state
        move_sheep, move_wolves = move

        if boat == 0:  # Moving boat from left to right
            return (sheep - move_sheep, wolves - move_wolves, 1)
        else:  # Moving boat from right to left
            return (sheep + move_sheep, wolves + move_wolves, 0)

    def is_move_valid(self, state, total_sheep, total_wolves):
        sheep, wolves, boat = state

        # Check sheep and wolves counts
        if sheep < 0 or wolves < 0 or sheep > total_sheep or wolves > total_wolves:
            return False

        # Wolves cannot outnumber sheep on either side
        left_sheep = sheep
        left_wolves = wolves
        right_sheep = total_sheep - sheep
        right_wolves = total_wolves - wolves

        if (left_sheep > 0 and left_wolves > left_sheep) or (right_sheep > 0 and right_wolves > right_sheep):
            return False
        return True
