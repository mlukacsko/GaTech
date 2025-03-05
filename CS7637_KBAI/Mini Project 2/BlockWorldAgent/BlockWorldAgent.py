import time
import copy

class BlockWorldAgent:
    def __init__(self):
        pass

    def solve(self, initial_arrangement, goal_arrangement):
        # Start measuring the time
        start_time = time.time()

        # Get the total number of blocks in the initial arrangement
        total_blocks = self.count_total_blocks(initial_arrangement)

        # Generate a solution by planning the moves to reach the goal state
        solution = self.plan_solution(initial_arrangement, goal_arrangement, total_blocks)

        # Measure execution time
        end_time = time.time()
        execution_time = str((end_time - start_time) * 1000)
        print(f"Execution time: {execution_time}ms")

        return solution

    def plan_solution(self, current_state, goal_state, total_blocks):
        """Plan the moves necessary to transform current_state into goal_state."""
        moves_list = []
        while self.calculate_difference(current_state, goal_state, total_blocks) != 0:
            current_state, best_move = self.find_best_move(current_state, goal_state, total_blocks)
            moves_list.append(best_move)
        return moves_list

    def find_best_move(self, current_state, goal_state, total_blocks):
        """Find and return the best move to reduce the difference between current and goal states."""
        optimal_move = None
        optimal_state = None
        optimal_difference = self.calculate_difference(current_state, goal_state, total_blocks)

        # Try moving the top block between stacks
        for from_stack_index, from_stack in enumerate(current_state):
            if len(from_stack) == 0:
                continue
            top_block = from_stack[-1]

            # Try placing it on other stacks
            for to_stack_index, to_stack in enumerate(current_state):
                if from_stack_index != to_stack_index:  # Avoid placing on the same stack
                    new_state, move = self.apply_move(current_state, from_stack_index, to_stack_index)
                    new_difference = self.calculate_difference(new_state, goal_state, total_blocks)
                    if new_difference < optimal_difference:
                        optimal_difference = new_difference
                        optimal_state = new_state
                        optimal_move = move

            # Try moving the block to the table
            if len(from_stack) > 1:  # Don't move it if it's already alone
                new_state, move = self.apply_move(current_state, from_stack_index, -1)
                new_difference = self.calculate_difference(new_state, goal_state, total_blocks)
                if new_difference <= optimal_difference:
                    optimal_difference = new_difference
                    optimal_state = new_state
                    optimal_move = move

        return optimal_state, optimal_move

    def apply_move(self, state, from_stack_index, to_stack_index):
        """Move a block from one stack to another or to the table."""
        updated_state = [stack[:] for stack in state]
        block_to_move = updated_state[from_stack_index].pop()

        if to_stack_index == -1:  # Move to the table
            updated_state.append([block_to_move])
            move = (block_to_move, 'Table')
        else:  # Move to another stack
            target_stack = updated_state[to_stack_index]
            if len(target_stack) == 0:
                move = (block_to_move, 'Table')
            else:
                move = (block_to_move, target_stack[-1])
            target_stack.append(block_to_move)

        if len(updated_state[from_stack_index]) == 0:
            updated_state.remove(updated_state[from_stack_index])  # Remove empty stacks

        return updated_state, move

    def calculate_difference(self, current_state, goal_state, total_blocks):
        """Calculate the number of blocks not yet in the correct position."""
        correct_positions = 0
        for goal_stack in goal_state:
            for current_stack in current_state:
                index = 0
                while index < len(goal_stack) and index < len(current_stack) and goal_stack[index] == current_stack[index]:
                    correct_positions += 1
                    index += 1
                if index > 0:  # If there are common blocks, stop comparing this stack
                    break
        return total_blocks - correct_positions

    def count_total_blocks(self, arrangement):
        """Count the total number of blocks in the arrangement."""
        return sum(len(stack) for stack in arrangement)
