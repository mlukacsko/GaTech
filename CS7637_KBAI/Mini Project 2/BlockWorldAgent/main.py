import matplotlib.pyplot as plt
import time
from BlockWorldAgent import BlockWorldAgent

def test():
    test_agent = BlockWorldAgent()

    # Initial arrangements and goal arrangements
    initial_arrangement_1 = [["A", "B", "C"], ["D", "E"]]
    goal_arrangement_1 = [["A", "C"], ["D", "E", "B"]]
    goal_arrangement_2 = [["A", "B", "C", "D", "E"]]
    goal_arrangement_3 = [["D", "E", "A", "B", "C"]]
    goal_arrangement_4 = [["C", "D"], ["E", "A", "B"]]

    initial_arrangement_2 = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]]
    goal_arrangement_5 = [["A", "B", "C", "D", "E", "F", "G", "H", "I"]]
    goal_arrangement_6 = [["I", "H", "G", "F", "E", "D", "C", "B", "A"]]
    goal_arrangement_7 = [["H", "E", "F", "A", "C"], ["B", "D"], ["G", "I"]]
    goal_arrangement_8 = [["F", "D", "C", "I", "G", "A"], ["B", "E", "H"]]

    initial_arrangement_3 = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"], ["J", "K", "L"]]
    goal_arrangement_9 = [["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]]
    goal_arrangement_10 = [["L", "K", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A"]]
    goal_arrangement_11 = [["H", "E", "F", "A", "C"], ["B", "D"], ["G", "I"], "L", "J", "K"]
    goal_arrangement_12 = [["J", "K", "L", "I", "G", "A"], ["B", "E", "H"], ["F", "D", "C"]]

    initial_arrangement_4 = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"], ["J", "K", "L"], ["M", "N", "O"]]
    goal_arrangement_13 = [["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]]
    goal_arrangement_14 = [["O", "N", "M", "L", "K", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A"]]
    goal_arrangement_15 = [["H", "E", "F", "A", "C"],["M", "N", "O"], ["B", "D"], ["G", "I"], "L", "J", "K"]
    goal_arrangement_16 = [["J", "K", "L", ], ["B", "M", "N", "O", "E", "H"], ["F", "D", "C"], ["I", "G", "A"]]

    cases = [
        (initial_arrangement_1, [goal_arrangement_1, goal_arrangement_2, goal_arrangement_3, goal_arrangement_4]),
        (initial_arrangement_2, [goal_arrangement_5, goal_arrangement_6, goal_arrangement_7, goal_arrangement_8]),
        (initial_arrangement_3, [goal_arrangement_9, goal_arrangement_10, goal_arrangement_11, goal_arrangement_12]),
        (initial_arrangement_4, [goal_arrangement_13, goal_arrangement_14, goal_arrangement_15, goal_arrangement_16]),
    ]

    block_counts = []
    time_taken = []
    move_counts = []

    for initial, goals in cases:
        for goal in goals:
            start_time = time.time()
            moves = test_agent.solve(initial, goal)
            end_time = time.time()

            total_blocks = sum(len(stack) for stack in initial)
            block_counts.append(total_blocks)
            time_taken.append((end_time - start_time) * 1000)  # Execution time in ms
            move_counts.append(len(moves))

    plt.figure(figsize=(8, 6))
    plt.plot(block_counts, time_taken, 'bo-', label='Execution Time')
    plt.xlabel('Number of Blocks')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time vs. Number of Blocks')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(block_counts, move_counts, 'go-', label='Number of Moves')
    plt.xlabel('Number of Blocks')
    plt.ylabel('Number of Moves')
    plt.title('Number of Moves vs. Number of Blocks')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test()
