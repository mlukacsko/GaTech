import matplotlib.pyplot as plt
import time
from SemanticNetsAgent import SemanticNetsAgent


def test_and_collect_data():
    test_agent = SemanticNetsAgent()

    # Specified range of sheep and wolves values
    sheep_values = [1, 10, 25, 50, 100, 250, 500]
    wolves_values = [1, 10, 25, 50, 100, 250, 500]

    time_results = []
    move_results = []
    sheep_wolves_pairs = []
    solvable_cases = []  # To distinguish between solvable and unsolvable cases

    # Loop through each combination of sheep and wolves values
    for sheep in sheep_values:
        for wolves in wolves_values:
            start_time = time.time()  # Start timing
            result = test_agent.solve(sheep, wolves)
            end_time = time.time()  # End timing

            elapsed_time = end_time - start_time
            num_moves = len(result)

            # Store results
            time_results.append(elapsed_time)
            move_results.append(num_moves)
            sheep_wolves_pairs.append((sheep, wolves))

            # Mark whether the case is solvable
            solvable_cases.append(num_moves > 0)

            print(f"Test case: {sheep} sheep, {wolves} wolves → Number of moves: {num_moves}")
            print(f"Time taken: {elapsed_time:.9f} seconds\n")

    return sheep_wolves_pairs, time_results, move_results, solvable_cases


def plot_solvable_results_with_moves(sheep_wolves_pairs, time_results, move_results, solvable_cases):
    # Combine sheep and wolves as individual labels for the x-axis
    combined_labels = [f"{sheep}S-{wolves}W" for sheep, wolves in sheep_wolves_pairs]

    # Separate solvable data
    solvable_time_results = [time_results[i] for i in range(len(solvable_cases)) if solvable_cases[i]]
    solvable_move_results = [move_results[i] for i in range(len(solvable_cases)) if solvable_cases[i]]
    solvable_labels = [combined_labels[i] for i in range(len(solvable_cases)) if solvable_cases[i]]

    # Plot time taken for solvable cases
    plt.figure(figsize=(14, 6))
    plt.plot(solvable_labels, solvable_time_results, marker='o', color='blue', label='Time taken (seconds)', linestyle='-')
    plt.xlabel('Sheep and Wolves (S:W)')
    plt.ylabel('Time Taken (seconds)')
    plt.title('Time Taken vs Sheep and Wolves Pairs (Only Solvable)')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.show()

    # Plot number of moves for solvable cases
    plt.figure(figsize=(14, 6))
    plt.plot(solvable_labels, solvable_move_results, marker='x', color='green', label='Number of moves', linestyle='-')
    plt.xlabel('Sheep and Wolves (S:W)')
    plt.ylabel('Number of Moves')
    plt.title('Number of Moves vs Sheep and Wolves Pairs (Only Solvable)')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.show()


if __name__ == "__main__":
    # Collect data
    sheep_wolves_pairs, time_results, move_results, solvable_cases = test_and_collect_data()

    # Plot only the solvable cases with time and moves
    plot_solvable_results_with_moves(sheep_wolves_pairs, time_results, move_results, solvable_cases)
