import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def cosine_similarity(a, b):
    """Calculate cosine similarity between two skill vectors."""
    dot_product = sum([x * y for x, y in zip(a, b)])
    mag_a = (sum([x ** 2 for x in a])) ** 0.5
    mag_b = (sum([y ** 2 for y in b])) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (mag_a * mag_b)


def plot_similarity_matrix(agents, humans):
    # Generate similarity matrix for visualization
    S = [[cosine_similarity(agent, human) for human in humans] for agent in agents]
    plt.imshow(S, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(range(len(humans)), [f"H{i+1}" for i in range(len(humans))])
    plt.yticks(range(len(agents)), [f"A{i+1}" for i in range(len(agents))])
    plt.title("AI-Human Similarity Matrix")
    plt.savefig("visualizations/similarity_matrix.png")
    plt.close()


def hungarian_algorithm(cost_matrix):
    n = len(cost_matrix)  # we assume square matrix
    matrix = np.array(cost_matrix)
    
    # Subtract row minima
    for i in range(n):
        matrix[i] -= np.min(matrix[i])
    
    # Subtract column minima
    for j in range(n):
        matrix[:, j] -= np.min(matrix[:, j])
    
    # Cover all zeros with minimum number of lines
    def cover_zeros(matrix):
        n = len(matrix)
        covered_rows = set()
        covered_cols = set()
        marked_zeros = []
        
        # Mark zeros greedily
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 0 and i not in covered_rows and j not in covered_cols:
                    marked_zeros.append((i, j))
                    covered_rows.add(i)
                    covered_cols.add(j)
                    break  # needed google for this. move on to next row after finishing one row
        return marked_zeros, covered_rows, covered_cols
    
    # Initial covering
    marked_zeros, covered_rows, covered_cols = cover_zeros(matrix)
    
    while len(covered_rows) + len(covered_cols) < n:
        # I didn't know how to access the uncovered_vals so I needed chatgpt for this
        uncovered_vals = matrix[~np.isin(range(n), list(covered_rows)), :]
        uncovered_vals = uncovered_vals[:, ~np.isin(range(n), list(covered_cols))]
        min_uncovered = np.min(uncovered_vals)
        
        # Subtract and add as required
        for i in range(n):
            if i not in covered_rows:
                matrix[i] -= min_uncovered
            if i in covered_cols:
                # in intersection of covered row and column
                matrix[:, i] += min_uncovered
        
        # Updates variables from updated matrix
        marked_zeros, covered_rows, covered_cols = cover_zeros(matrix)
    
    return marked_zeros


def find_optimal_pairs(agents, humans):
    # Generate cost matrix and find optimal pairs
    S = [[cosine_similarity(agent, human) for human in humans] for agent in agents]
    C = [[1.0 - s for s in row] for row in S]
    marked_zeros = hungarian_algorithm(C)
    total_similarity = 0
    print("\nSummary Table:")
    print("AI Agent | Human Expert | Cosine Similarity")
    print("---------|--------------|------------------")
    for i, j in marked_zeros:
        print(f"A{i+1}      | H{j+1}         | {S[i][j]:.3f}")
        total_similarity += S[i][j]
    print(f"Total Similarity: {total_similarity:.3f}")


def main():
    # Load data and run the algorithm
    data = pd.read_csv("data/sample_dataset.csv")
    agents = data[data["Type"] == "AI"][["Technical", "Creative", "Communication"]].values.tolist()
    humans = data[data["Type"] == "Human"][["Technical", "Creative", "Communication"]].values.tolist()
    plot_similarity_matrix(agents, humans)
    print("\nOptimal Pairs:")
    find_optimal_pairs(agents, humans)


if __name__ == "__main__":
    main()
