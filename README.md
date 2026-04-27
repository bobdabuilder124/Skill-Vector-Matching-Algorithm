# Skill Vector Matching Algorithm

This project optimizes AI-human collaboration by representing AI agent capabilities and human expert skills as multi-dimensional vectors. It utilizes a complementarity scoring function to identify optimal AI-human pairings based on skill compatibility and generates an interactive visualization showing potential matches alongside their complementarity scores.

## 🚀 Key Features

* **Vector Representation of Skills:** Evaluates both AI systems and human experts across specific dimensions (e.g., technical, creative, and communication skills) using $n$-dimensional vectors standardized to a [0, 1] scale.
* **Complementarity Scoring:** Calculates the Cosine Similarity between vectors to measure directional alignment, ensuring the algorithm accurately captures strengths and potential gaps in the pairing.
* **Optimal Assignment:** Applies the Hungarian Algorithm (Kuhn-Munkres Algorithm) to find the one-to-one assignment that maximizes total skill compatibility.
* **Data Visualization:** Outputs visual similarity matrices (heatmaps) to easily interpret cosine similarities and review final match evaluations.

## 🧮 How It Works

1.  **Calculate Similarity:** The algorithm computes the dot product of the AI and Human vectors and divides it by the product of their magnitudes to find the Cosine Similarity:
    $$C(A_{i},H_{j})=\frac{A_{i}\cdot H_{j}}{||A_{i}||||H_{j}||}$$
2.  **Generate Cost Matrix:** Because the Hungarian Algorithm minimizes cost by default, the similarity matrix $S$ is converted into a cost matrix $C$ using the formula:
    $$C_{ij}=S_{max}-S_{ij}$$
3.  **Assign Pairs:** The algorithm processes the cost matrix to extract the optimal AI-human pairs and calculates the maximum total similarity score for the group.
