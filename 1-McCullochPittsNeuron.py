"""
CHAPTER 1: McCulloch-Pitts Neuron (1943) - The Foundation of All AI
================================================================

Historical Context:
- Warren McCulloch (neurophysiologist) + Walter Pitts (mathematician)
- Published "A logical calculus of the ideas immanent in nervous activity"
- First mathematical model of how brain neurons might compute
- Revolutionary idea: Binary logic could model intelligence!

The Breakthrough Insight:
- Neurons either fire (1) or don't fire (0) - pure binary
- Multiple inputs combine to produce single output
- If combined input exceeds threshold → neuron fires
- This simple rule could theoretically compute ANYTHING!

Why This Was Revolutionary:
- Proved that simple binary operations could model complex thought
- Showed computation and intelligence were connected
- Laid groundwork for all future neural networks
- First step toward artificial intelligence

The Limitation:
- NO LEARNING! Weights were fixed forever
- Had to hand-design every connection
- But this limitation led to the next breakthrough...
"""

import numpy as np
import matplotlib.pyplot as plt


class McCullochPittsNeuron:
    """
    The very first artificial neuron (1943)

    Key Features:
    - Binary inputs (0 or 1)
    - Fixed weights (no learning!)
    - Binary output (0 or 1)
    - Simple threshold activation
    """

    def __init__(self, weights, threshold):
        """
        Initialize the neuron

        Args:
            weights: Fixed weights for each input (no learning!)
            threshold: If weighted sum >= threshold, neuron fires
        """
        self.weights = np.array(weights)
        self.threshold = threshold

    def activate(self, inputs):
        """
        The core computation: weighted sum vs threshold

        This is the fundamental operation that started it all!
        """
        inputs = np.array(inputs)

        # Calculate weighted sum
        weighted_sum = np.sum(inputs * self.weights)

        # Apply threshold (step function)
        output = 1 if weighted_sum >= self.threshold else 0

        print(f"Inputs: {inputs}")
        print(f"Weights: {self.weights}")
        print(f"Weighted sum: {weighted_sum}")
        print(f"Threshold: {self.threshold}")
        print(f"Output: {output}")
        print("-" * 40)

        return output


def demonstrate_logic_gates():
    """
    The AMAZING discovery: McCulloch-Pitts neurons can implement ANY logic gate!
    This proved that binary neurons could perform logical reasoning!
    """
    print("🧠 McCulloch-Pitts Neurons as Logic Gates")
    print("=" * 50)

    # AND Gate
    print("AND GATE (both inputs must be 1)")
    and_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=2)
    print("0 AND 0 =", and_neuron.activate([0, 0]))
    print("0 AND 1 =", and_neuron.activate([0, 1]))
    print("1 AND 0 =", and_neuron.activate([1, 0]))
    print("1 AND 1 =", and_neuron.activate([1, 1]))

    # OR Gate
    print("\nOR GATE (at least one input must be 1)")
    or_neuron = McCullochPittsNeuron(weights=[1, 1], threshold=1)
    print("0 OR 0 =", or_neuron.activate([0, 0]))
    print("0 OR 1 =", or_neuron.activate([0, 1]))
    print("1 OR 0 =", or_neuron.activate([1, 0]))
    print("1 OR 1 =", or_neuron.activate([1, 1]))

    # NOT Gate (single input with negative weight)
    print("\nNOT GATE (inverts the input)")
    not_neuron = McCullochPittsNeuron(weights=[-1], threshold=0)
    print("NOT 0 =", not_neuron.activate([0]))
    print("NOT 1 =", not_neuron.activate([1]))


def complex_logic_circuit():
    """
    The revolutionary insight: You can combine neurons to make complex circuits!
    This is like building complex molecules from atoms!
    """
    print("\n🔬 Building Complex Logic: (A AND B) OR (C AND D)")
    print("=" * 55)

    # First layer: Two AND gates
    and_gate_1 = McCullochPittsNeuron(weights=[1, 1], threshold=2)  # A AND B
    and_gate_2 = McCullochPittsNeuron(weights=[1, 1], threshold=2)  # C AND D

    # Second layer: OR gate combining the AND results
    or_gate = McCullochPittsNeuron(weights=[1, 1], threshold=1)

    # Test all combinations
    test_cases = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]

    print("A B C D | A∧B | C∧D | (A∧B)∨(C∧D)")
    print("-" * 35)

    for inputs in test_cases:
        A, B, C, D = inputs

        # Layer 1: AND gates
        ab_result = and_gate_1.activate([A, B])
        cd_result = and_gate_2.activate([C, D])

        # Layer 2: OR gate
        final_result = or_gate.activate([ab_result, cd_result])

        print(f"{A} {B} {C} {D} |  {ab_result}  |  {cd_result}  |     {final_result}")


def visualize_neuron():
    """
    Visualize how the McCulloch-Pitts neuron works
    """
    print("\n📊 Visualizing the McCulloch-Pitts Neuron")
    print("=" * 45)

    # Create a simple 2-input neuron for visualization
    neuron = McCullochPittsNeuron(weights=[0.6, 0.8], threshold=1.0)

    # Generate input combinations
    x1_vals = np.linspace(0, 2, 100)
    x2_vals = np.linspace(0, 2, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Calculate weighted sum for each point
    Z = X1 * 0.6 + X2 * 0.8

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot the weighted sum surface
    plt.contour(X1, X2, Z, levels=[1.0], colors="red", linewidths=2)
    plt.contourf(
        X1,
        X2,
        Z >= 1.0,
        levels=[0.5, 1.5],
        colors=["lightblue", "lightcoral"],
        alpha=0.7,
    )

    # Add labels and title
    plt.xlabel("Input 1 (x₁)")
    plt.ylabel("Input 2 (x₂)")
    plt.title("McCulloch-Pitts Neuron Decision Boundary\nRed line: 0.6x₁ + 0.8x₂ = 1.0")
    plt.grid(True, alpha=0.3)

    # Add text annotations
    plt.text(
        1.5,
        0.3,
        "Output = 0\n(Below threshold)",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )
    plt.text(
        0.3,
        1.5,
        "Output = 1\n(Above threshold)",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
    )

    plt.show()


def the_fundamental_limitation():
    """
    The crucial limitation that led to the next breakthrough
    """
    print("\n⚠️  The Fundamental Limitation of McCulloch-Pitts Neurons")
    print("=" * 60)
    print(
        """
    The Problem:
    - Weights are FIXED forever - no learning!
    - You have to hand-design every single connection
    - For complex problems, this becomes impossible
    - How do you know what weights to use?
    
    The Question This Raised:
    "What if neurons could LEARN the right weights automatically?"
    
    This question led to the next breakthrough: THE PERCEPTRON (1957)
    - Frank Rosenblatt's revolutionary learning algorithm
    - First machine that could learn from experience
    - The birth of machine learning!
    """
    )


if __name__ == "__main__":
    print("🚀 Welcome to the Foundation of AI!")
    print("McCulloch-Pitts Neuron (1943) - Where It All Began")
    print("=" * 55)

    # Demonstrate basic logic gates
    demonstrate_logic_gates()

    # Show complex logic circuits
    complex_logic_circuit()

    # Visualize the neuron (uncomment to see plot)
    # visualize_neuron()

    # Explain the limitation that led to the next breakthrough
    the_fundamental_limitation()

    print("\n🎯 KEY INSIGHTS:")
    print("1. Binary neurons can perform ANY logical computation")
    print("2. Complex intelligence might emerge from simple binary operations")
    print("3. BUT... no learning ability - weights are forever fixed!")
    print("4. This limitation led to the PERCEPTRON breakthrough...")
    print("\n🔜 NEXT: The Perceptron (1957) - The First Learning Machine!")
