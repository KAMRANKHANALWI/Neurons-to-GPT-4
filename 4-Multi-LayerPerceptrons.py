"""
🚀 CHAPTER 4: Multi-Layer Perceptrons - The Breakthrough!
=========================================================

Historical Context:
- 1980s: After the AI Winter, some brave researchers kept working
- The question: "What if we use MULTIPLE perceptrons working together?"
- The breakthrough: Multi-Layer Perceptrons (MLPs) can solve ANYTHING!
- This led to backpropagation and modern deep learning!

Let's see how TEAMWORK solves what individuals cannot!
"""

import numpy as np
import matplotlib.pyplot as plt


def the_breakthrough_idea():
    """
    Explain the core breakthrough idea
    """
    print("💡 THE BREAKTHROUGH IDEA")
    print("=" * 30)

    print("🤔 After XOR failure, researchers asked:")
    print("   'If ONE perceptron can only draw ONE line...'")
    print("   'What if we use MULTIPLE perceptrons?'")
    print()

    print("🧠 The insight:")
    print(
        "   👨‍💼 Manager (single perceptron): 'I can't solve this complex problem alone'"
    )
    print("   👥 Team (multiple perceptrons): 'Let's work together!'")
    print()

    print("🏗️ The architecture:")
    print("   Layer 1: Multiple perceptrons work on the input")
    print("   Layer 2: Another perceptron combines their results")
    print("   Result: Can solve ANY pattern!")
    print()

    print("🎯 Think of it like:")
    print("   🔍 Detective team solving a complex case:")
    print("      Detective A: 'I'll look for financial clues'")
    print("      Detective B: 'I'll check alibis'")
    print("      Detective C: 'I'll examine physical evidence'")
    print("      Chief: 'I'll combine all your findings to solve the case!'")


def xor_solution_step_by_step():
    """
    Show exactly how multiple perceptrons solve XOR
    """
    print("\n\n🔧 SOLVING XOR STEP BY STEP")
    print("=" * 35)

    print("🎯 Goal: Create XOR using multiple perceptrons")
    print("   XOR truth table:")
    print("   [0,0] → 0, [0,1] → 1, [1,0] → 1, [1,1] → 0")
    print()

    print("💡 The strategy:")
    print("   XOR = 'output 1 if inputs are DIFFERENT'")
    print("   XOR = (A OR B) AND NOT(A AND B)")
    print("   XOR = 'at least one is 1' AND 'not both are 1'")
    print()

    print("🏗️ The architecture:")
    print("   Layer 1:")
    print("      Perceptron 1: Learns OR gate (A OR B)")
    print("      Perceptron 2: Learns AND gate (A AND B)")
    print("   Layer 2:")
    print("      Perceptron 3: Learns 'P1 AND NOT P2'")
    print()

    print("📊 Let's trace through each step:")

    # Test cases
    test_cases = [
        ([0, 0], "both OFF"),
        ([0, 1], "first OFF, second ON"),
        ([1, 0], "first ON, second OFF"),
        ([1, 1], "both ON"),
    ]

    print("\n   Input | OR  | AND | XOR")
    print("   ------|-----|-----|----")

    for inputs, description in test_cases:
        A, B = inputs

        # Layer 1 computations
        or_result = 1 if (A == 1 or B == 1) else 0  # OR gate
        and_result = 1 if (A == 1 and B == 1) else 0  # AND gate

        # Layer 2 computation
        # XOR = OR AND (NOT AND) = OR AND (1 - AND)
        xor_result = 1 if (or_result == 1 and and_result == 0) else 0

        print(f"   {inputs} |  {or_result}  |  {and_result}  |  {xor_result}")

    print()
    print("✅ SUCCESS! Multi-layer network solves XOR!")


def implement_xor_network():
    """
    Implement the actual XOR network with real weights
    """
    print("\n\n🔢 IMPLEMENTING THE XOR NETWORK")
    print("=" * 40)

    print("Let's build the actual network with real weights!")
    print()

    # Define the network weights (these are hand-designed for clarity)
    print("🏗️ Network Architecture:")
    print("   Input Layer: 2 neurons (A, B)")
    print("   Hidden Layer: 2 neurons (OR detector, AND detector)")
    print("   Output Layer: 1 neuron (combines results)")
    print()

    # Layer 1 weights (input to hidden)
    # OR gate: needs A + B >= 0.5, so weights [1, 1], bias -0.5
    # AND gate: needs A + B >= 1.5, so weights [1, 1], bias -1.5
    W1 = np.array([[1, 1], [1, 1]])  # OR gate weights  # AND gate weights
    b1 = np.array([-0.5, -1.5])  # OR bias, AND bias

    # Layer 2 weights (hidden to output)
    # XOR = OR AND (NOT AND) = 1*OR + (-1)*AND, bias 0
    W2 = np.array([1, -1])  # Positive weight for OR, negative for AND
    b2 = np.array([0])

    print("🎯 Layer 1 (Hidden Layer) weights:")
    print(f"   OR gate:  weights={W1[0]}, bias={b1[0]}")
    print(f"   AND gate: weights={W1[1]}, bias={b1[1]}")
    print()
    print("🎯 Layer 2 (Output Layer) weights:")
    print(f"   XOR gate: weights={W2}, bias={b2[0]}")
    print()

    def forward_pass(inputs):
        """Run inputs through the network"""
        # Layer 1: Hidden layer
        z1 = np.dot(inputs, W1.T) + b1  # Linear combination
        a1 = (z1 >= 0).astype(int)  # Step activation function

        # Layer 2: Output layer
        z2 = np.dot(a1, W2) + b2  # Linear combination
        a2 = (z2 >= 0).astype(int)  # Step activation function

        return a1, a2[0]

    print("🧪 TESTING THE NETWORK:")
    print("   Input | Hidden Layer | Output")
    print("         | OR  | AND    |")
    print("   ------|-----|--------|-------")

    # Test all combinations
    test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected_outputs = [0, 1, 1, 0]

    all_correct = True
    for inputs, expected in zip(test_inputs, expected_outputs):
        hidden, output = forward_pass(np.array(inputs))
        or_output, and_output = hidden

        status = "✅" if output == expected else "❌"
        if output != expected:
            all_correct = False

        print(f"   {inputs} |  {or_output}  |  {and_output}   |   {output} {status}")

    print()
    if all_correct:
        print("🎉 PERFECT! Multi-layer network successfully learned XOR!")
    else:
        print("❌ Something went wrong...")

    return W1, b1, W2, b2


def visualize_xor_solution():
    """
    Visualize how multiple lines solve XOR
    """
    print("\n\n📊 VISUALIZING THE XOR SOLUTION")
    print("=" * 40)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("How Multiple Perceptrons Solve XOR!", fontsize=16)

    # Data points
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_outputs = [0, 1, 1, 0]

    # Plot 1: Original XOR problem
    ax1 = axes[0]
    for i, (point, output) in enumerate(zip(points, xor_outputs)):
        color = "green" if output == 1 else "red"
        marker = "o" if output == 1 else "s"
        ax1.scatter(
            point[0], point[1], s=200, c=color, marker=marker, edgecolors="black"
        )
        ax1.annotate(
            f"{point}\n→{output}",
            point,
            xytext=(5, -20),
            textcoords="offset points",
            fontsize=9,
            ha="center",
        )

    ax1.set_title("XOR Problem\n(Impossible for single line)")
    ax1.text(
        0.5,
        0.5,
        "🚫\nNo single line\ncan separate!",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"),
    )

    # Plot 2: OR gate (first hidden neuron)
    ax2 = axes[1]
    or_outputs = [0, 1, 1, 1]
    for i, (point, output) in enumerate(zip(points, or_outputs)):
        color = "green" if output == 1 else "red"
        marker = "o" if output == 1 else "s"
        ax2.scatter(
            point[0], point[1], s=200, c=color, marker=marker, edgecolors="black"
        )
        ax2.annotate(
            f"{point}\n→{output}",
            point,
            xytext=(5, -20),
            textcoords="offset points",
            fontsize=9,
            ha="center",
        )

    # Draw OR decision line: x + y = 0.5
    x_line = np.array([0, 0.5])
    y_line = 0.5 - x_line
    ax2.plot(x_line, y_line, "blue", linewidth=2, linestyle="--")
    ax2.set_title("Hidden Neuron 1\n(OR Gate)")

    # Plot 3: AND gate (second hidden neuron)
    ax3 = axes[2]
    and_outputs = [0, 0, 0, 1]
    for i, (point, output) in enumerate(zip(points, and_outputs)):
        color = "green" if output == 1 else "red"
        marker = "o" if output == 1 else "s"
        ax3.scatter(
            point[0], point[1], s=200, c=color, marker=marker, edgecolors="black"
        )
        ax3.annotate(
            f"{point}\n→{output}",
            point,
            xytext=(5, -20),
            textcoords="offset points",
            fontsize=9,
            ha="center",
        )

    # Draw AND decision line: x + y = 1.5
    x_line = np.array([0.5, 1.5])
    y_line = 1.5 - x_line
    ax3.plot(x_line, y_line, "red", linewidth=2, linestyle="--")
    ax3.set_title("Hidden Neuron 2\n(AND Gate)")

    # Plot 4: Final XOR result
    ax4 = axes[3]
    for i, (point, output) in enumerate(zip(points, xor_outputs)):
        color = "green" if output == 1 else "red"
        marker = "o" if output == 1 else "s"
        ax4.scatter(
            point[0], point[1], s=200, c=color, marker=marker, edgecolors="black"
        )
        ax4.annotate(
            f"{point}\n→{output}",
            point,
            xytext=(5, -20),
            textcoords="offset points",
            fontsize=9,
            ha="center",
        )

    ax4.set_title("Output Neuron\n(XOR = OR AND NOT AND)")
    ax4.text(
        0.5,
        0.5,
        "✅\nXOR SOLVED!\nby teamwork!",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
    )

    # Format all plots
    for ax in axes:
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("Input A")
        ax.set_ylabel("Input B")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

    plt.tight_layout()
    plt.show()

    print("👁️ WHAT YOU'RE SEEING:")
    print("   Panel 1: XOR problem - impossible for single line")
    print("   Panel 2: First hidden neuron learns OR gate")
    print("   Panel 3: Second hidden neuron learns AND gate")
    print("   Panel 4: Output neuron combines them → XOR solved!")


def the_universal_approximation_theorem():
    """
    Explain the universal approximation theorem
    """
    print("\n\n🌟 THE UNIVERSAL APPROXIMATION THEOREM")
    print("=" * 45)

    print("🤯 The AMAZING discovery:")
    print("   Multi-layer networks with enough neurons can approximate ANY function!")
    print()

    print("🎯 What this means:")
    print("   Give me ANY pattern, no matter how complex...")
    print("   I can build a multi-layer network to learn it!")
    print()

    print("🧠 Examples of complex patterns MLPs can learn:")
    print("   🎨 Recognize any image (cat, dog, person)")
    print("   🗣️ Understand speech in any language")
    print("   🎮 Play any video game better than humans")
    print("   📊 Predict stock prices, weather, elections")
    print("   🔬 Discover new drugs, materials, physics")
    print()

    print("💡 The key insight:")
    print("   Single perceptron: Can only draw 1 line")
    print("   Multi-layer network: Can draw ANY shape!")
    print("   More layers + more neurons = More complex shapes!")


def the_learning_challenge():
    """
    Introduce the challenge of training multi-layer networks
    """
    print("\n\n🤔 THE LEARNING CHALLENGE")
    print("=" * 30)

    print("✅ GOOD NEWS: Multi-layer networks can solve anything!")
    print("❓ CHALLENGE: How do we TRAIN them?")
    print()

    print("🧠 The problem:")
    print("   For XOR, we hand-designed the weights")
    print("   But what about complex real-world problems?")
    print("   How do we find the right weights automatically?")
    print()

    print("🎯 Questions that needed answers:")
    print("   1. How do we calculate errors for hidden layers?")
    print("   2. How do we know which hidden neuron is to blame?")
    print("   3. How do we use Rosenblatt's formula for multiple layers?")
    print()

    print("💡 This challenge led to the next breakthrough:")
    print("   BACKPROPAGATION (1986)")
    print("   'The algorithm that taught networks to teach themselves!'")


def key_insights():
    """
    Summarize key insights about multi-layer perceptrons
    """
    print("\n\n🎯 KEY INSIGHTS ABOUT MULTI-LAYER PERCEPTRONS")
    print("=" * 55)

    print("1️⃣ TEAMWORK BEATS INDIVIDUAL EFFORT:")
    print("   Multiple simple perceptrons > One complex perceptron")
    print()

    print("2️⃣ LAYERS CREATE HIERARCHY:")
    print("   Layer 1: Detects simple patterns")
    print("   Layer 2: Combines simple patterns into complex ones")
    print("   Layer 3+: Even more complex combinations!")
    print()

    print("3️⃣ UNIVERSAL APPROXIMATION:")
    print("   With enough layers/neurons, can learn ANY pattern!")
    print()

    print("4️⃣ XOR WAS THE CATALYST:")
    print("   'Failure' that led to the greatest breakthrough!")
    print()

    print("5️⃣ THE LEARNING PROBLEM:")
    print("   Can solve anything... if we can train it!")
    print("   This led to backpropagation algorithm")
    print()

    print("6️⃣ MODERN DEEP LEARNING:")
    print("   Today's AI (ChatGPT, etc.) are just VERY deep MLPs!")
    print("   Same principles, much bigger scale!")


if __name__ == "__main__":
    print("🚀 MULTI-LAYER PERCEPTRONS: The Breakthrough That Saved AI!")
    print("=" * 65)
    print("From AI Winter to AI Spring - the power of teamwork!")
    print()

    # The breakthrough idea
    the_breakthrough_idea()

    # Show XOR solution step by step
    xor_solution_step_by_step()

    # Implement the actual network
    implement_xor_network()

    # Visualize the solution
    visualize_xor_solution()

    # Universal approximation theorem
    the_universal_approximation_theorem()

    # The learning challenge
    the_learning_challenge()

    # Key insights
    key_insights()

    print("\n🔜 NEXT: Backpropagation - Teaching Networks to Teach Themselves!")
    print("The algorithm that made modern AI possible!")
    print("Get ready for the most elegant algorithm in computer science! ✨")
