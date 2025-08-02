"""
👁️ VISUAL PROOF: Why XOR is Impossible for Single Perceptrons!
==============================================================

Let's SEE with our own eyes why single perceptrons can solve AND/OR
but completely FAIL at XOR!

This will show you the "decision boundaries" - the lines that separate
the ✅ from the ❌
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_logic_gates():
    """
    Create visual plots showing why AND/OR work but XOR doesn't
    """

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Why Single Perceptrons Can Solve AND/OR but NOT XOR!",
        fontsize=16,
        fontweight="bold",
    )

    # Define the 4 possible input combinations
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Define outputs for each gate
    and_outputs = np.array([0, 0, 0, 1])  # Only [1,1] → 1
    or_outputs = np.array([0, 1, 1, 1])  # Only [0,0] → 0
    xor_outputs = np.array([0, 1, 1, 0])  # Different inputs → 1

    gates = [
        ("AND Gate", and_outputs, "red"),
        ("OR Gate", or_outputs, "blue"),
        ("XOR Gate", xor_outputs, "purple"),
    ]

    for i, (title, outputs, color) in enumerate(gates):
        ax = axes[i]

        # Plot the points
        for j, (input_point, output) in enumerate(zip(inputs, outputs)):
            if output == 1:
                ax.scatter(
                    input_point[0],
                    input_point[1],
                    s=200,
                    c="green",
                    marker="o",
                    label="✅ Output = 1" if j == 0 and output == 1 else "",
                )
            else:
                ax.scatter(
                    input_point[0],
                    input_point[1],
                    s=200,
                    c="red",
                    marker="x",
                    label="❌ Output = 0" if j == 0 and output == 0 else "",
                )

            # Add text labels
            ax.annotate(
                f"{input_point}→{output}",
                (input_point[0], input_point[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        # Try to draw decision boundary (line)
        if title == "AND Gate":
            # Line: x + y = 1.5 (anything above this line is 1)
            x_line = np.array([0.5, 1.5])
            y_line = 1.5 - x_line
            ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=3,
                label="Decision Line",
                linestyle="--",
            )
            ax.text(
                0.6,
                0.7,
                "Line separates ✅ from ❌!",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
            )

        elif title == "OR Gate":
            # Line: x + y = 0.5 (anything above this line is 1)
            x_line = np.array([0, 0.5])
            y_line = 0.5 - x_line
            ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=3,
                label="Decision Line",
                linestyle="--",
            )
            ax.text(
                0.1,
                0.7,
                "Line separates ✅ from ❌!",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
            )

        elif title == "XOR Gate":
            # Try to draw a line - but it's impossible!
            ax.text(
                0.5,
                0.5,
                "🚫 NO SINGLE LINE\nCAN SEPARATE THEM!",
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"),
            )

        # Format the plot
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("Input 1", fontsize=12)
        ax.set_ylabel("Input 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

    plt.tight_layout()
    plt.show()

    print("👁️ WHAT YOU'RE SEEING:")
    print("=" * 25)
    print("🔴 AND Gate: One line easily separates green ✅ from red ❌")
    print("🔵 OR Gate: One line easily separates green ✅ from red ❌")
    print("🔶 XOR Gate: IMPOSSIBLE! ✅ and ❌ are mixed up - no single line works!")


def plot_decision_boundaries_detailed():
    """
    Show more detailed decision boundaries with equations
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Decision Boundaries: The Math Behind the Magic!",
        fontsize=16,
        fontweight="bold",
    )

    # Create a grid of points to visualize the decision boundary
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)

    # Define the 4 input points
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    gates_detailed = [
        ("AND Gate", [0, 0, 0, 1], [1, 1, -1.5], "Reds"),
        ("OR Gate", [0, 1, 1, 1], [1, 1, -0.5], "Blues"),
        ("NOT Gate", [1, 0], [1, 0, -0.5], "Greens"),
        ("XOR Gate - IMPOSSIBLE!", [0, 1, 1, 0], None, "Purples"),
    ]

    for idx, (title, truth_table, weights, colormap) in enumerate(gates_detailed):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        if weights is not None and title != "XOR Gate - IMPOSSIBLE!":
            # Calculate decision boundary: w1*x + w2*y + bias = 0
            w1, w2, bias = weights
            Z = w1 * X + w2 * Y + bias

            # Plot the decision boundary (where Z = 0)
            ax.contour(X, Y, Z, levels=[0], colors="black", linewidths=3)
            ax.contourf(
                X, Y, Z >= 0, levels=[0.5, 1.5], colors=["lightcoral"], alpha=0.3
            )
            ax.contourf(
                X, Y, Z < 0, levels=[-1.5, -0.5], colors=["lightblue"], alpha=0.3
            )

            # Add equation text
            equation = f"{w1}x₁ + {w2}x₂ + {bias} = 0"
            ax.text(
                0.02,
                0.98,
                f"Decision line: {equation}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white"),
            )

        # Plot the actual data points
        if len(truth_table) == 4:  # For 2-input gates
            for i, (point, output) in enumerate(zip(points, truth_table)):
                if output == 1:
                    ax.scatter(
                        point[0],
                        point[1],
                        s=300,
                        c="green",
                        marker="o",
                        edgecolors="black",
                        linewidth=2,
                    )
                    ax.annotate(
                        "✅",
                        point,
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                    )
                else:
                    ax.scatter(
                        point[0],
                        point[1],
                        s=300,
                        c="red",
                        marker="s",
                        edgecolors="black",
                        linewidth=2,
                    )
                    ax.annotate(
                        "❌",
                        point,
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                    )

                # Add coordinate labels
                ax.text(
                    point[0],
                    point[1] - 0.15,
                    f"{point}",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

        else:  # For 1-input NOT gate
            single_points = np.array([[0], [1]])
            for i, (point, output) in enumerate(zip(single_points, truth_table)):
                x_pos = point[0]
                y_pos = 0.5  # Center vertically
                if output == 1:
                    ax.scatter(
                        x_pos,
                        y_pos,
                        s=300,
                        c="green",
                        marker="o",
                        edgecolors="black",
                        linewidth=2,
                    )
                    ax.annotate(
                        "✅",
                        (x_pos, y_pos),
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                    )
                else:
                    ax.scatter(
                        x_pos,
                        y_pos,
                        s=300,
                        c="red",
                        marker="s",
                        edgecolors="black",
                        linewidth=2,
                    )
                    ax.annotate(
                        "❌",
                        (x_pos, y_pos),
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                    )

        # Special handling for XOR
        if title == "XOR Gate - IMPOSSIBLE!":
            # Draw some failed attempt lines
            ax.plot(
                [0, 1], [1, 0], "r--", linewidth=2, alpha=0.5, label="Failed attempt 1"
            )
            ax.plot(
                [0, 1], [0, 1], "b--", linewidth=2, alpha=0.5, label="Failed attempt 2"
            )
            ax.plot(
                [0.5, 0.5],
                [0, 1],
                "g--",
                linewidth=2,
                alpha=0.5,
                label="Failed attempt 3",
            )

            ax.text(
                0.5,
                0.5,
                "🚫\nNO SINGLE LINE\nWORKS!",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow"),
            )

        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_xlabel("Input 1 (x₁)", fontsize=12)
        ax.set_ylabel("Input 2 (x₂)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

    plt.tight_layout()
    plt.show()


def show_xor_solution_preview():
    """
    Preview how multiple perceptrons can solve XOR
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Preview: How Multiple Perceptrons Solve XOR!", fontsize=16, fontweight="bold"
    )

    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Step 1: First perceptron learns OR
    ax1 = axes[0]
    or_outputs = [0, 1, 1, 1]
    for i, (point, output) in enumerate(zip(points, or_outputs)):
        color = "green" if output == 1 else "red"
        marker = "o" if output == 1 else "s"
        ax1.scatter(point[0], point[1], s=200, c=color, marker=marker)
        ax1.annotate(
            f"{point}→{output}",
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    # Draw OR decision line
    x_line = np.array([0, 0.5])
    y_line = 0.5 - x_line
    ax1.plot(x_line, y_line, "b--", linewidth=2, label="OR boundary")
    ax1.set_title('Perceptron 1: Learns OR\n"Are ANY inputs ON?"')
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, alpha=0.3)

    # Step 2: Second perceptron learns AND
    ax2 = axes[1]
    and_outputs = [0, 0, 0, 1]
    for i, (point, output) in enumerate(zip(points, and_outputs)):
        color = "green" if output == 1 else "red"
        marker = "o" if output == 1 else "s"
        ax2.scatter(point[0], point[1], s=200, c=color, marker=marker)
        ax2.annotate(
            f"{point}→{output}",
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    # Draw AND decision line
    x_line = np.array([0.5, 1.5])
    y_line = 1.5 - x_line
    ax2.plot(x_line, y_line, "r--", linewidth=2, label="AND boundary")
    ax2.set_title('Perceptron 2: Learns AND\n"Are ALL inputs ON?"')
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, alpha=0.3)

    # Step 3: Third perceptron combines them
    ax3 = axes[2]
    # XOR = OR AND (NOT AND) = "exactly one is on"
    xor_outputs = [0, 1, 1, 0]
    for i, (point, output) in enumerate(zip(points, xor_outputs)):
        color = "green" if output == 1 else "red"
        marker = "o" if output == 1 else "s"
        ax3.scatter(point[0], point[1], s=200, c=color, marker=marker)
        ax3.annotate(
            f"{point}→{output}",
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax3.set_title('Perceptron 3: Combines Results\n"OR AND (NOT AND)"')
    ax3.text(
        0.5,
        0.5,
        "✅ XOR SOLVED!\nWith teamwork!",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
    )
    ax3.set_xlim(-0.2, 1.2)
    ax3.set_ylim(-0.2, 1.2)
    ax3.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

    plt.tight_layout()
    plt.show()

    print("\n🎯 THE SOLUTION PREVIEW:")
    print("=" * 25)
    print("💡 One perceptron can't solve XOR...")
    print("💪 But THREE perceptrons working together CAN!")
    print("🏗️ This is the birth of MULTI-LAYER NETWORKS!")


if __name__ == "__main__":
    print("👁️ VISUAL PROOF: Why XOR Breaks Single Perceptrons!")
    print("=" * 60)
    print("Let's SEE why single perceptrons fail at XOR...")
    print()

    # Show the basic visualization
    print("📊 BASIC VISUALIZATION:")
    print("Look at the plots - can you draw ONE line to separate ✅ from ❌?")
    plot_logic_gates()

    print("\n" + "=" * 60)
    input("Press Enter to see the detailed mathematical visualization...")

    # Show detailed decision boundaries
    print("\n🧮 DETAILED DECISION BOUNDARIES:")
    print("See the math equations and decision regions!")
    plot_decision_boundaries_detailed()

    print("\n" + "=" * 60)
    input("Press Enter to see how multiple perceptrons solve XOR...")

    # Show the solution preview
    print("\n🔮 SOLUTION PREVIEW:")
    print("Watch how TEAMWORK solves the impossible!")
    show_xor_solution_preview()

    print("\n🎯 WHAT YOU SAW:")
    print("✅ AND/OR: Clean separation with one line")
    print("❌ XOR: Impossible with one line - points are 'mixed up'")
    print("💡 Solution: Multiple perceptrons working together!")
    print()
    print("🚀 Ready to learn how Multi-Layer Perceptrons work?")
