"""
🧮 THE DEEP MATHEMATICS: Why Perceptrons Draw Lines!
===================================================

You asked the PERFECT question! Let's understand exactly WHY the
weighted sum formula creates a line, and what this means mathematically.

This is the heart of understanding perceptrons!
"""

import matplotlib.pyplot as plt
import numpy as np


def the_fundamental_equation():
    """
    Start with the basic perceptron equation and show it's a line
    """
    print("🧮 THE FUNDAMENTAL EQUATION")
    print("=" * 30)

    print("🎯 Perceptron decision rule:")
    print("   if (w₁×x₁ + w₂×x₂ + bias) ≥ 0:")
    print("       output = 1")
    print("   else:")
    print("       output = 0")
    print()

    print("🔍 The KEY insight:")
    print("   The BOUNDARY between output=1 and output=0 happens when:")
    print("   w₁×x₁ + w₂×x₂ + bias = 0")
    print()

    print("🎓 Wait... this looks familiar!")
    print("   Remember from high school: The equation of a line is:")
    print("   Ax + By + C = 0")
    print()

    print("🤯 THEY'RE THE SAME THING!")
    print("   w₁×x₁ + w₂×x₂ + bias = 0")
    print("   ↑     ↑     ↑     ↑")
    print("   A  ×  x  +  B × y + C = 0")
    print()

    print("💡 So the perceptron's weighted sum IS a line equation!")
    print("   - w₁ and w₂ are the line's 'slope parameters'")
    print("   - bias is the line's 'intercept parameter'")
    print("   - The line divides the plane into two regions!")


def visualize_line_equation():
    """
    Show how different weights create different lines
    """
    print("\n\n📊 VISUALIZING THE LINE EQUATION")
    print("=" * 35)

    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("How Weights and Bias Create Different Lines!", fontsize=16)

    # Create a grid for plotting
    x = np.linspace(-2, 3, 400)

    # Different weight combinations
    examples = [
        ([1, 1, -1], "w₁=1, w₂=1, bias=-1", "x₁ + x₂ - 1 = 0"),
        ([1, -1, 0], "w₁=1, w₂=-1, bias=0", "x₁ - x₂ = 0"),
        ([2, 1, -2], "w₁=2, w₂=1, bias=-2", "2x₁ + x₂ - 2 = 0"),
        ([0, 1, -0.5], "w₁=0, w₂=1, bias=-0.5", "x₂ - 0.5 = 0"),
    ]

    for i, (weights, title, equation) in enumerate(examples):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        w1, w2, bias = weights

        if w2 != 0:  # Normal case
            # Solve for x₂: w₁×x₁ + w₂×x₂ + bias = 0
            # x₂ = -(w₁×x₁ + bias) / w₂
            y = -(w1 * x + bias) / w2
            ax.plot(x, y, "b-", linewidth=3, label=f"Decision Line")
        else:  # Vertical line case
            # x₁ = -bias / w₁
            x_line = -bias / w1
            ax.axvline(x=x_line, color="b", linewidth=3, label="Decision Line")

        # Add regions
        xx, yy = np.meshgrid(np.linspace(-1, 2, 50), np.linspace(-1, 2, 50))
        Z = w1 * xx + w2 * yy + bias

        ax.contourf(
            xx,
            yy,
            Z >= 0,
            levels=[0.5, 1.5],
            colors=["lightcoral"],
            alpha=0.3,
            label="Output = 1",
        )
        ax.contourf(
            xx,
            yy,
            Z < 0,
            levels=[-1.5, -0.5],
            colors=["lightblue"],
            alpha=0.3,
            label="Output = 0",
        )

        # Add the four standard points
        points = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for point in points:
            x1, x2 = point
            value = w1 * x1 + w2 * x2 + bias
            output = 1 if value >= 0 else 0
            color = "red" if output == 0 else "green"
            marker = "s" if output == 0 else "o"

            ax.scatter(
                x1, x2, s=150, c=color, marker=marker, edgecolors="black", linewidth=2
            )
            ax.annotate(
                f"{point}\n→{output}",
                (x1, x2),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("x₁ (Input 1)")
        ax.set_ylabel("x₂ (Input 2)")
        ax.set_title(f"{title}\n{equation}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

    print("👁️ WHAT YOU'RE SEEING:")
    print("- Blue line: The decision boundary (where weighted sum = 0)")
    print("- Red region: Where output = 0 (weighted sum < 0)")
    print("- Light red region: Where output = 1 (weighted sum ≥ 0)")
    print("- Each weight combination creates a DIFFERENT line!")


def why_lines_matter():
    """
    Explain why lines are the limitation
    """
    print("\n\n🎯 WHY LINES ARE THE LIMITATION")
    print("=" * 35)

    print("🔍 The mathematical truth:")
    print("   A perceptron can ONLY separate data with a STRAIGHT line!")
    print()

    print("📐 What this means:")
    print("   ✅ If you can draw ONE straight line to separate ✅ from ❌ → Solvable!")
    print("   ❌ If you need a curved line or multiple lines → IMPOSSIBLE!")
    print()

    print("🎓 Technical term: 'Linear Separability'")
    print("   - Linearly separable = can be separated by a line")
    print("   - Non-linearly separable = needs curves or multiple lines")
    print()

    print("🔴 AND Gate: Linearly separable")
    print("   [0,0]→0, [0,1]→0, [1,0]→0, [1,1]→1")
    print("   Line: x₁ + x₂ = 1.5 separates them perfectly!")
    print()

    print("🔵 OR Gate: Linearly separable")
    print("   [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→1")
    print("   Line: x₁ + x₂ = 0.5 separates them perfectly!")
    print()

    print("🔶 XOR Gate: NOT linearly separable")
    print("   [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0")
    print("   NO straight line can separate them!")
    print("   You need a curve or multiple lines!")


def the_geometry_explanation():
    """
    Explain the geometric meaning
    """
    print("\n\n📐 THE GEOMETRIC EXPLANATION")
    print("=" * 35)

    print("🧮 From the line equation: w₁×x₁ + w₂×x₂ + bias = 0")
    print()

    print("📏 The weights (w₁, w₂) determine the line's DIRECTION:")
    print("   - If w₁ > w₂: Line slopes more towards x₂ axis")
    print("   - If w₂ > w₁: Line slopes more towards x₁ axis")
    print("   - If w₁ = w₂: Line has 45° slope")
    print()

    print("📍 The bias determines the line's POSITION:")
    print("   - Positive bias: Shifts line away from origin")
    print("   - Negative bias: Shifts line towards origin")
    print("   - Zero bias: Line passes through origin")
    print()

    print("🎯 The line divides the plane into TWO regions:")
    print("   - Region 1: w₁×x₁ + w₂×x₂ + bias ≥ 0 → output = 1")
    print("   - Region 2: w₁×x₁ + w₂×x₂ + bias < 0 → output = 0")


def demonstrate_xor_impossibility():
    """
    Show mathematically why XOR is impossible
    """
    print("\n\n🚫 MATHEMATICAL PROOF: Why XOR is Impossible")
    print("=" * 50)

    print("🎯 For XOR to work, we need to find w₁, w₂, bias such that:")
    print()

    requirements = [
        ("[0,0] → 0", "w₁×0 + w₂×0 + bias < 0", "bias < 0"),
        ("[0,1] → 1", "w₁×0 + w₂×1 + bias ≥ 0", "w₂ + bias ≥ 0"),
        ("[1,0] → 1", "w₁×1 + w₂×0 + bias ≥ 0", "w₁ + bias ≥ 0"),
        ("[1,1] → 0", "w₁×1 + w₂×1 + bias < 0", "w₁ + w₂ + bias < 0"),
    ]

    for i, (case, full_eq, simplified) in enumerate(requirements, 1):
        print(f"   {i}. {case}: {full_eq} → {simplified}")

    print()
    print("🔍 Now let's find the contradiction:")
    print("   From equations 2 & 3: w₁ + bias ≥ 0 AND w₂ + bias ≥ 0")
    print("   Adding them: (w₁ + bias) + (w₂ + bias) ≥ 0 + 0")
    print("   Simplifying: w₁ + w₂ + 2×bias ≥ 0")
    print()
    print("   From equation 1: bias < 0")
    print("   So: w₁ + w₂ + 2×(negative number) ≥ 0")
    print("   This means: w₁ + w₂ ≥ -2×bias = positive number")
    print()
    print("   But equation 4 says: w₁ + w₂ + bias < 0")
    print("   Since bias < 0: w₁ + w₂ < -bias = positive number")
    print()
    print("   💥 CONTRADICTION!")
    print("   We need w₁ + w₂ to be BOTH ≥ some positive AND < some positive!")
    print("   This is mathematically IMPOSSIBLE!")


def the_solution_hint():
    """
    Hint at how multiple perceptrons solve this
    """
    print("\n\n💡 THE SOLUTION HINT")
    print("=" * 25)

    print("🤔 If one line can't separate XOR...")
    print("   What if we use MULTIPLE lines?")
    print()

    print("🧠 The breakthrough insight:")
    print("   XOR = (A OR B) AND NOT(A AND B)")
    print("   XOR = 'exactly one is true'")
    print()

    print("🏗️ Multi-layer approach:")
    print("   Layer 1, Perceptron A: Learns A OR B")
    print("   Layer 1, Perceptron B: Learns A AND B")
    print("   Layer 2, Perceptron C: Learns A AND NOT(B)")
    print("   Result: XOR solved!")
    print()

    print("📐 Geometrically:")
    print("   - Each perceptron draws its own line")
    print("   - Combined, they create complex decision regions")
    print("   - Can solve ANY pattern!")


def key_mathematical_insights():
    """
    Summarize the key mathematical insights
    """
    print("\n\n🎯 KEY MATHEMATICAL INSIGHTS")
    print("=" * 35)

    print("1️⃣ PERCEPTRON = LINE EQUATION:")
    print("   w₁×x₁ + w₂×x₂ + bias = 0 is literally a line!")
    print()

    print("2️⃣ WEIGHTS = LINE DIRECTION:")
    print("   Different weights → different line slopes")
    print()

    print("3️⃣ BIAS = LINE POSITION:")
    print("   Different bias → line shifts up/down")
    print()

    print("4️⃣ DECISION BOUNDARY:")
    print("   The line separates output=1 from output=0 regions")
    print()

    print("5️⃣ LINEAR SEPARABILITY:")
    print("   ✅ Can separate with one line → Perceptron can learn")
    print("   ❌ Need curves/multiple lines → Perceptron CANNOT learn")
    print()

    print("6️⃣ XOR IMPOSSIBILITY:")
    print("   Mathematical proof shows contradictory requirements")
    print()

    print("7️⃣ THE SOLUTION:")
    print("   Multiple perceptrons = Multiple lines = Complex shapes!")


if __name__ == "__main__":
    print("🧮 THE DEEP MATHEMATICS: Why Perceptrons Draw Lines!")
    print("=" * 60)
    print("Understanding the mathematical heart of perceptrons...")
    print()

    # The fundamental equation
    the_fundamental_equation()

    # Visual demonstration
    visualize_line_equation()

    # Why lines matter
    why_lines_matter()

    # Geometric explanation
    the_geometry_explanation()

    # XOR impossibility proof
    demonstrate_xor_impossibility()

    # Solution hint
    the_solution_hint()

    # Key insights
    key_mathematical_insights()

    print("\n🚀 NOW YOU UNDERSTAND:")
    print("- WHY perceptrons draw lines (it's the math!)")
    print("- WHY some problems are impossible (linear separability)")
    print("- HOW multiple perceptrons can solve anything (multiple lines!)")
    print()
    print("Ready to see Multi-Layer Perceptrons in action? 💪")
