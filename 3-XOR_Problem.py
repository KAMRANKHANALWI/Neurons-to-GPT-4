"""
💥 CHAPTER 3: The XOR Problem - Why Perceptrons Failed!
======================================================

Historical Context:
- 1969: Minsky and Papert published "Perceptrons" book
- Showed that perceptrons CANNOT learn XOR (Exclusive OR)
- This caused the "AI Winter" - funding dried up, research stopped
- People thought neural networks were useless!

But this "failure" led to the next breakthrough: Multi-layer networks!

Let's understand WHY XOR is impossible for single perceptrons...
"""

import numpy as np
import matplotlib.pyplot as plt


def what_is_xor():
    """
    Explain what XOR means in simple terms
    """
    print("🤔 WHAT IS XOR (EXCLUSIVE OR)?")
    print("=" * 35)

    print("XOR means: 'Output 1 if inputs are DIFFERENT'")
    print()

    print("📊 The XOR truth table:")
    print("   [0, 0] → 0  (both same → output 0)")
    print("   [0, 1] → 1  (different → output 1)")
    print("   [1, 0] → 1  (different → output 1)")
    print("   [1, 1] → 0  (both same → output 0)")
    print()

    print("🎯 Real-world examples of XOR:")
    print("   🚪 'Turn light ON if exactly ONE switch is pressed'")
    print("   🎮 'Player wins if exactly ONE of them reaches the goal'")
    print(
        "   💑 'Go on date if exactly ONE of us is free (not both busy or both free)'"
    )


def why_and_or_work():
    """
    First show why AND and OR are easy for perceptrons
    """
    print("\n\n✅ WHY AND/OR ARE EASY")
    print("=" * 25)

    print("Let's visualize why AND and OR work:")
    print()

    print("🔴 AND Gate: Output 1 only in top-right corner")
    print("   [0,0] → 0   [0,1] → 0")
    print("   [1,0] → 0   [1,1] → 1")
    print()
    print("   You can draw ONE straight line to separate:")
    print("   ❌❌")
    print("   ❌✅  ← Line goes here")
    print()

    print("🔵 OR Gate: Output 0 only in bottom-left corner")
    print("   [0,0] → 0   [0,1] → 1")
    print("   [1,0] → 1   [1,1] → 1")
    print()
    print("   You can draw ONE straight line to separate:")
    print("   ❌✅")
    print("   ✅✅  ← Line goes here")


def why_xor_fails():
    """
    Show why XOR is impossible
    """
    print("\n\n❌ WHY XOR IS IMPOSSIBLE")
    print("=" * 30)

    print("🔶 XOR Gate: Output 1 for different inputs")
    print("   [0,0] → 0   [0,1] → 1")
    print("   [1,0] → 1   [1,1] → 0")
    print()
    print("   Try to draw ONE straight line to separate:")
    print("   ❌✅")
    print("   ✅❌")
    print()
    print("   🚫 IMPOSSIBLE! You need a CURVED line or TWO lines!")
    print("   But perceptrons can only draw ONE straight line!")


def the_line_drawing_analogy():
    """
    Explain using the line drawing analogy
    """
    print("\n\n📏 THE LINE DRAWING LIMITATION")
    print("=" * 35)

    print("Think of a perceptron as someone who can only draw STRAIGHT lines:")
    print()

    print("🎨 The perceptron's job:")
    print("   'Draw ONE straight line to separate the ✅ from the ❌'")
    print()

    print("✅ AND pattern:")
    print("   ❌ ❌")
    print("   ❌ ✅  ← Easy! One line separates them")
    print()

    print("✅ OR pattern:")
    print("   ❌ ✅")
    print("   ✅ ✅  ← Easy! One line separates them")
    print()

    print("❌ XOR pattern:")
    print("   ❌ ✅")
    print("   ✅ ❌  ← Impossible! No single line can separate them")
    print()

    print("💡 You need either:")
    print("   1. A CURVED line (but perceptrons can't curve!)")
    print("   2. TWO straight lines (but perceptrons can only draw one!)")


def demonstrate_xor_failure():
    """
    Show a perceptron trying and failing to learn XOR
    """
    print("\n\n🔬 WATCHING XOR FAILURE IN ACTION")
    print("=" * 40)

    print("Let's watch a perceptron try to learn XOR and fail...")
    print()

    # XOR training data
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_outputs = np.array([0, 1, 1, 0])

    print("🎯 Goal: Learn XOR pattern")
    for i, (inputs, output) in enumerate(zip(xor_inputs, xor_outputs)):
        print(f"   {inputs} → {output}")
    print()

    # Try to learn with different starting weights
    print("🧪 Experiment: Different starting weights")

    starting_weights = [
        ([0.5, 0.5], "Positive weights"),
        ([-0.5, -0.5], "Negative weights"),
        ([0.5, -0.5], "Mixed weights"),
        ([1.0, 1.0], "Strong positive"),
    ]

    for weights, description in starting_weights:
        print(f"\n   🎲 Try {description}: {weights}")

        # Test each XOR example
        successes = 0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            weighted_sum = inputs[0] * weights[0] + inputs[1] * weights[1]
            prediction = 1 if weighted_sum >= 0 else 0

            if prediction == expected:
                successes += 1
                status = "✅"
            else:
                status = "❌"

            print(f"      {inputs} → {prediction} (expected {expected}) {status}")

        accuracy = successes / 4 * 100
        print(f"      Accuracy: {accuracy:.0f}%")

        if accuracy < 100:
            print(f"      💔 Failed to learn XOR!")


def the_mathematical_proof():
    """
    Simple mathematical explanation of why it's impossible
    """
    print("\n\n🧮 THE MATHEMATICAL PROOF")
    print("=" * 30)

    print("Let's prove mathematically why XOR is impossible:")
    print()

    print("🎯 Perceptron formula: output = 1 if (w₁×x₁ + w₂×x₂ + bias) ≥ 0")
    print()

    print("📝 For XOR to work, we need:")
    print("   [0,0]: w₁×0 + w₂×0 + bias < 0  →  bias < 0")
    print("   [0,1]: w₁×0 + w₂×1 + bias ≥ 0  →  w₂ + bias ≥ 0")
    print("   [1,0]: w₁×1 + w₂×0 + bias ≥ 0  →  w₁ + bias ≥ 0")
    print("   [1,1]: w₁×1 + w₂×1 + bias < 0  →  w₁ + w₂ + bias < 0")
    print()

    print("🤯 The contradiction:")
    print("   From equations 2 & 3: w₁ + bias ≥ 0 AND w₂ + bias ≥ 0")
    print("   Adding them: w₁ + w₂ + 2×bias ≥ 0")
    print("   Since bias < 0, let's say bias = -1:")
    print("   Then: w₁ + w₂ + 2×(-1) ≥ 0")
    print("   So: w₁ + w₂ ≥ 2")
    print()
    print("   But equation 4 says: w₁ + w₂ + bias < 0")
    print("   With bias = -1: w₁ + w₂ - 1 < 0")
    print("   So: w₁ + w₂ < 1")
    print()
    print("   💥 CONTRADICTION! We need w₁ + w₂ ≥ 2 AND w₁ + w₂ < 1")
    print("   This is mathematically IMPOSSIBLE!")


def the_historical_impact():
    """
    Explain the historical impact of this discovery
    """
    print("\n\n📚 THE HISTORICAL IMPACT")
    print("=" * 30)

    print("🏛️ 1969: Minsky & Papert's 'Perceptrons' book")
    print("   💔 Showed perceptrons can't learn XOR")
    print("   💸 Funding for neural network research dried up")
    print("   🥶 'AI Winter' began - research nearly stopped")
    print("   😔 People thought neural networks were useless")
    print()

    print("⏰ 1970s: The Dark Years")
    print("   🚫 Very little neural network research")
    print("   🎓 Students advised to avoid the field")
    print("   💰 Government stopped funding AI research")
    print()

    print("💡 But some visionaries kept working:")
    print("   🧠 'What if we use MULTIPLE perceptrons?'")
    print("   🤔 'What if we stack them in layers?'")
    print("   🔥 This led to MULTI-LAYER PERCEPTRONS!")


def the_solution_preview():
    """
    Preview how multi-layer networks solve XOR
    """
    print("\n\n🔮 PREVIEW: THE SOLUTION")
    print("=" * 25)

    print("💡 The breakthrough insight:")
    print("   'If one perceptron can't draw curved lines...'")
    print("   'What if we use MULTIPLE perceptrons working together?'")
    print()

    print("🏗️ Multi-layer approach to XOR:")
    print("   Layer 1: Two perceptrons detect patterns")
    print("      Perceptron A: Detects 'first input OR second input'")
    print("      Perceptron B: Detects 'first input AND second input'")
    print("   Layer 2: One perceptron combines results")
    print("      Final: 'A AND (NOT B)' = XOR!")
    print()

    print("🎯 This is like having a TEAM solve the problem:")
    print("   👨‍💼 Manager: 'I can't solve this alone...'")
    print("   👥 Team: 'Let's work together!'")
    print("   ✅ Problem solved!")


def key_insights():
    """
    Summarize the key insights
    """
    print("\n\n🎯 KEY INSIGHTS FROM THE XOR PROBLEM")
    print("=" * 45)

    print("1️⃣ LIMITATION DISCOVERED:")
    print("   ❌ Single perceptrons can only solve 'linearly separable' problems")
    print("   ❌ They can only draw ONE straight line")
    print()

    print("2️⃣ MATHEMATICAL PROOF:")
    print("   🧮 XOR creates contradictory requirements")
    print("   🚫 Mathematically impossible for single perceptron")
    print()

    print("3️⃣ HISTORICAL IMPACT:")
    print("   🥶 Caused the 'AI Winter' of 1970s")
    print("   💔 Nearly killed neural network research")
    print()

    print("4️⃣ THE SOLUTION PREVIEW:")
    print("   💡 Multiple perceptrons working together!")
    print("   🏗️ Multi-layer networks can solve ANY problem!")
    print("   🚀 This led to modern deep learning!")
    print()

    print("5️⃣ THE QUANTUM PHYSICS PARALLEL:")
    print("   ⚛️ Just like how classical physics 'failed' for atoms...")
    print("   🔬 But led to quantum mechanics!")
    print("   🌟 Sometimes 'failures' lead to greater breakthroughs!")


if __name__ == "__main__":
    print("💥 THE XOR PROBLEM: Where Beautiful Learning Breaks!")
    print("=" * 60)
    print("The 'failure' that led to the greatest breakthrough in AI!")
    print()

    # Explain XOR
    what_is_xor()

    # Show why AND/OR work
    why_and_or_work()

    # Show why XOR fails
    why_xor_fails()

    # Line drawing analogy
    the_line_drawing_analogy()

    # Demonstrate failure
    demonstrate_xor_failure()

    # Mathematical proof
    the_mathematical_proof()

    # Historical impact
    the_historical_impact()

    # Solution preview
    the_solution_preview()

    # Key insights
    key_insights()

    print("\n🔜 NEXT: Multi-Layer Perceptrons - Breaking the Linear Barrier!")
    print("Finally solving the problem that stumped everyone!")
    print("This is where things get REALLY exciting! 🚀")
