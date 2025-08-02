"""
📐 GRADIENT DESCENT: The Calculus Connection
===========================================

YES! Gradient descent is 100% CALCULUS!
Let me show you exactly how calculus concepts power all of AI learning!

This is why calculus is so important for understanding AI!
"""

import numpy as np
import matplotlib.pyplot as plt


def calculus_concepts_in_gradient_descent():
    """
    Show the direct calculus concepts used in gradient descent
    """
    print("📐 GRADIENT DESCENT = APPLIED CALCULUS!")
    print("=" * 45)

    print("🎯 CORE CALCULUS CONCEPTS USED:")
    print()

    print("1️⃣ DERIVATIVES (Single Variable):")
    print("   f'(x) = df/dx = slope of function at point x")
    print("   Tells us: 'If I move x slightly, how much does f(x) change?'")
    print("   Example: f(x) = x², f'(x) = 2x")
    print()

    print("2️⃣ PARTIAL DERIVATIVES (Multiple Variables):")
    print("   ∂f/∂x = how f changes when only x changes")
    print("   ∂f/∂y = how f changes when only y changes")
    print("   Example: f(x,y) = x² + y², ∂f/∂x = 2x, ∂f/∂y = 2y")
    print()

    print("3️⃣ GRADIENTS (Vector of Partial Derivatives):")
    print("   ∇f = [∂f/∂x₁, ∂f/∂x₂, ∂f/∂x₃, ...]")
    print("   Points in direction of steepest increase")
    print("   Example: ∇f = [2x, 2y] for f(x,y) = x² + y²")
    print()

    print("4️⃣ CHAIN RULE (For Nested Functions):")
    print("   If y = f(g(x)), then dy/dx = (dy/df) × (df/dx)")
    print("   Critical for backpropagation through layers!")
    print()

    print("💡 GRADIENT DESCENT = Finding where derivative = 0!")
    print("   At minimum: f'(x) = 0 (slope is flat)")
    print("   We iteratively approach this point!")


def simple_derivative_example():
    """
    Show derivatives in action for optimization
    """
    print("\n\n🧮 CALCULUS IN ACTION: Finding Minimum")
    print("=" * 45)

    print("📊 FUNCTION: f(x) = x² - 6x + 9")
    print("   This is a parabola opening upward")
    print("   Has one minimum point")
    print()

    print("📐 CALCULUS APPROACH:")
    print("   Step 1: Find derivative f'(x)")
    print("   f(x) = x² - 6x + 9")
    print("   f'(x) = 2x - 6")
    print()

    print("   Step 2: Set derivative = 0")
    print("   2x - 6 = 0")
    print("   2x = 6")
    print("   x = 3")
    print()

    print("   Step 3: Verify it's minimum")
    print("   f''(x) = 2 > 0 ✅ (positive = minimum)")
    print("   Minimum value: f(3) = 9 - 18 + 9 = 0")
    print()

    print("🎯 ANALYTICAL SOLUTION: Minimum at (3, 0)")
    print()

    print("🔄 GRADIENT DESCENT APPROACH:")
    print("   Start with guess: x = 0")
    print("   Learning rate: α = 0.1")
    print()

    # Gradient descent simulation
    x = 0.0
    learning_rate = 0.1

    def f(x):
        return x**2 - 6 * x + 9

    def f_derivative(x):
        return 2 * x - 6

    print("   Step | x value | f(x) | f'(x) | Next x")
    print("   -----|---------|------|-------|--------")

    for i in range(8):
        fx = f(x)
        derivative = f_derivative(x)
        next_x = x - learning_rate * derivative

        print(f"   {i:4} | {x:7.3f} | {fx:4.1f} | {derivative:5.1f} | {next_x:6.3f}")
        x = next_x

    print()
    print(f"🎉 GRADIENT DESCENT RESULT: x ≈ {x:.3f}")
    print(f"   Compare to calculus: x = 3.000")
    print(f"   Gradient descent approximated the calculus solution!")


def multivariable_example():
    """
    Show partial derivatives for multiple variables
    """
    print("\n\n🌐 MULTIVARIABLE CALCULUS: Partial Derivatives")
    print("=" * 55)

    print("📊 FUNCTION: f(x,y) = x² + y² - 2x - 4y + 5")
    print("   This is a bowl-shaped surface in 3D")
    print("   Has one minimum point")
    print()

    print("📐 PARTIAL DERIVATIVES:")
    print("   ∂f/∂x = 2x - 2")
    print("   ∂f/∂y = 2y - 4")
    print()

    print("   GRADIENT: ∇f = [2x - 2, 2y - 4]")
    print()

    print("📐 ANALYTICAL SOLUTION:")
    print("   Set both partial derivatives = 0:")
    print("   2x - 2 = 0 → x = 1")
    print("   2y - 4 = 0 → y = 2")
    print("   Minimum at (1, 2) with value f(1,2) = 1")
    print()

    print("🔄 GRADIENT DESCENT:")
    print("   Start: (x,y) = (0,0)")
    print("   Learning rate: α = 0.1")
    print()

    # Multivariable gradient descent
    x, y = 0.0, 0.0
    learning_rate = 0.1

    def f_multi(x, y):
        return x**2 + y**2 - 2 * x - 4 * y + 5

    def gradient_x(x, y):
        return 2 * x - 2

    def gradient_y(x, y):
        return 2 * y - 4

    print("   Step | x value | y value | f(x,y) | ∇x  | ∇y")
    print("   -----|---------|---------|--------|-----|----")

    for i in range(8):
        fxy = f_multi(x, y)
        grad_x = gradient_x(x, y)
        grad_y = gradient_y(x, y)

        print(
            f"   {i:4} | {x:7.3f} | {y:7.3f} | {fxy:6.1f} | {grad_x:3.1f} | {grad_y:3.1f}"
        )

        # Update both variables
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

    print()
    print(f"🎉 RESULT: ({x:.3f}, {y:.3f})")
    print(f"   Calculus solution: (1.000, 2.000)")
    print(f"   Very close! ✅")


def chain_rule_explanation():
    """
    Explain the chain rule and its importance for neural networks
    """
    print("\n\n⛓️ CHAIN RULE: The Heart of Backpropagation")
    print("=" * 50)

    print("📐 CHAIN RULE FORMULA:")
    print("   If z = f(y) and y = g(x), then:")
    print("   dz/dx = (dz/dy) × (dy/dx)")
    print()

    print("🧮 SIMPLE EXAMPLE:")
    print("   z = y²")
    print("   y = 3x + 1")
    print("   Therefore: z = (3x + 1)²")
    print()

    print("   METHOD 1 - Direct:")
    print("   z = (3x + 1)² = 9x² + 6x + 1")
    print("   dz/dx = 18x + 6")
    print()

    print("   METHOD 2 - Chain Rule:")
    print("   dz/dy = 2y")
    print("   dy/dx = 3")
    print("   dz/dx = (dz/dy) × (dy/dx) = 2y × 3 = 6y")
    print("   Substitute y = 3x + 1:")
    print("   dz/dx = 6(3x + 1) = 18x + 6 ✅ Same result!")
    print()

    print("🧠 WHY THIS MATTERS FOR NEURAL NETWORKS:")
    print("   Neural network: Input → Layer1 → Layer2 → Output")
    print(
        "   Chain rule: ∂Error/∂Input = (∂Error/∂Output) × (∂Output/∂Layer2) × (∂Layer2/∂Layer1) × (∂Layer1/∂Input)"
    )
    print("   This IS backpropagation! 🤯")


def neural_network_calculus():
    """
    Show calculus in neural networks explicitly
    """
    print("\n\n🧠 CALCULUS IN NEURAL NETWORKS")
    print("=" * 35)

    print("📊 SIMPLE NEURAL NETWORK:")
    print("   Input: x")
    print("   Hidden: h = σ(w₁x + b₁)")
    print("   Output: y = σ(w₂h + b₂)")
    print("   Error: E = ½(target - y)²")
    print("   (σ = sigmoid activation function)")
    print()

    print("🎯 GOAL: Find ∂E/∂w₁ (how error changes with first weight)")
    print()

    print("⛓️ CHAIN RULE APPLICATION:")
    print("   ∂E/∂w₁ = (∂E/∂y) × (∂y/∂h) × (∂h/∂w₁)")
    print()

    print("📐 CALCULATE EACH PART:")
    print("   ∂E/∂y = -(target - y)")
    print("   ∂y/∂h = σ'(w₂h + b₂) × w₂")
    print("   ∂h/∂w₁ = σ'(w₁x + b₁) × x")
    print()

    print("🔄 COMBINE USING CHAIN RULE:")
    print("   ∂E/∂w₁ = -(target - y) × σ'(w₂h + b₂) × w₂ × σ'(w₁x + b₁) × x")
    print()

    print("💡 GRADIENT DESCENT UPDATE:")
    print("   w₁ₙₑw = w₁ₒₗd - α × ∂E/∂w₁")
    print()

    print("🎉 THIS IS BACKPROPAGATION!")
    print("   Pure calculus applied to neural network learning!")


def why_calculus_makes_ai_possible():
    """
    Explain why calculus is fundamental to AI
    """
    print("\n\n🌟 WHY CALCULUS MAKES AI POSSIBLE")
    print("=" * 40)

    print("🎯 CALCULUS GIVES US:")
    print()

    print("1️⃣ OPTIMIZATION POWER:")
    print("   Find best parameters automatically")
    print("   No need to guess - calculus finds the optimum!")
    print()

    print("2️⃣ EFFICIENCY:")
    print("   Derivatives tell us exactly which direction to move")
    print("   Much faster than random search!")
    print()

    print("3️⃣ SCALABILITY:")
    print("   Works for millions/billions of parameters")
    print("   Same calculus principles scale to any size!")
    print()

    print("4️⃣ UNIVERSAL APPLICABILITY:")
    print("   Any differentiable function can be optimized")
    print("   Enables all types of neural networks!")
    print()

    print("💡 WITHOUT CALCULUS:")
    print("   ❌ No gradient descent")
    print("   ❌ No backpropagation")
    print("   ❌ No deep learning")
    print("   ❌ No modern AI!")
    print()

    print("✅ WITH CALCULUS:")
    print("   ✅ Automatic optimization")
    print("   ✅ Efficient learning")
    print("   ✅ Deep networks possible")
    print("   ✅ Modern AI revolution!")


def calculus_concepts_summary():
    """
    Summarize the key calculus concepts used in AI
    """
    print("\n\n📚 CALCULUS CONCEPTS IN AI - SUMMARY")
    print("=" * 45)

    concepts = [
        ("Derivatives", "f'(x)", "Rate of change, slope", "Single weight updates"),
        (
            "Partial Derivatives",
            "∂f/∂x",
            "Change w.r.t. one variable",
            "Multiple weight updates",
        ),
        (
            "Gradients",
            "∇f",
            "Vector of partial derivatives",
            "Direction to move all weights",
        ),
        (
            "Chain Rule",
            "dy/dx = (dy/du)(du/dx)",
            "Derivative of composition",
            "Backpropagation through layers",
        ),
        (
            "Critical Points",
            "f'(x) = 0",
            "Where derivative is zero",
            "Optimal weight values",
        ),
        (
            "Second Derivatives",
            "f''(x)",
            "Concavity, min/max test",
            "Learning rate adaptation",
        ),
    ]

    print(
        "   Concept           | Notation        | Meaning              | AI Application"
    )
    print(
        "   ------------------|-----------------|----------------------|------------------"
    )
    for concept, notation, meaning, application in concepts:
        print(f"   {concept:<17} | {notation:<15} | {meaning:<20} | {application}")

    print()
    print("🎯 THE BIG PICTURE:")
    print("   Calculus = Language of change and optimization")
    print("   AI = Finding optimal parameters through change")
    print("   Perfect match! 🎯")


def connection_to_previous_learning():
    """
    Connect back to everything we've learned
    """
    print("\n\n🔗 CONNECTION TO EVERYTHING WE'VE LEARNED")
    print("=" * 50)

    print("✅ NOW YOU UNDERSTAND WHY:")
    print()

    print("🧠 McCulloch-Pitts (1943):")
    print("   Had the computation but no calculus-based learning")
    print()

    print("🎯 Perceptron (1957):")
    print("   Rosenblatt applied calculus (gradient descent) to learning")
    print("   His formula = derivative of error function!")
    print()

    print("🔗 Multi-layer (1980s):")
    print("   Chain rule enabled backpropagation")
    print("   Calculus through multiple layers!")
    print()

    print("👁️ CNNs:")
    print("   Same calculus, applied to convolution operations")
    print("   Partial derivatives of filters!")
    print()

    print("🧠 RNNs:")
    print("   Chain rule through time steps")
    print("   Calculus with memory!")
    print()

    print("💡 THE UNIVERSAL TRUTH:")
    print("   Every AI breakthrough = New way to apply calculus!")
    print("   All learning = Gradient descent = Applied calculus!")


if __name__ == "__main__":
    print("📐 GRADIENT DESCENT: The Calculus Connection")
    print("=" * 50)
    print("How calculus powers ALL of artificial intelligence!")
    print()

    # Core calculus concepts
    calculus_concepts_in_gradient_descent()

    # Simple derivative example
    simple_derivative_example()

    # Multivariable example
    multivariable_example()

    # Chain rule explanation
    chain_rule_explanation()

    # Neural network calculus
    neural_network_calculus()

    # Why calculus makes AI possible
    why_calculus_makes_ai_possible()

    # Summary of concepts
    calculus_concepts_summary()

    # Connection to previous learning
    connection_to_previous_learning()

    print("\n🌟 CALCULUS CONNECTION REVEALED!")
    print("You now understand the mathematical foundation")
    print("that makes ALL of AI possible!")
    print()
    print("Ready for RNN memory math with full calculus understanding? 🧠⚡")
