"""
✨ CHAPTER 5: Backpropagation - The Most Elegant Algorithm!
==========================================================

Historical Context:
- 1986: Rumelhart, Hinton, and Williams solved the training problem
- The breakthrough: How to calculate errors for hidden layers
- The magic: Propagate errors BACKWARDS through the network
- The result: Multi-layer networks could finally learn automatically!

This is Rosenblatt's SRD formula extended to multiple layers! 🚀
"""

import numpy as np
import matplotlib.pyplot as plt


def the_brilliant_insight():
    """
    Explain the core insight behind backpropagation
    """
    print("💡 THE BRILLIANT INSIGHT")
    print("=" * 30)

    print("🤔 The challenge we faced:")
    print("   Output layer: 'I know my error - it's correct_answer - my_prediction'")
    print("   Hidden layer: 'But what's MY error? Nobody told me!'")
    print()

    print("🧠 Hinton, Rumelhart & Williams' genius idea:")
    print("   'Let's send the error BACKWARDS through the network!'")
    print("   'Each layer gets its fair share of blame!'")
    print()

    print("🏢 Think of it like a company hierarchy:")
    print("   📊 CEO (Output): 'We lost $1M this quarter!'")
    print("   👨‍💼 Manager A (Hidden): 'How much is MY department to blame?'")
    print("   👩‍💼 Manager B (Hidden): 'And how much is MY department to blame?'")
    print("   🔄 Backpropagation: 'Let me distribute blame fairly!'")
    print()

    print("🎯 The magic formula:")
    print("   Each hidden neuron gets error = (weighted sum of errors from above)")
    print("   Then use our beloved SRD formula: Speed × Responsibility × Direction!")


def simple_example_walkthrough():
    """
    Walk through a simple example step by step
    """
    print("\n\n🚶‍♂️ STEP-BY-STEP WALKTHROUGH")
    print("=" * 35)

    print("Let's use a tiny network to understand the concept:")
    print()
    print("🏗️ Network structure:")
    print("   Input: 1 neuron (x)")
    print("   Hidden: 1 neuron (h)")
    print("   Output: 1 neuron (y)")
    print()
    print("   x → h → y")
    print()

    # Initialize simple network
    print("🎲 Initial values:")
    x = 1.0  # Input
    w1 = 0.5  # Weight from input to hidden
    w2 = 0.8  # Weight from hidden to output
    target = 0.0  # What we want the output to be
    learning_rate = 0.5

    print(f"   Input (x): {x}")
    print(f"   Weight input→hidden (w1): {w1}")
    print(f"   Weight hidden→output (w2): {w2}")
    print(f"   Target output: {target}")
    print(f"   Learning rate: {learning_rate}")
    print()

    print("🔄 FORWARD PASS:")
    print("   Step 1: Calculate hidden neuron")
    h = w1 * x  # Hidden neuron activation (simplified, no bias)
    print(f"      h = w1 × x = {w1} × {x} = {h}")

    print("   Step 2: Calculate output neuron")
    y = w2 * h  # Output neuron activation
    print(f"      y = w2 × h = {w2} × {h} = {y}")

    print("   Step 3: Calculate final error")
    output_error = target - y
    print(f"      output_error = target - y = {target} - {y} = {output_error}")
    print()

    print("🔙 BACKWARD PASS (The Magic!):")
    print("   Step 1: Output layer error (we know this)")
    print(f"      output_error = {output_error}")

    print("   Step 2: Hidden layer error (calculate from output error)")
    # Hidden error = output_error × weight_connecting_hidden_to_output
    hidden_error = output_error * w2
    print(
        f"      hidden_error = output_error × w2 = {output_error} × {w2} = {hidden_error}"
    )
    print("      💡 Hidden neuron gets blame proportional to its connection strength!")

    print("   Step 3: Update weights using SRD formula")
    print("      For w2 (hidden→output):")
    w2_new = w2 + learning_rate * output_error * h
    print(f"         w2_new = w2 + (lr × error × input)")
    print(
        f"         w2_new = {w2} + ({learning_rate} × {output_error} × {h}) = {w2_new}"
    )

    print("      For w1 (input→hidden):")
    w1_new = w1 + learning_rate * hidden_error * x
    print(f"         w1_new = w1 + (lr × error × input)")
    print(
        f"         w1_new = {w1} + ({learning_rate} × {hidden_error} × {x}) = {w1_new}"
    )

    print()
    print("✨ MAGIC ACCOMPLISHED!")
    print("   Each weight updated using its own 'fair share' of the error!")


def the_chain_rule_connection():
    """
    Explain how this connects to the chain rule from calculus
    """
    print("\n\n⛓️ THE CHAIN RULE CONNECTION")
    print("=" * 35)

    print("🎓 For those who know calculus:")
    print("   Backpropagation is just the CHAIN RULE applied to neural networks!")
    print()

    print("📐 Chain rule in calculus:")
    print("   If y = f(g(x)), then dy/dx = (dy/df) × (df/dx)")
    print()

    print("🧠 In neural networks:")
    print(
        "   Error flows backwards: ∂Error/∂w1 = (∂Error/∂output) × (∂output/∂hidden) × (∂hidden/∂w1)"
    )
    print()

    print("💡 In simple terms:")
    print("   'How much does w1 affect the final error?'")
    print("   = 'Error sensitivity' × 'Chain of influences'")
    print()

    print("🎯 This is why it's elegant:")
    print("   It's not some random algorithm - it's fundamental mathematics!")
    print("   The most efficient way to calculate gradients!")


def company_analogy_detailed():
    """
    Detailed company analogy for error propagation
    """
    print("\n\n🏢 COMPANY ANALOGY: Blame Distribution")
    print("=" * 45)

    print("🎯 Scenario: Company lost $100,000 this quarter")
    print()

    print("🏗️ Company structure:")
    print("   CEO (Output layer)")
    print("   ├── Sales Manager (Hidden neuron 1)")
    print("   └── Marketing Manager (Hidden neuron 2)")
    print("       ├── Employee A (Input 1)")
    print("       └── Employee B (Input 2)")
    print()

    print("🔄 How blame gets distributed:")
    print()

    print("   Step 1: CEO knows the total loss")
    print("      'We lost $100,000 total'")
    print()

    print("   Step 2: Distribute blame to managers")
    print("      Sales Manager impact: 60% of CEO's decisions")
    print("      Marketing Manager impact: 40% of CEO's decisions")
    print("      Sales Manager's blame: $100,000 × 0.6 = $60,000")
    print("      Marketing Manager's blame: $100,000 × 0.4 = $40,000")
    print()

    print("   Step 3: Managers distribute blame to employees")
    print("      Employee A impact on Marketing: 70%")
    print("      Employee B impact on Marketing: 30%")
    print("      Employee A's blame: $40,000 × 0.7 = $28,000")
    print("      Employee B's blame: $40,000 × 0.3 = $12,000")
    print()

    print("   Step 4: Everyone improves based on their blame")
    print("      'Bigger blame = Bigger changes needed!'")
    print()

    print("💡 This is EXACTLY how backpropagation works:")
    print("   Errors flow backwards, weighted by connection strengths!")


def implement_backprop_xor():
    """
    Implement backpropagation to learn XOR from scratch
    """
    print("\n\n🔧 IMPLEMENTING BACKPROP FOR XOR")
    print("=" * 40)

    print("Let's watch backpropagation learn XOR automatically!")
    print()

    # XOR training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print("🎯 Goal: Learn XOR function")
    print("   Training data:")
    for i, (inputs, target) in enumerate(zip(X, y)):
        print(f"      {inputs} → {target[0]}")
    print()

    # Initialize network
    np.random.seed(42)  # For reproducible results
    input_size = 2
    hidden_size = 2
    output_size = 1
    learning_rate = 5.0  # Higher learning rate for faster learning

    # Initialize weights randomly
    W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
    b1 = np.random.uniform(-1, 1, (1, hidden_size))
    W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
    b2 = np.random.uniform(-1, 1, (1, output_size))

    print("🎲 Initial random weights:")
    print(f"   W1 (input→hidden): \n{W1}")
    print(f"   W2 (hidden→output): \n{W2}")
    print()

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def sigmoid_derivative(x):
        return x * (1 - x)

    # Training loop
    print("🎓 TRAINING PROCESS:")
    errors_history = []

    for epoch in range(10000):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Calculate loss
        loss = np.mean((y - a2) ** 2)
        errors_history.append(loss)

        # Backward pass
        # Output layer gradients
        dz2 = (a2 - y) * sigmoid_derivative(a2)
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients (the magic of backprop!)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * sigmoid_derivative(a1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        # Print progress
        if epoch % 2000 == 0:
            print(f"   Epoch {epoch}: Loss = {loss:.6f}")

    print()
    print("🧪 FINAL TEST:")
    final_predictions = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)

    print("   Input | Target | Prediction | Correct?")
    print("   ------|--------|------------|----------")

    all_correct = True
    for i, (inputs, target, pred) in enumerate(zip(X, y, final_predictions)):
        pred_binary = 1 if pred[0] > 0.5 else 0
        correct = "✅" if pred_binary == target[0] else "❌"
        if pred_binary != target[0]:
            all_correct = False
        print(f"   {inputs} |   {target[0]}    |   {pred[0]:.3f}    |    {correct}")

    print()
    if all_correct:
        print("🎉 SUCCESS! Backpropagation learned XOR perfectly!")
    else:
        print("🔄 Still learning... might need more epochs")

    return errors_history


def visualize_learning_process():
    """
    Visualize how the network learns over time
    """
    print("\n\n📊 VISUALIZING THE LEARNING PROCESS")
    print("=" * 40)

    # Run the training and get error history
    errors = implement_backprop_xor()

    # Plot learning curve
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(errors)
    plt.title("Learning Curve: Error Decreases Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Show error decrease in first 1000 epochs
    plt.plot(errors[:1000])
    plt.title("Error Decrease (First 1000 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("👁️ WHAT YOU'RE SEEING:")
    print("   Left: Complete learning curve (log scale)")
    print("   Right: Rapid initial learning in first 1000 epochs")
    print("   The network learns XOR from scratch using backpropagation!")


def key_insights_backprop():
    """
    Summarize key insights about backpropagation
    """
    print("\n\n🎯 KEY INSIGHTS ABOUT BACKPROPAGATION")
    print("=" * 45)

    print("1️⃣ IT'S ROSENBLATT'S FORMULA FOR MULTIPLE LAYERS:")
    print("   Same SRD principle: Speed × Responsibility × Direction")
    print("   But now 'Direction' comes from layers above!")
    print()

    print("2️⃣ ERROR FLOWS BACKWARDS:")
    print("   Output layer: Knows error directly")
    print("   Hidden layers: Get 'fair share' of blame")
    print()

    print("3️⃣ WEIGHT OF CONNECTION = AMOUNT OF BLAME:")
    print("   Strong connection → More blame")
    print("   Weak connection → Less blame")
    print()

    print("4️⃣ IT'S MATHEMATICALLY OPTIMAL:")
    print("   Based on chain rule from calculus")
    print("   Most efficient way to calculate gradients")
    print()

    print("5️⃣ AUTOMATIC LEARNING:")
    print("   No hand-designing weights!")
    print("   Network figures out the solution by itself!")
    print()

    print("6️⃣ ENABLES DEEP LEARNING:")
    print("   Can train networks with hundreds of layers")
    print("   Foundation of modern AI (GPT, ChatGPT, etc.)")


if __name__ == "__main__":
    print("✨ BACKPROPAGATION: The Most Elegant Algorithm in Computer Science!")
    print("=" * 70)
    print("How to teach networks to teach themselves!")
    print()

    # The brilliant insight
    the_brilliant_insight()

    # Simple walkthrough
    simple_example_walkthrough()

    # Chain rule connection
    the_chain_rule_connection()

    # Company analogy
    company_analogy_detailed()

    # Implement backprop for XOR
    visualize_learning_process()

    # Key insights
    key_insights_backprop()

    print("\n🌟 YOU NOW UNDERSTAND:")
    print("   How errors flow backwards through networks")
    print("   Why it's just an extension of Rosenblatt's formula")
    print("   How networks learn complex patterns automatically")
    print()
    print("🚀 NEXT: CNNs - How AI Learned to See!")
    print("The architecture that revolutionized computer vision!")
