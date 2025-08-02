"""
🧠 WHERE DOES ROSENBLATT'S FORMULA COME FROM?
============================================

You asked the PERFECT question! Let's understand exactly WHERE this
magical formula comes from and WHY it's pure genius!

new_weight = old_weight + (learning_rate × error × input)

Let's break it down piece by piece with clear examples!
"""


def the_intuition_behind_formula():
    """
    Start with the basic intuition - why this makes sense
    """
    print("🎯 THE BASIC INTUITION")
    print("=" * 25)

    print("Think of learning to cook pasta:")
    print("🍝 You try 2 cups water, pasta is too dry")
    print("🤔 What should you do? ADD MORE WATER!")
    print()
    print("The learning rule is just common sense:")
    print("   If result is TOO LOW → increase the input that caused it")
    print("   If result is TOO HIGH → decrease the input that caused it")
    print()

    print("🧠 For the perceptron:")
    print("   If output should be HIGHER → increase weights")
    print("   If output should be LOWER → decrease weights")


def the_error_part():
    """
    Explain where the error comes from
    """
    print("\n\n🎯 PART 1: THE ERROR")
    print("=" * 20)

    print("error = correct_answer - my_prediction")
    print()

    print("📊 Let's see what this means:")
    print()

    scenarios = [
        ("I should say YES but said NO", 1, 0, 1),
        ("I should say NO but said YES", 0, 1, -1),
        ("I was correct (YES)", 1, 1, 0),
        ("I was correct (NO)", 0, 0, 0),
    ]

    for description, correct, prediction, error in scenarios:
        print(f"   {description}:")
        print(f"      error = {correct} - {prediction} = {error:+}")
        if error > 0:
            print(f"      → Need to INCREASE output (error is positive)")
        elif error < 0:
            print(f"      → Need to DECREASE output (error is negative)")
        else:
            print(f"      → Perfect! No change needed")
        print()


def the_input_part():
    """
    Explain why we multiply by input
    """
    print("\n🎯 PART 2: WHY MULTIPLY BY INPUT?")
    print("=" * 35)

    print("🤔 Why not just: new_weight = old_weight + error?")
    print()
    print("💡 Because we only want to blame inputs that were ACTIVE!")
    print()

    print("🍕 Pizza example:")
    print("   You try a pizza with [good_cheese=1, expensive_price=0]")
    print("   You hate it! (prediction=1, should be 0, error=-1)")
    print()

    print("   Without input multiplication:")
    print("      cheese_weight -= 1  (blame cheese)")
    print("      price_weight -= 1   (blame price too?? But price was 0!)")
    print("   ❌ This is wrong! You can't blame price for something that wasn't there!")
    print()

    print("   With input multiplication:")
    print("      cheese_weight += (-1) × 1 = -1  (blame cheese, it was involved)")
    print(
        "      price_weight += (-1) × 0 = 0    (don't blame price, it wasn't involved)"
    )
    print("   ✅ This makes sense! Only blame what was actually present!")


def the_learning_rate_part():
    """
    Explain the learning rate
    """
    print("\n🎯 PART 3: THE LEARNING RATE")
    print("=" * 30)

    print("🤔 Why not just: new_weight = old_weight + (error × input)?")
    print()
    print("💡 Because we need to control HOW MUCH we change!")
    print()

    print("🚗 Think of learning to drive:")
    print("   You're drifting left, need to turn right")
    print()

    print("   Without learning rate:")
    print("      Turn wheel HARD right → overcorrect, now drifting right!")
    print("      Turn wheel HARD left → overcorrect again!")
    print("      → You'll zigzag forever!")
    print()

    print("   With learning rate (0.1 = gentle corrections):")
    print("      Turn wheel slightly right → small correction")
    print("      Still drifting left? Turn slightly more right")
    print("      → Gradually find the perfect steering!")


def the_complete_derivation():
    """
    Show where the formula actually comes from mathematically
    """
    print("\n\n🧮 THE MATHEMATICAL DERIVATION")
    print("=" * 35)

    print("Let's say we want to minimize our mistakes.")
    print("Our GOAL: Make the error as small as possible!")
    print()

    print("🎯 Current situation:")
    print("   Weighted sum = input₁×weight₁ + input₂×weight₂ + bias")
    print("   Prediction = 1 if weighted_sum ≥ 0, else 0")
    print("   Error = correct - prediction")
    print()

    print(
        "🤔 Question: If error is not zero, which direction should we move the weights?"
    )
    print()

    print("💡 Rosenblatt's insight:")
    print("   If error > 0 (output too low):")
    print("      → We need BIGGER weighted sum")
    print("      → So INCREASE weights for active inputs")
    print()
    print("   If error < 0 (output too high):")
    print("      → We need SMALLER weighted sum")
    print("      → So DECREASE weights for active inputs")
    print()

    print("🧠 The formula emerges naturally:")
    print("   new_weight = old_weight + (small_step × error × input)")
    print("   ↑                        ↑           ↑        ↑")
    print(
        "   │                        │           │        └─ Only blame active inputs"
    )
    print("   │                        │           └─ Direction to move")
    print("   │                        └─ Don't overshoot")
    print("   └─ Start from current weight")


def step_by_step_example():
    """
    Work through a complete example
    """
    print("\n\n📝 COMPLETE STEP-BY-STEP EXAMPLE")
    print("=" * 40)

    print("🎯 Goal: Learn that [1,1] should output 1 (AND gate)")
    print()
    print("Initial state:")
    weights = [0.2, 0.3]
    bias = 0.1
    learning_rate = 0.5

    print(f"   weights = {weights}")
    print(f"   bias = {bias}")
    print(f"   learning_rate = {learning_rate}")
    print()

    print("🧮 Try input [1,1], should output 1:")

    # Step 1: Calculate prediction
    inputs = [1, 1]
    weighted_sum = inputs[0] * weights[0] + inputs[1] * weights[1] + bias
    prediction = 1 if weighted_sum >= 0 else 0

    print(
        f"   weighted_sum = {inputs[0]}×{weights[0]} + {inputs[1]}×{weights[1]} + {bias}"
    )
    print(f"                = {weighted_sum}")
    print(f"   prediction = {prediction} (since {weighted_sum} >= 0)")
    print()

    # Step 2: Calculate error
    correct = 1
    error = correct - prediction
    print(f"   error = {correct} - {prediction} = {error}")
    print()

    if error != 0:
        print("❌ Wrong! Let's apply the learning rule:")
        print()

        # Step 3: Apply learning rule
        print("📚 Applying: new_weight = old_weight + (learning_rate × error × input)")
        print()

        for i in range(len(weights)):
            old_weight = weights[i]
            adjustment = learning_rate * error * inputs[i]
            new_weight = old_weight + adjustment

            print(f"   Weight[{i}]:")
            print(f"      old_weight = {old_weight}")
            print(
                f"      adjustment = {learning_rate} × {error} × {inputs[i]} = {adjustment}"
            )
            print(f"      new_weight = {old_weight} + {adjustment} = {new_weight}")
            print()

            weights[i] = new_weight

        # Bias adjustment
        old_bias = bias
        bias_adjustment = learning_rate * error
        new_bias = old_bias + bias_adjustment

        print(f"   Bias:")
        print(f"      old_bias = {old_bias}")
        print(f"      adjustment = {learning_rate} × {error} = {bias_adjustment}")
        print(f"      new_bias = {old_bias} + {bias_adjustment} = {new_bias}")
        print()

        bias = new_bias

        print(f"🧠 Updated brain state:")
        print(f"   weights = {weights}")
        print(f"   bias = {bias}")
        print()

        # Test again
        print("🧪 Let's test again with new weights:")
        new_weighted_sum = inputs[0] * weights[0] + inputs[1] * weights[1] + bias
        new_prediction = 1 if new_weighted_sum >= 0 else 0

        print(
            f"   weighted_sum = {inputs[0]}×{weights[0]:.1f} + {inputs[1]}×{weights[1]:.1f} + {bias:.1f}"
        )
        print(f"                = {new_weighted_sum:.1f}")
        print(f"   prediction = {new_prediction}")
        print()

        if new_prediction == correct:
            print("✅ SUCCESS! Now it's correct!")
        else:
            print("🔄 Still learning... need more adjustments")


def why_its_genius():
    """
    Explain why this formula is so brilliant
    """
    print("\n\n🎭 WHY THIS FORMULA IS PURE GENIUS")
    print("=" * 40)

    print("🏆 Rosenblatt's formula is brilliant because:")
    print()

    print("1️⃣ AUTOMATIC DIRECTION:")
    print("   - Error > 0 → weights go UP (need more activation)")
    print("   - Error < 0 → weights go DOWN (need less activation)")
    print("   - Error = 0 → no change (already perfect!)")
    print()

    print("2️⃣ PROPORTIONAL LEARNING:")
    print("   - Big mistake → big adjustment")
    print("   - Small mistake → small adjustment")
    print("   - No mistake → no adjustment")
    print()

    print("3️⃣ SELECTIVE BLAME:")
    print("   - Only adjusts weights for inputs that were active")
    print("   - Inactive inputs (0) don't get blamed for mistakes")
    print()

    print("4️⃣ CONTROLLABLE SPEED:")
    print("   - Learning rate lets you control how fast to learn")
    print("   - Prevents wild oscillations")
    print()

    print("5️⃣ GUARANTEED CONVERGENCE:")
    print("   - For linearly separable problems, this WILL find the solution!")
    print("   - Mathematical proof exists!")


if __name__ == "__main__":
    print("🧠 THE COMPLETE STORY OF ROSENBLATT'S FORMULA")
    print("=" * 55)
    print("new_weight = old_weight + (learning_rate × error × input)")
    print("=" * 55)

    # Build up the intuition
    the_intuition_behind_formula()

    # Explain each part
    the_error_part()
    the_input_part()
    the_learning_rate_part()

    # Mathematical derivation
    the_complete_derivation()

    # Complete example
    step_by_step_example()

    # Why it's genius
    why_its_genius()

    print("\n🎯 KEY INSIGHTS:")
    print("✅ The formula comes from COMMON SENSE about learning!")
    print("✅ ERROR tells us which DIRECTION to move")
    print("✅ INPUT ensures we only blame what was ACTIVE")
    print("✅ LEARNING_RATE controls how FAST we change")
    print("✅ It's not magic - it's logical reasoning made mathematical!")
    print()
    print("🚀 Now you understand the heart of machine learning!")
    print("Ready to see why this breaks down for XOR?")
