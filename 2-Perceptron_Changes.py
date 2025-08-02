"""
🎩 THE LEARNING MAGIC: How Perceptron Changes Its Mind!
======================================================

Let's watch a perceptron learn step by step, like watching a child
figure out what makes a good ice cream!

We'll use the simplest example: learning the AND gate
Input: [0,0] → 0, [0,1] → 0, [1,0] → 0, [1,1] → 1
"""

import random


def learning_step_by_step():
    """
    Let's manually walk through EXACTLY how learning works
    """
    print("🧠 LEARNING THE AND GATE STEP BY STEP")
    print("=" * 45)

    print("🎯 Goal: Learn that output is 1 ONLY when BOTH inputs are 1")
    print("Training examples:")
    print("   [0,0] should give 0")
    print("   [0,1] should give 0")
    print("   [1,0] should give 0")
    print("   [1,1] should give 1")
    print()

    # Step 1: Start with random weights
    print("STEP 1: Start with random guesses")
    print("-" * 35)
    weights = [0.3, 0.7]  # Random starting weights
    bias = 0.2  # Random starting bias
    learning_rate = 0.5  # Learn at medium speed

    print(f"🎲 Random starting weights: {weights}")
    print(f"🎲 Random starting bias: {bias}")
    print(f"🎓 Learning rate: {learning_rate}")
    print()

    # Training examples
    training_data = [
        ([0, 0], 0, "both OFF"),
        ([0, 1], 0, "first OFF, second ON"),
        ([1, 0], 0, "first ON, second OFF"),
        ([1, 1], 1, "both ON"),
    ]

    print("STEP 2: Try each example and learn from mistakes")
    print("-" * 50)

    for round_num in range(3):  # 3 learning rounds
        print(f"\n🔄 LEARNING ROUND {round_num + 1}")
        print("=" * 30)

        total_mistakes = 0

        for example_num, (inputs, correct_output, description) in enumerate(
            training_data, 1
        ):
            print(f"\n📝 Example {example_num}: {inputs} ({description})")
            print(f"   Should output: {correct_output}")

            # STEP A: Make prediction
            weighted_sum = inputs[0] * weights[0] + inputs[1] * weights[1]
            total_with_bias = weighted_sum + bias
            prediction = 1 if total_with_bias >= 0 else 0

            print(f"   🧮 Calculation:")
            print(
                f"      ({inputs[0]} × {weights[0]:.2f}) + ({inputs[1]} × {weights[1]:.2f}) + {bias:.2f}"
            )
            print(f"      = {weighted_sum:.2f} + {bias:.2f} = {total_with_bias:.2f}")
            print(f"   🤖 My prediction: {prediction}")

            # STEP B: Check if correct
            if prediction == correct_output:
                print("   ✅ CORRECT! No need to change anything")
            else:
                print(f"   ❌ WRONG! Expected {correct_output}, got {prediction}")
                print("   🔧 Time to adjust my brain...")
                total_mistakes += 1

                # STEP C: Learn from mistake (THE MAGIC!)
                error = correct_output - prediction
                print(f"      Error amount: {error}")

                # Adjust weights using Rosenblatt's rule
                print("      Adjusting weights:")
                for i in range(len(weights)):
                    old_weight = weights[i]
                    adjustment = learning_rate * error * inputs[i]
                    weights[i] += adjustment

                    if inputs[i] != 0:  # Only show when input was involved
                        print(
                            f"         Weight[{i}]: {old_weight:.2f} + ({learning_rate} × {error} × {inputs[i]}) = {weights[i]:.2f}"
                        )
                    else:
                        print(
                            f"         Weight[{i}]: {old_weight:.2f} (no change, input was 0)"
                        )

                # Adjust bias
                old_bias = bias
                bias += learning_rate * error
                print(
                    f"         Bias: {old_bias:.2f} + ({learning_rate} × {error}) = {bias:.2f}"
                )

        print(f"\n📊 Round {round_num + 1} complete: {total_mistakes} mistakes")
        print(f"🧠 Current brain state:")
        print(f"   Weights: [{weights[0]:.2f}, {weights[1]:.2f}]")
        print(f"   Bias: {bias:.2f}")

        if total_mistakes == 0:
            print("   🎉 PERFECT! No more mistakes - learning complete!")
            break

    # Test the learned perceptron
    print(f"\n🧪 TESTING THE LEARNED PERCEPTRON")
    print("=" * 35)

    for inputs, expected, description in training_data:
        weighted_sum = inputs[0] * weights[0] + inputs[1] * weights[1]
        total_with_bias = weighted_sum + bias
        prediction = 1 if total_with_bias >= 0 else 0

        status = "✅" if prediction == expected else "❌"
        print(f"   {inputs} → {prediction} (expected {expected}) {status}")


def the_learning_rule_explained():
    """
    Break down the actual learning rule in simple terms
    """
    print("\n\n🔍 THE LEARNING RULE EXPLAINED")
    print("=" * 35)

    print("🎯 Rosenblatt's Genius Rule:")
    print("   new_weight = old_weight + (learning_rate × error × input)")
    print()

    print("Let's understand each part:")
    print()

    print("1️⃣ ERROR = correct_answer - my_prediction")
    print("   If I should say YES but said NO: error = 1 - 0 = +1")
    print("   If I should say NO but said YES: error = 0 - 1 = -1")
    print("   If I was correct: error = 0")
    print()

    print("2️⃣ INPUT = the actual input value")
    print("   If input was 0: don't blame this input (it wasn't involved)")
    print("   If input was 1: this input was involved in the decision")
    print()

    print("3️⃣ LEARNING_RATE = how much to adjust")
    print("   0.1 = make small adjustments (careful)")
    print("   0.9 = make big adjustments (aggressive)")
    print()

    print("🧠 THE LOGIC:")
    print("   If input helped make a WRONG decision:")
    print("   → Reduce its weight (trust it less)")
    print("   If input could have helped make RIGHT decision:")
    print("   → Increase its weight (trust it more)")


def why_this_works():
    """
    Explain the intuition behind why this learning rule works
    """
    print("\n\n💡 WHY DOES THIS LEARNING RULE WORK?")
    print("=" * 40)

    print("Think of learning to pick good movies:")
    print()

    print("🎬 Example: You watch a movie with [good_reviews=1, cheap_price=0]")
    print("   Current weights: [reviews=0.2, price=0.8]")
    print("   Your prediction: 0.2×1 + 0.8×0 = 0.2 → 'Don't watch' (prediction=0)")
    print("   Reality: You actually loved it! (correct=1)")
    print("   Error: 1 - 0 = +1 (you were wrong!)")
    print()

    print("🔧 Learning adjustment:")
    print("   Reviews weight: 0.2 + (0.5 × 1 × 1) = 0.7 ← INCREASED!")
    print("   Price weight: 0.8 + (0.5 × 1 × 0) = 0.8 ← No change")
    print()

    print("🧠 What happened:")
    print("   ✅ Good reviews helped you enjoy the movie, so trust them MORE")
    print("   ⚪ Price wasn't a factor (was 0), so don't change that weight")
    print()

    print("🎯 The perceptron learned:")
    print("   'I should pay more attention to reviews!'")
    print("   'Price doesn't matter as much as I thought!'")
    print()

    print("💫 After many examples, the weights settle on the BEST values!")


def common_questions():
    """
    Answer common questions about the learning process
    """
    print("\n\n❓ COMMON QUESTIONS")
    print("=" * 20)

    print("Q: Why do we multiply by the input?")
    print("A: If input=0, that input didn't participate in the wrong decision,")
    print("   so don't blame it! Only adjust weights for inputs that were 'ON'.")
    print()

    print("Q: Why start with random weights?")
    print("A: Like a baby learning - start with random guesses, then improve!")
    print("   If we started with all zeros, nothing would ever change.")
    print()

    print("Q: What if learning_rate is too big?")
    print("A: The weights might 'bounce around' and never settle down.")
    print("   Like overcorrecting when learning to drive!")
    print()

    print("Q: What if learning_rate is too small?")
    print("A: Learning will be very slow, might take forever.")
    print("   Like being too cautious when adjusting a recipe.")
    print()

    print("Q: How do we know when to stop?")
    print("A: When errors = 0! The perceptron got everything right.")


if __name__ == "__main__":
    print("🎩 WATCH THE LEARNING MAGIC HAPPEN!")
    print("Step-by-step breakdown of how perceptrons learn")
    print("=" * 55)

    # Show the learning process
    learning_step_by_step()

    # Explain the rule
    the_learning_rule_explained()

    # Explain why it works
    why_this_works()

    # Answer questions
    common_questions()

    print("\n🎯 KEY INSIGHTS:")
    print("✅ Learning = adjusting weights after each mistake")
    print("✅ The rule: new_weight = old_weight + (rate × error × input)")
    print("✅ Only blame inputs that were actually 'ON' (non-zero)")
    print("✅ Error tells us WHICH DIRECTION to adjust")
    print("✅ Learning rate controls HOW MUCH to adjust")
    print("✅ After enough examples, weights become perfect!")
    print()
    print("🚀 Ready to understand why XOR breaks this beautiful system?")
