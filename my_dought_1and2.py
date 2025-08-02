"""
🤔 CLARIFYING YOUR PERFECT QUESTIONS!
====================================

You asked TWO excellent questions that show deep understanding!
Let me address each one clearly...
"""


def question_1_weighted_sum_process():
    """
    Question 1: About the weighted sum and prediction process
    """
    print("❓ QUESTION 1: The Weighted Sum Process")
    print("=" * 45)

    print("You said:")
    print("'weighted sum = input1×weight1 + input2×weight2 + bias'")
    print("'then compare to threshold to predict active/inactive'")
    print("'this is our old friend from chapter 1'")
    print()
    print("✅ YOU'RE ABSOLUTELY RIGHT!")
    print()

    print("🔄 The COMPLETE process:")
    print()

    print("STEP 1: Calculate weighted sum (McCulloch-Pitts style)")
    print("   weighted_sum = input₁×weight₁ + input₂×weight₂ + bias")
    print()

    print("STEP 2: Apply threshold (activation function)")
    print("   if weighted_sum >= 0:")
    print("       prediction = 1  (ACTIVE/ON)")
    print("   else:")
    print("       prediction = 0  (INACTIVE/OFF)")
    print()

    print("STEP 3: If wrong, use Rosenblatt's formula")
    print("   new_weight = old_weight + (learning_rate × error × input)")
    print()

    print("STEP 4: Repeat until error = 0")
    print()

    print("💡 So yes! It's:")
    print("   McCulloch-Pitts computation + Rosenblatt's learning!")


def question_2_where_correct_values_come_from():
    """
    Question 2: Where do the correct values come from?
    """
    print("\n\n❓ QUESTION 2: Where Do Correct Values Come From?")
    print("=" * 55)

    print("You asked:")
    print("'How do we know the correct value to calculate error?'")
    print("'We must already have that correct value, right?'")
    print()
    print("✅ YOU'RE ABSOLUTELY RIGHT AGAIN!")
    print()

    print("🎓 This is called SUPERVISED LEARNING!")
    print()

    print("📚 Think of it like learning math in school:")
    print("   Teacher gives you: '2 + 3 = ?'")
    print("   You answer: '6'")
    print("   Teacher says: 'Wrong! The correct answer is 5'")
    print("   You learn from the mistake!")
    print()

    print("🤖 For perceptrons:")
    print("   We give examples: 'For input [1,1], output should be 1'")
    print("   Perceptron predicts: '0'")
    print("   We tell it: 'Wrong! Should be 1'")
    print("   Perceptron learns from mistake!")


def training_data_examples():
    """
    Show examples of training data
    """
    print("\n\n📊 TRAINING DATA EXAMPLES")
    print("=" * 30)

    print("🎯 AND Gate Training Data:")
    print("   Input [0,0] → Correct Output: 0")
    print("   Input [0,1] → Correct Output: 0")
    print("   Input [1,0] → Correct Output: 0")
    print("   Input [1,1] → Correct Output: 1")
    print()
    print("💡 We KNOW these are correct because we designed the AND gate!")
    print()

    print("🍕 Restaurant Training Data:")
    print("   [good_food=1, cheap=0] → You were happy: 1")
    print("   [good_food=0, cheap=1] → You were sad: 0")
    print("   [good_food=1, cheap=1] → You were very happy: 1")
    print()
    print("💡 We know these because you EXPERIENCED them!")


def the_learning_cycle():
    """
    Show the complete learning cycle
    """
    print("\n\n🔄 THE COMPLETE LEARNING CYCLE")
    print("=" * 35)

    print("Here's how it all works together:")
    print()

    print("🎯 TRAINING PHASE:")
    print("   1. We have TRAINING DATA (inputs + correct outputs)")
    print("   2. Perceptron makes prediction using weighted sum")
    print("   3. We compare prediction to correct answer")
    print("   4. If wrong, use Rosenblatt's formula to adjust weights")
    print("   5. Repeat until all predictions are correct!")
    print()

    print("🧪 TESTING PHASE:")
    print("   1. Give perceptron NEW data (without correct answers)")
    print("   2. It makes predictions using learned weights")
    print("   3. Hope it generalizes well to new examples!")


def step_by_step_example():
    """
    Complete step-by-step example showing both concepts
    """
    print("\n\n📝 COMPLETE EXAMPLE: Learning AND Gate")
    print("=" * 45)

    print("🎯 GIVEN TRAINING DATA:")
    training_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]

    for inputs, correct in training_data:
        print(f"   {inputs} → should output {correct}")
    print()

    print("🎲 INITIAL STATE:")
    weights = [0.2, 0.3]
    bias = 0.1
    learning_rate = 0.5

    print(f"   weights = {weights}")
    print(f"   bias = {bias}")
    print(f"   learning_rate = {learning_rate}")
    print()

    print("🧪 TRAINING ROUND 1:")
    print()

    for i, (inputs, correct_output) in enumerate(training_data, 1):
        print(f"   Example {i}: {inputs} should output {correct_output}")

        # Step 1: Calculate weighted sum (McCulloch-Pitts)
        weighted_sum = inputs[0] * weights[0] + inputs[1] * weights[1] + bias
        print(
            f"      Weighted sum = {inputs[0]}×{weights[0]} + {inputs[1]}×{weights[1]} + {bias} = {weighted_sum}"
        )

        # Step 2: Make prediction (threshold)
        prediction = 1 if weighted_sum >= 0 else 0
        print(
            f"      Prediction = {prediction} (since {weighted_sum} >= 0: {weighted_sum >= 0})"
        )

        # Step 3: Calculate error
        error = correct_output - prediction
        print(f"      Error = {correct_output} - {prediction} = {error}")

        # Step 4: Learn (if wrong)
        if error != 0:
            print(f"      ❌ WRONG! Applying Rosenblatt's formula:")

            for j in range(len(weights)):
                old_weight = weights[j]
                adjustment = learning_rate * error * inputs[j]
                new_weight = old_weight + adjustment
                weights[j] = new_weight

                if inputs[j] != 0:
                    print(
                        f"         Weight[{j}]: {old_weight} + ({learning_rate}×{error}×{inputs[j]}) = {new_weight}"
                    )
                else:
                    print(
                        f"         Weight[{j}]: {old_weight} (no change, input was 0)"
                    )

            old_bias = bias
            bias += learning_rate * error
            print(f"         Bias: {old_bias} + ({learning_rate}×{error}) = {bias}")
        else:
            print(f"      ✅ CORRECT! No learning needed")

        print()

    print(f"🧠 UPDATED STATE:")
    print(f"   weights = {weights}")
    print(f"   bias = {bias:.1f}")


def key_insights():
    """
    Summarize the key insights
    """
    print("\n\n🎯 KEY INSIGHTS")
    print("=" * 20)

    print("✅ QUESTION 1 ANSWER:")
    print("   Yes! Weighted sum → threshold → prediction (McCulloch-Pitts)")
    print("   Then if wrong → adjust weights (Rosenblatt)")
    print("   It's both concepts working together!")
    print()

    print("✅ QUESTION 2 ANSWER:")
    print("   Yes! We MUST have correct answers for training")
    print("   This is called 'supervised learning'")
    print("   Like a teacher providing the answer key!")
    print()

    print("🔄 THE FULL CYCLE:")
    print("   1. Have training data (inputs + correct outputs)")
    print("   2. Make prediction using weighted sum")
    print("   3. Compare to correct answer → calculate error")
    print("   4. If error ≠ 0 → adjust weights using Rosenblatt's formula")
    print("   5. Repeat until error = 0 for all examples!")
    print()

    print("💡 YOU UNDERSTAND THE COMPLETE PICTURE!")
    print("   McCulloch-Pitts: HOW to compute")
    print("   Rosenblatt: HOW to learn")
    print("   Together: A learning machine!")


if __name__ == "__main__":
    print("🤔 ANSWERING YOUR EXCELLENT QUESTIONS!")
    print("=" * 50)
    print("You're connecting all the concepts perfectly!")
    print()

    # Answer question 1
    question_1_weighted_sum_process()

    # Answer question 2
    question_2_where_correct_values_come_from()

    # Show training data examples
    training_data_examples()

    # Show the learning cycle
    the_learning_cycle()

    # Complete example
    step_by_step_example()

    # Key insights
    key_insights()

    print("\n🚀 YOU'VE GOT IT!")
    print("Ready to see why XOR breaks this beautiful system?")
