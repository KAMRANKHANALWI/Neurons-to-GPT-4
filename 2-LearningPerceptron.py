"""
🚀 CHAPTER 2: THE PERCEPTRON (1957) - The First Learning Machine!
================================================================

Historical Context:
- Frank Rosenblatt at Cornell Aeronautical Laboratory
- 1957: First machine that could LEARN from experience!
- New York Times headline: "New Navy Device Learns By Doing"
- The birth of machine learning!

The Revolutionary Breakthrough:
❌ McCulloch-Pitts: Weights are FIXED forever
✅ Perceptron: Weights can CHANGE and IMPROVE!

The Magic Question Rosenblatt Asked:
"What if the neuron could adjust its 'importance levels' (weights)
based on whether it makes correct or wrong decisions?"

This is like learning from your mistakes! 🎯
"""

import numpy as np
import matplotlib.pyplot as plt


class LearningPerceptron:
    """
    The revolutionary Perceptron that can LEARN!

    The breakthrough: Instead of fixed weights, it can:
    1. Make a prediction
    2. See if it was right or wrong
    3. Adjust its weights to do better next time!
    """

    def __init__(self, learning_rate=0.1):
        """
        Initialize a learning perceptron

        Args:
            learning_rate: How fast it learns (0.1 = learns slowly but safely)
        """
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.errors_history = []

    def predict(self, inputs):
        """
        Make a prediction (same as McCulloch-Pitts)
        """
        # Calculate weighted sum + bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        # Apply step function (threshold at 0)
        return 1 if weighted_sum >= 0 else 0

    def learn_from_mistake(self, inputs, expected_output, actual_output):
        """
        🎯 THE MAGIC LEARNING RULE!

        This is Rosenblatt's breakthrough algorithm:
        - If prediction is CORRECT → don't change anything
        - If prediction is WRONG → adjust weights in right direction
        """
        error = expected_output - actual_output

        if error != 0:  # Only learn if we made a mistake!
            print(f"   ❌ MISTAKE! Expected {expected_output}, got {actual_output}")
            print(f"   🔧 Adjusting weights...")

            # Rosenblatt's learning rule:
            # If we should have said YES but said NO → increase weights
            # If we should have said NO but said YES → decrease weights
            for i in range(len(self.weights)):
                old_weight = self.weights[i]
                self.weights[i] += self.learning_rate * error * inputs[i]
                print(f"      Weight[{i}]: {old_weight:.2f} → {self.weights[i]:.2f}")

            # Also adjust bias
            old_bias = self.bias
            self.bias += self.learning_rate * error
            print(f"      Bias: {old_bias:.2f} → {self.bias:.2f}")
        else:
            print(f"   ✅ CORRECT! Expected {expected_output}, got {actual_output}")

    def train(self, training_inputs, training_outputs, epochs=10):
        """
        Train the perceptron on examples

        Args:
            training_inputs: Examples to learn from
            training_outputs: Correct answers
            epochs: How many times to go through all examples
        """
        # Initialize random weights (small values)
        num_inputs = len(training_inputs[0])
        self.weights = np.random.normal(0, 0.1, num_inputs)
        self.bias = np.random.normal(0, 0.1)

        print(f"🎯 Starting training with random weights: {self.weights}")
        print(f"   Initial bias: {self.bias:.2f}")
        print("=" * 60)

        for epoch in range(epochs):
            print(f"\n📚 EPOCH {epoch + 1}:")
            total_errors = 0

            for i, (inputs, expected) in enumerate(
                zip(training_inputs, training_outputs)
            ):
                print(f"\n   Example {i + 1}: inputs={inputs}, expected={expected}")

                # Make prediction
                prediction = self.predict(inputs)

                # Learn from result
                self.learn_from_mistake(inputs, expected, prediction)

                # Track errors
                if prediction != expected:
                    total_errors += 1

            self.errors_history.append(total_errors)
            print(f"\n   📊 Epoch {epoch + 1} complete: {total_errors} errors")

            # Stop if perfect!
            if total_errors == 0:
                print(f"   🎉 PERFECT! Learned everything in {epoch + 1} epochs!")
                break

        print(f"\n🎯 Final weights: {self.weights}")
        print(f"   Final bias: {self.bias:.2f}")


def demonstrate_and_gate_learning():
    """
    Let's watch the perceptron LEARN the AND gate!
    This is the same problem McCulloch-Pitts could solve,
    but now the perceptron figures out the weights BY ITSELF!
    """
    print("🧠 WATCHING THE PERCEPTRON LEARN: AND Gate")
    print("=" * 50)

    print("Teaching the perceptron the AND gate:")
    print("   Input [0,0] should output 0")
    print("   Input [0,1] should output 0")
    print("   Input [1,0] should output 0")
    print("   Input [1,1] should output 1")
    print("\nLet's see if it can figure out the pattern...")

    # Training data for AND gate
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])

    # Create and train perceptron
    perceptron = LearningPerceptron(learning_rate=0.5)
    perceptron.train(X_train, y_train, epochs=10)

    # Test the learned perceptron
    print("\n🧪 TESTING THE LEARNED PERCEPTRON:")
    print("=" * 40)
    for inputs, expected in zip(X_train, y_train):
        prediction = perceptron.predict(inputs)
        status = "✅" if prediction == expected else "❌"
        print(f"   {inputs} → {prediction} (expected {expected}) {status}")

    return perceptron


def demonstrate_or_gate_learning():
    """
    Now let's teach it the OR gate - different pattern!
    """
    print("\n\n🧠 TEACHING A NEW PATTERN: OR Gate")
    print("=" * 45)

    print("Teaching the perceptron the OR gate:")
    print("   Input [0,0] should output 0")
    print("   Input [0,1] should output 1")
    print("   Input [1,0] should output 1")
    print("   Input [1,1] should output 1")

    # Training data for OR gate
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 1])

    # Create and train perceptron
    perceptron = LearningPerceptron(learning_rate=0.5)
    perceptron.train(X_train, y_train, epochs=10)

    # Test the learned perceptron
    print("\n🧪 TESTING THE LEARNED OR GATE:")
    print("=" * 35)
    for inputs, expected in zip(X_train, y_train):
        prediction = perceptron.predict(inputs)
        status = "✅" if prediction == expected else "❌"
        print(f"   {inputs} → {prediction} (expected {expected}) {status}")

    return perceptron


def the_famous_limitation():
    """
    The limitation that stumped researchers for years!
    """
    print("\n\n⚠️  THE FAMOUS XOR PROBLEM")
    print("=" * 35)

    print("Let's try to teach the XOR (exclusive OR) pattern:")
    print("   Input [0,0] should output 0")
    print("   Input [0,1] should output 1")
    print("   Input [1,0] should output 1")
    print("   Input [1,1] should output 0  ← This is the tricky part!")
    print("\nThis means: output 1 if inputs are DIFFERENT")

    # Training data for XOR gate
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])

    # Try to train perceptron
    perceptron = LearningPerceptron(learning_rate=0.5)
    perceptron.train(X_train, y_train, epochs=20)

    # Test the result
    print("\n🧪 TESTING XOR LEARNING:")
    print("=" * 25)
    correct = 0
    for inputs, expected in zip(X_train, y_train):
        prediction = perceptron.predict(inputs)
        status = "✅" if prediction == expected else "❌"
        if prediction == expected:
            correct += 1
        print(f"   {inputs} → {prediction} (expected {expected}) {status}")

    print(f"\n📊 Accuracy: {correct}/4 = {correct/4*100:.0f}%")

    if correct < 4:
        print("\n❌ THE PERCEPTRON FAILED TO LEARN XOR!")
        print("This limitation led to the 'AI Winter' of the 1970s...")
        print("But it also led to the next breakthrough: MULTI-LAYER NETWORKS!")


def real_world_example():
    """
    A more realistic example - learning to recognize patterns
    """
    print("\n\n🌟 REAL-WORLD EXAMPLE: Weather Decision")
    print("=" * 45)

    print("Teaching perceptron when to go outside based on weather:")
    print("Features: [sunny, warm, windy]")
    print("Goal: Learn when conditions are good for going outside")

    # More realistic training data
    # [sunny, warm, windy] → go_outside
    weather_data = np.array(
        [
            [1, 1, 0],  # sunny, warm, not windy → YES
            [1, 0, 0],  # sunny, cold, not windy → NO
            [0, 1, 0],  # cloudy, warm, not windy → NO
            [1, 1, 1],  # sunny, warm, windy → NO (too windy!)
            [0, 0, 0],  # cloudy, cold, not windy → NO
            [1, 0, 1],  # sunny, cold, windy → NO
            [0, 1, 1],  # cloudy, warm, windy → NO
            [0, 0, 1],  # cloudy, cold, windy → NO
        ]
    )

    weather_labels = np.array([1, 0, 0, 0, 0, 0, 0, 0])

    # Train the perceptron
    weather_perceptron = LearningPerceptron(learning_rate=0.3)
    weather_perceptron.train(weather_data, weather_labels, epochs=15)

    # Test on new weather conditions
    print("\n🧪 TESTING ON NEW WEATHER CONDITIONS:")
    print("=" * 40)

    test_conditions = [
        ([1, 1, 0], "Perfect day: sunny, warm, no wind"),
        ([0, 0, 1], "Terrible day: cloudy, cold, windy"),
        ([1, 0, 0], "Sunny but cold"),
    ]

    for conditions, description in test_conditions:
        prediction = weather_perceptron.predict(conditions)
        decision = "Go outside! ☀️" if prediction == 1 else "Stay inside! 🏠"
        print(f"   {conditions} ({description}) → {decision}")


def plot_learning_progress(perceptron):
    """
    Visualize how the perceptron's errors decrease over time
    """
    if len(perceptron.errors_history) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(perceptron.errors_history) + 1),
            perceptron.errors_history,
            "bo-",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Number of Errors")
        plt.title("Perceptron Learning Progress")
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    print("🚀 THE PERCEPTRON: First Machine That Could LEARN!")
    print("=" * 55)
    print("Watch as the perceptron figures out patterns BY ITSELF!")
    print()

    # Demonstrate learning AND gate
    and_perceptron = demonstrate_and_gate_learning()

    # Demonstrate learning OR gate
    or_perceptron = demonstrate_or_gate_learning()

    # Show the famous limitation
    the_famous_limitation()

    # Real-world example
    real_world_example()

    print("\n🎯 KEY BREAKTHROUGHS:")
    print("1. ✅ First machine that could LEARN from examples!")
    print("2. ✅ Automatically finds the right weights!")
    print("3. ✅ Gets better with practice!")
    print("4. ❌ BUT... cannot learn non-linear patterns (like XOR)")
    print()
    print("🔜 NEXT: Multi-Layer Perceptrons - Breaking the Linear Barrier!")
    print("(Finally solving the XOR problem that stumped everyone!)")
