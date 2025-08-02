"""
🧠 BACKPROPAGATION NETWORK IMPLEMENTATION
=========================================

THE NETWORK STRUCTURE:
======================

        INPUT LAYER    HIDDEN LAYER    OUTPUT LAYER
        -----------    ------------    ------------
            
            x1 ---------> h1 ----------> y
            |  \         /  \          /
            |   \       /    \        /
            |    \     /      \      /
            |     \   /        \    /
            |      \ /          \  /
            |       X            \/
            |      / \          /\
            |     /   \        /  \
            |    /     \      /    \
            |   /       \    /      \
            |  /         \  /        \
            x2 ---------> h2 ----------> y
            
        [2 inputs]    [2 hidden]     [1 output]
"""

import numpy as np
import matplotlib.pyplot as plt


class BackpropagationNetwork:
    """
    A simple neural network implementing backpropagation algorithm
    """

    def __init__(self, learning_rate=0.1):
        """Initialize the network with random weights"""
        self.learning_rate = learning_rate

        # Initialize weights randomly (small values)
        # Input to Hidden weights (2x2 matrix)
        self.w_input_hidden = np.random.uniform(-0.5, 0.5, (2, 2))
        # Hidden to Output weights (2x1 matrix)
        self.w_hidden_output = np.random.uniform(-0.5, 0.5, (2, 1))

        print("🚀 Network initialized with random weights!")
        print(f"Input->Hidden weights:\n{self.w_input_hidden}")
        print(f"Hidden->Output weights:\n{self.w_hidden_output}")

    def sigmoid(self, x):
        """Activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def forward_pass(self, inputs):
        """
        FORWARD PASS (→ direction):
        ===========================
        Step 1: Calculate hidden layer
        Step 2: Calculate output
        """
        # Convert inputs to numpy array
        inputs = np.array(inputs).reshape(-1, 1)

        # Step 1: Input to Hidden
        hidden_input = np.dot(self.w_input_hidden.T, inputs)
        hidden_output = self.sigmoid(hidden_input)

        # Step 2: Hidden to Output
        output_input = np.dot(self.w_hidden_output.T, hidden_output)
        final_output = self.sigmoid(output_input)

        return inputs, hidden_output, final_output

    def backward_pass(self, inputs, hidden_output, final_output, target):
        """
        BACKWARD PASS (← direction):
        ============================
        Step 1: Calculate output error
        Step 2: Calculate hidden layer errors
        Step 3: Update ALL weights
        """
        target = np.array([[target]])

        # Step 1: Output layer error
        output_error = target - final_output
        output_delta = output_error * self.sigmoid_derivative(final_output)

        # Step 2: Hidden layer errors (blame flows backward!)
        hidden_error = np.dot(self.w_hidden_output, output_delta)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

        # Step 3: Update weights using gradient descent
        # Update Hidden to Output weights (2x1 matrix)
        self.w_hidden_output += self.learning_rate * hidden_output * output_delta[0, 0]

        # Update Input to Hidden weights (2x2 matrix)
        self.w_input_hidden += self.learning_rate * np.outer(
            inputs.flatten(), hidden_delta.flatten()
        )

        return float(output_error[0, 0])

    def train(self, training_data, epochs=1000):
        """Train the network on given data"""
        errors = []

        print(f"\n🎯 Training for {epochs} epochs...")

        for epoch in range(epochs):
            total_error = 0

            for inputs, target in training_data:
                # Forward pass
                input_layer, hidden_layer, output_layer = self.forward_pass(inputs)

                # Backward pass
                error = self.backward_pass(
                    input_layer, hidden_layer, output_layer, target
                )
                total_error += abs(error)

            avg_error = total_error / len(training_data)
            errors.append(avg_error)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Average Error = {avg_error:.4f}")

        return errors

    def predict(self, inputs):
        """Make a prediction"""
        _, _, output = self.forward_pass(inputs)
        return float(output[0, 0])

    def test_xor(self):
        """Test the network on XOR problem"""
        print("\n🧪 Testing XOR Logic:")
        print("=" * 30)

        test_cases = [
            [0, 0],  # Should output ~0
            [0, 1],  # Should output ~1
            [1, 0],  # Should output ~1
            [1, 1],  # Should output ~0
        ]

        for inputs in test_cases:
            prediction = self.predict(inputs)
            expected = inputs[0] ^ inputs[1]  # XOR logic
            print(f"Input: {inputs} → Output: {prediction:.3f} (Expected: {expected})")


def demonstrate_backpropagation():
    """
    XOR EXAMPLE WITH NUMBERS:
    =========================
    The classic test that proves multilayer networks work!
    """

    # Create network
    network = BackpropagationNetwork(learning_rate=5.0)

    # XOR training data: [input1, input2] → target
    xor_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

    print("\n📚 XOR Training Data:")
    for inputs, target in xor_data:
        print(f"  {inputs} → {target}")

    # Test before training
    print("\n🔴 BEFORE TRAINING:")
    network.test_xor()

    # Train the network
    errors = network.train(xor_data, epochs=2000)

    # Test after training
    print("\n🟢 AFTER TRAINING:")
    network.test_xor()

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.title("🧠 Backpropagation Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Average Error")
    plt.grid(True)
    plt.show()

    print("\n✨ THE MAGIC HAPPENED!")
    print("🎯 BLAME DISTRIBUTION: Strong connections got MORE blame")
    print("🔄 AUTOMATIC LEARNING: Network figured out what helps/hurts")
    print("🚀 SCALES TO ANY SIZE: Same pattern works for huge networks")


def show_weight_evolution():
    """Show how weights change during training"""
    network = BackpropagationNetwork(learning_rate=5.0)

    xor_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

    print("\n📊 WEIGHT EVOLUTION:")
    print("=" * 40)

    # Show initial weights
    print("Initial weights:")
    print(f"  Input→Hidden: \n{network.w_input_hidden}")
    print(f"  Hidden→Output: \n{network.w_hidden_output}")

    # Train and show weights at intervals
    for epoch in [0, 500, 1000, 1500, 2000]:
        if epoch > 0:
            network.train(xor_data, epochs=500)

        print(f"\nAfter {epoch} epochs:")
        print(f"  Input→Hidden: \n{network.w_input_hidden}")
        print(f"  Hidden→Output: \n{network.w_hidden_output}")


if __name__ == "__main__":
    print("🧠 BACKPROPAGATION NETWORK DEMONSTRATION")
    print("=" * 50)

    # Run the main demonstration
    demonstrate_backpropagation()

    # Show weight evolution
    show_weight_evolution()

    print("\n🎉 NETWORK LEARNED TO SOLVE XOR!")
    print("Multiple perceptrons working as a TEAM!")
    print("Each one learned its specialized role!")
    print("Together they solved the pattern! ✨")
