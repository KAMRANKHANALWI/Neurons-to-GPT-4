"""
🧮 BACKPROPAGATION: THE EASIEST MATH EVER!
==========================================

Let's use the TINIEST possible network with REAL NUMBERS
so you can see EXACTLY how the math works!

Network: Input → Hidden → Output
Just 1 input, 1 hidden neuron, 1 output!
"""


def tiny_network_example():
    """
    The simplest possible network with actual numbers!
    """
    print("🔢 TINY NETWORK: 1 Input → 1 Hidden → 1 Output")
    print("=" * 50)

    print("🎯 THE SETUP:")
    print("   Input: x = 1")
    print("   Weight 1 (input→hidden): w1 = 0.5")
    print("   Weight 2 (hidden→output): w2 = 0.3")
    print("   Target output: 1.0")
    print("   Learning rate: 0.5")
    print()

    # Initial values
    x = 1
    w1 = 0.5
    w2 = 0.3
    target = 1.0
    learning_rate = 0.5

    print("📈 STEP 1: FORWARD PASS (Calculate prediction)")
    print("-" * 45)

    # Hidden neuron calculation
    hidden_input = x * w1
    hidden_output = hidden_input  # No activation function for simplicity

    print(f"   Hidden neuron:")
    print(f"      Input to hidden = x × w1 = {x} × {w1} = {hidden_input}")
    print(f"      Hidden output = {hidden_output}")
    print()

    # Output neuron calculation
    output_input = hidden_output * w2
    final_output = output_input  # No activation function for simplicity

    print(f"   Output neuron:")
    print(
        f"      Input to output = hidden × w2 = {hidden_output} × {w2} = {output_input}"
    )
    print(f"      Final output = {final_output}")
    print()

    # Calculate error
    error = target - final_output
    print(f"   📊 ERROR CALCULATION:")
    print(f"      Error = target - actual = {target} - {final_output} = {error}")
    print(f"      We're {abs(error)} points too LOW! 📉")
    print()

    return {
        "x": x,
        "w1": w1,
        "w2": w2,
        "hidden_output": hidden_output,
        "final_output": final_output,
        "error": error,
        "learning_rate": learning_rate,
    }


def backward_pass_math(data):
    """
    The backward pass with EXACT mathematical steps
    """
    print("🔙 STEP 2: BACKWARD PASS (The blame distribution!)")
    print("-" * 55)

    print("🎯 RULE: Each weight gets blame = (error from above) × (its input)")
    print()

    # Step 1: Output layer error (we already know this)
    output_error = data["error"]
    print(f"   📊 Output layer error: {output_error}")
    print(f"      'I need to be {abs(output_error)} points HIGHER!'")
    print()

    # Step 2: Update w2 (hidden → output)
    print(f"   🔧 UPDATING w2 (hidden → output weight):")
    print(f"      Current w2 = {data['w2']}")
    print(f"      w2's input was = {data['hidden_output']}")
    print(f"      w2's blame = output_error × w2's_input")
    print(
        f"      w2's blame = {output_error} × {data['hidden_output']} = {output_error * data['hidden_output']}"
    )
    print()

    w2_adjustment = data["learning_rate"] * output_error * data["hidden_output"]
    new_w2 = data["w2"] + w2_adjustment

    print(f"      📐 Rosenblatt's Formula:")
    print(f"      new_w2 = old_w2 + (learning_rate × error × input)")
    print(
        f"      new_w2 = {data['w2']} + ({data['learning_rate']} × {output_error} × {data['hidden_output']})"
    )
    print(f"      new_w2 = {data['w2']} + {w2_adjustment} = {new_w2}")
    print()

    # Step 3: Calculate hidden layer error (THE MAGIC!)
    print(f"   🧠 CALCULATING HIDDEN LAYER ERROR (The magic!):")
    print(f"      Hidden neuron asks: 'How much am I to blame?'")
    print(f"      Hidden error = output_error × w2")
    print(
        f"      Hidden error = {output_error} × {data['w2']} = {output_error * data['w2']}"
    )
    print(f"      'I'm to blame proportional to my connection strength!'")
    print()

    hidden_error = output_error * data["w2"]

    # Step 4: Update w1 (input → hidden)
    print(f"   🔧 UPDATING w1 (input → hidden weight):")
    print(f"      Current w1 = {data['w1']}")
    print(f"      w1's input was = {data['x']}")
    print(f"      w1's blame = hidden_error × w1's_input")
    print(
        f"      w1's blame = {hidden_error} × {data['x']} = {hidden_error * data['x']}"
    )
    print()

    w1_adjustment = data["learning_rate"] * hidden_error * data["x"]
    new_w1 = data["w1"] + w1_adjustment

    print(f"      📐 Rosenblatt's Formula:")
    print(f"      new_w1 = old_w1 + (learning_rate × hidden_error × input)")
    print(
        f"      new_w1 = {data['w1']} + ({data['learning_rate']} × {hidden_error} × {data['x']})"
    )
    print(f"      new_w1 = {data['w1']} + {w1_adjustment} = {new_w1}")
    print()

    return {
        "new_w1": new_w1,
        "new_w2": new_w2,
        "hidden_error": hidden_error,
        "w1_adjustment": w1_adjustment,
        "w2_adjustment": w2_adjustment,
    }


def test_improved_network(original_data, new_weights):
    """
    Test the network with the new weights to see improvement
    """
    print("🧪 STEP 3: TEST THE IMPROVED NETWORK!")
    print("-" * 40)

    print("🎯 Let's try the SAME input with our NEW weights:")
    print(f"   Input: x = {original_data['x']}")
    print(f"   NEW w1 = {new_weights['new_w1']:.3f}")
    print(f"   NEW w2 = {new_weights['new_w2']:.3f}")
    print()

    # Forward pass with new weights
    x = original_data["x"]
    new_hidden_input = x * new_weights["new_w1"]
    new_hidden_output = new_hidden_input

    new_output_input = new_hidden_output * new_weights["new_w2"]
    new_final_output = new_output_input

    print("📈 FORWARD PASS WITH NEW WEIGHTS:")
    print(f"   Hidden: {x} × {new_weights['new_w1']:.3f} = {new_hidden_output:.3f}")
    print(
        f"   Output: {new_hidden_output:.3f} × {new_weights['new_w2']:.3f} = {new_final_output:.3f}"
    )
    print()

    # Compare results
    old_output = original_data["final_output"]
    target = 1.0
    old_error = abs(target - old_output)
    new_error = abs(target - new_final_output)

    print("📊 COMPARISON:")
    print(f"   🔴 OLD output: {old_output} (error: {old_error})")
    print(f"   🟢 NEW output: {new_final_output:.3f} (error: {new_error:.3f})")
    print(f"   📈 Improvement: {old_error - new_error:.3f} points closer!")
    print()

    if new_error < old_error:
        print("✅ SUCCESS! The network got BETTER! 🎉")
        print("   Backpropagation is working!")
    else:
        print("🤔 Hmm, might need more iterations...")


def the_key_insights():
    """
    The crucial mathematical insights
    """
    print("\n💡 THE KEY MATHEMATICAL INSIGHTS!")
    print("=" * 40)

    print("1️⃣ FORWARD PASS:")
    print("   output = input × weight")
    print("   Just multiplication! 🧮")
    print()

    print("2️⃣ ERROR CALCULATION:")
    print("   error = target - actual")
    print("   How far off were we? 📏")
    print()

    print("3️⃣ BACKWARD PASS (The magic!):")
    print("   Each layer's error = (error from above) × (connection weight)")
    print("   Blame flows backward through connections! 🔙")
    print()

    print("4️⃣ WEIGHT UPDATES:")
    print("   new_weight = old_weight + (learning_rate × error × input)")
    print("   Our beloved Rosenblatt formula! ❤️")
    print()

    print("5️⃣ THE BEAUTIFUL PATTERN:")
    print("   💫 Error flows BACKWARD")
    print("   💫 Weight updates flow BACKWARD")
    print("   💫 Each layer gets its 'fair share' of blame")
    print("   💫 Network automatically improves!")


def bigger_network_preview():
    """
    Show how this scales to bigger networks
    """
    print("\n\n🚀 HOW THIS SCALES TO BIGGER NETWORKS:")
    print("=" * 45)

    print("For a bigger network like XOR:")
    print()
    print("🏗️ NETWORK STRUCTURE:")
    print("   Input Layer:  [x1, x2]")
    print("   Hidden Layer: [h1, h2] ")
    print("   Output Layer: [y]")
    print()

    print("📊 SAME MATH, MORE NEURONS:")
    print("   1. Forward pass: Calculate all neurons")
    print("   2. Output error: target - actual")
    print("   3. Hidden errors: output_error × their_weights")
    print("   4. Update ALL weights using Rosenblatt's formula")
    print()

    print("🎯 THE BEAUTY:")
    print("   Each neuron gets blamed proportional to its influence!")
    print("   Strong connections → more blame → bigger changes")
    print("   Weak connections → less blame → smaller changes")
    print()
    print("   The network automatically figures out:")
    print("   'Which connections are helping?'")
    print("   'Which connections are hurting?'")
    print("   'How much should each one change?'")


def simple_xor_math():
    """
    Show XOR numbers briefly
    """
    print("\n\n🔶 XOR EXAMPLE (SIMPLIFIED NUMBERS):")
    print("=" * 40)

    print("Input [0,1] should give 1, but network gives 0.2")
    print("Error = 1 - 0.2 = 0.8")
    print()
    print("Backward flow:")
    print("   Output neuron: 'I need to be 0.8 higher!'")
    print("   Hidden neuron 1: 'My error = 0.8 × my_weight_to_output'")
    print("   Hidden neuron 2: 'My error = 0.8 × my_weight_to_output'")
    print("   Input weights: 'Adjust us based on our blame!'")
    print()
    print("After many iterations:")
    print("   Network discovers the XOR pattern automatically! 🎉")


if __name__ == "__main__":
    print("🧮 BACKPROPAGATION: THE EASIEST MATH BREAKDOWN!")
    print("=" * 55)
    print("Watch the numbers flow backward through the network!")
    print()

    # Run the tiny network example
    data = tiny_network_example()

    # Show the backward pass math
    new_weights = backward_pass_math(data)

    # Test improvement
    test_improved_network(data, new_weights)

    # Key insights
    the_key_insights()

    # Bigger network preview
    bigger_network_preview()

    # XOR example
    simple_xor_math()

    print("\n🌟 NOW YOU UNDERSTAND THE MATH!")
    print("It's just:")
    print("   1. Multiply and add (forward)")
    print("   2. Subtract to get error")
    print("   3. Multiply error backward")
    print("   4. Update weights with Rosenblatt's formula")
    print("   5. Repeat until perfect! 🎯")
