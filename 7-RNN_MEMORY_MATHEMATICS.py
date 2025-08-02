"""
🧠 RNN MEMORY MATHEMATICS: The Complete Mechanism
=================================================

Now you'll understand EXACTLY how RNNs manage memory mathematically!
We'll break down every equation, every step, with clear examples.

This is where the magic of sequence learning happens! ✨
"""

import numpy as np
import matplotlib.pyplot as plt


def rnn_core_equations():
    """
    Start with the fundamental RNN equations
    """
    print("🧮 THE CORE RNN EQUATIONS")
    print("=" * 30)

    print("🎯 THE MATHEMATICAL HEART OF RNNs:")
    print()
    print("   h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print("   y_t = W_hy × h_t + b_y")
    print()

    print("🔍 BREAKING DOWN EACH SYMBOL:")
    print("   h_t     = hidden state at time t (THE MEMORY)")
    print("   h_{t-1} = hidden state at previous time (PREVIOUS MEMORY)")
    print("   x_t     = input at time t (CURRENT INPUT)")
    print("   y_t     = output at time t (CURRENT OUTPUT)")
    print()

    print("   W_hh = hidden-to-hidden weights (HOW MUCH TO REMEMBER)")
    print("   W_xh = input-to-hidden weights (HOW MUCH INPUT MATTERS)")
    print("   W_hy = hidden-to-output weights (HOW TO USE MEMORY)")
    print("   b_h  = hidden bias (MEMORY ADJUSTMENT)")
    print("   b_y  = output bias (OUTPUT ADJUSTMENT)")
    print()

    print("💡 THE MEMORY MAGIC:")
    print("   The hidden state h_t IS the memory!")
    print("   It combines: OLD MEMORY + NEW INPUT → NEW MEMORY")
    print("   Then uses this memory to make predictions!")


def memory_update_step_by_step():
    """
    Show exactly how memory gets updated step by step
    """
    print("\n\n👣 MEMORY UPDATE: Step-by-Step Breakdown")
    print("=" * 50)

    print("🎯 THE MEMORY UPDATE EQUATION:")
    print("   h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print()

    print("📊 STEP-BY-STEP PROCESS:")
    print()

    print("STEP 1: Take previous memory")
    print("   h_{t-1} = what we remembered from before")
    print("   Example: h_{t-1} = [0.3, -0.1, 0.7]")
    print()

    print("STEP 2: Process previous memory")
    print("   memory_contribution = W_hh × h_{t-1}")
    print("   W_hh decides 'how much of old memory to keep'")
    print("   Example: If W_hh = [[0.5, 0.2], [0.1, 0.8]]")
    print("           memory_contribution = [calculated values]")
    print()

    print("STEP 3: Take current input")
    print("   x_t = new information coming in")
    print("   Example: x_t = [1, 0] (word 'love')")
    print()

    print("STEP 4: Process current input")
    print("   input_contribution = W_xh × x_t")
    print("   W_xh decides 'how much new input matters'")
    print("   Example: input_contribution = [calculated values]")
    print()

    print("STEP 5: Combine everything")
    print("   combined = memory_contribution + input_contribution + b_h")
    print("   This is the 'raw new memory'")
    print()

    print("STEP 6: Apply activation function")
    print("   h_t = tanh(combined)")
    print("   tanh keeps values between -1 and +1")
    print("   This is the NEW MEMORY that gets passed forward!")


def concrete_numerical_example():
    """
    Work through a complete numerical example
    """
    print("\n\n🔢 CONCRETE NUMERICAL EXAMPLE")
    print("=" * 40)

    print("🎯 SCENARIO: Processing sentence 'I love cats'")
    print("   Vocabulary: {'I':0, 'love':1, 'cats':2}")
    print("   One-hot encoding: I=[1,0,0], love=[0,1,0], cats=[0,0,1]")
    print()

    print("🏗️ NETWORK SETUP:")
    print("   Input size: 3 (vocabulary size)")
    print("   Hidden size: 2 (memory size)")
    print("   Output size: 1 (sentiment: positive/negative)")
    print()

    # Initialize weights (small values for simplicity)
    W_xh = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # input to hidden

    W_hh = np.array([[0.7, 0.1], [0.2, 0.8]])  # hidden to hidden (memory)

    W_hy = np.array([[0.9, 0.3]])  # hidden to output

    b_h = np.array([0.1, -0.1])  # hidden bias
    b_y = np.array([0.2])  # output bias

    print("🔢 WEIGHT MATRICES:")
    print(f"   W_xh (input→hidden):\n{W_xh}")
    print(f"   W_hh (hidden→hidden):\n{W_hh}")
    print(f"   W_hy (hidden→output):\n{W_hy}")
    print(f"   b_h (hidden bias): {b_h}")
    print(f"   b_y (output bias): {b_y}")
    print()

    # Process each word
    words = [
        ("I", np.array([1, 0, 0])),
        ("love", np.array([0, 1, 0])),
        ("cats", np.array([0, 0, 1])),
    ]

    # Initial memory (zeros)
    h_prev = np.array([0.0, 0.0])

    print("🔄 PROCESSING EACH WORD:")
    print()

    for t, (word, x_t) in enumerate(words):
        print(f"TIME STEP {t+1}: '{word}'")
        print(f"   Input: {x_t}")
        print(f"   Previous memory: {h_prev}")

        # Step 1: Process previous memory
        memory_part = np.dot(W_hh, h_prev)
        print(f"   Memory contribution: W_hh × h_prev = {memory_part}")

        # Step 2: Process current input
        input_part = np.dot(W_xh, x_t)
        print(f"   Input contribution: W_xh × x_t = {input_part}")

        # Step 3: Combine
        combined = memory_part + input_part + b_h
        print(f"   Combined: {combined}")

        # Step 4: Apply activation
        h_current = np.tanh(combined)
        print(f"   New memory: tanh(combined) = {h_current}")

        # Step 5: Generate output
        output = np.dot(W_hy, h_current) + b_y
        print(f"   Output: W_hy × h_current + b_y = {output}")

        # Interpret output
        sentiment = "POSITIVE" if output[0] > 0 else "NEGATIVE"
        print(f"   Sentiment: {sentiment}")
        print()

        # Update for next time step
        h_prev = h_current

    print("🎉 FINAL RESULT:")
    print(f"   Final memory state: {h_prev}")
    print(f"   Final sentiment: {sentiment}")
    print("   The RNN built up understanding word by word!")


def memory_visualization():
    """
    Visualize how memory evolves over time
    """
    print("\n\n📊 MEMORY EVOLUTION VISUALIZATION")
    print("=" * 40)

    print("🎯 Let's trace how memory changes through the sentence:")
    print("   'The cat is very happy today'")
    print()

    # Simplified memory evolution (made-up values for illustration)
    words = ["The", "cat", "is", "very", "happy", "today"]
    memory_states = [
        [0.0, 0.0],  # Initial
        [0.1, 0.3],  # After "The"
        [0.4, 0.2],  # After "cat"
        [0.3, 0.4],  # After "is"
        [0.2, 0.6],  # After "very"
        [0.8, 0.7],  # After "happy"
        [0.9, 0.8],  # After "today"
    ]

    print("   Time | Word    | Memory State    | Interpretation")
    print("   -----|---------|-----------------|------------------")
    print("     0  | <start> | [0.0,  0.0]     | No context yet")

    for i, (word, memory) in enumerate(zip(words, memory_states[1:]), 1):
        interpretation = ""
        if word == "The":
            interpretation = "Article detected"
        elif word == "cat":
            interpretation = "Subject identified"
        elif word == "is":
            interpretation = "Verb linking"
        elif word == "very":
            interpretation = "Intensifier noted"
        elif word == "happy":
            interpretation = "POSITIVE emotion!"
        elif word == "today":
            interpretation = "Time context"

        print(
            f"     {i}  | {word:<7} | [{memory[0]:.1f}, {memory[1]:.1f}]     | {interpretation}"
        )

    print()
    print("👁️ OBSERVE THE PATTERN:")
    print("   Memory values change as context builds up")
    print("   'happy' causes a big jump → strong positive signal")
    print("   Each word modifies the overall understanding")


def the_gradient_flow():
    """
    Explain how gradients flow through RNN memory
    """
    print("\n\n🌊 GRADIENT FLOW THROUGH MEMORY")
    print("=" * 40)

    print("🎯 BACKPROPAGATION THROUGH TIME (BPTT)")
    print()

    print("📐 THE CHALLENGE:")
    print("   Regular backprop: Error flows through layers")
    print("   RNN backprop: Error flows through TIME STEPS")
    print("   Each time step uses the SAME weights!")
    print()

    print("🔄 THE UNFOLDED VIEW:")
    print("   Time 1: x₁ → [RNN] → h₁ → y₁")
    print("   Time 2: x₂ → [RNN] → h₂ → y₂")
    print("   Time 3: x₃ → [RNN] → h₃ → y₃")
    print("                  ↑      ↑")
    print("   Same weights: W_hh, W_xh, W_hy")
    print()

    print("⛓️ CHAIN RULE THROUGH TIME:")
    print("   ∂Error/∂W_hh = Σ(∂Error/∂h_t × ∂h_t/∂W_hh) for all t")
    print()
    print("   But ∂h_t/∂W_hh depends on ∂h_{t-1}/∂W_hh!")
    print("   So gradient chains through all previous time steps!")
    print()

    print("🌊 THE GRADIENT FLOW:")
    print("   Error at time T flows backward:")
    print("   T → T-1 → T-2 → ... → 1")
    print("   Each step multiplies by W_hh")
    print()

    print("⚠️ THE VANISHING GRADIENT PROBLEM:")
    print("   If |W_hh| < 1: gradients shrink exponentially")
    print("   If |W_hh| > 1: gradients explode exponentially")
    print("   Result: Hard to learn long-term dependencies!")


def vanishing_gradient_mathematics():
    """
    Show the mathematics of vanishing gradients
    """
    print("\n\n📉 VANISHING GRADIENT: The Mathematics")
    print("=" * 45)

    print("🎯 WHY GRADIENTS VANISH:")
    print()

    print("📐 SIMPLIFIED ANALYSIS:")
    print("   Consider: h_t = tanh(W_hh × h_{t-1})")
    print("   Derivative: ∂h_t/∂h_{t-1} = W_hh × tanh'(...)")
    print()
    print("   For tanh: |tanh'(x)| ≤ 1 always")
    print("   So: |∂h_t/∂h_{t-1}| ≤ |W_hh|")
    print()

    print("🔄 CHAIN RULE OVER T TIME STEPS:")
    print("   ∂h_T/∂h_1 = (∂h_T/∂h_{T-1}) × (∂h_{T-1}/∂h_{T-2}) × ... × (∂h_2/∂h_1)")
    print("   |∂h_T/∂h_1| ≤ |W_hh|^{T-1}")
    print()

    print("📊 NUMERICAL EXAMPLE:")
    scenarios = [
        ("Small weights", 0.5, 10),
        ("Medium weights", 0.9, 10),
        ("Large weights", 1.1, 10),
    ]

    print("   Scenario        | W_hh | Steps | |W_hh|^steps | Gradient")
    print("   ----------------|------|-------|-------------|----------")

    for scenario, w, steps in scenarios:
        gradient_factor = abs(w) ** steps
        if gradient_factor < 0.001:
            status = "VANISHED! 💀"
        elif gradient_factor > 1000:
            status = "EXPLODED! 💥"
        else:
            status = "OK ✅"

        print(
            f"   {scenario:<15} | {w:4.1f} | {steps:5} | {gradient_factor:11.6f} | {status}"
        )

    print()
    print("💡 THE INSIGHT:")
    print("   After 10 steps with W_hh=0.5: gradient ≈ 0.001")
    print("   Network can barely learn from early time steps!")
    print("   This is why basic RNNs struggle with long sequences!")


def memory_capacity_analysis():
    """
    Analyze the memory capacity of RNNs
    """
    print("\n\n🧠 RNN MEMORY CAPACITY ANALYSIS")
    print("=" * 40)

    print("🎯 HOW MUCH CAN RNNs REMEMBER?")
    print()

    print("📊 MEMORY BOTTLENECK:")
    print("   Hidden state size = Memory capacity")
    print("   Example: h_t has 100 dimensions")
    print("   Must encode ALL past information in 100 numbers!")
    print()

    print("🔍 INFORMATION COMPRESSION:")
    print("   Step 1: 'The' → Encode in 100 numbers")
    print("   Step 2: 'cat' + previous → Still 100 numbers")
    print("   Step 3: 'is' + previous → Still 100 numbers")
    print("   ...")
    print("   Step 50: All 50 words → Still 100 numbers!")
    print()
    print("   💔 Information must be COMPRESSED/FORGOTTEN!")
    print()

    print("📈 MEMORY DECAY ANALYSIS:")
    print("   Assume memory decays by factor α each step")
    print("   Information from t steps ago has weight α^t")
    print()

    decay_factors = [0.9, 0.8, 0.5]
    time_steps = [1, 5, 10, 20, 50]

    print("   Decay | Steps: 1    5     10    20    50")
    print("   ------|----------------------------------")

    for alpha in decay_factors:
        row = f"   {alpha:4.1f}  |"
        for t in time_steps:
            weight = alpha**t
            row += f" {weight:5.3f}"
        print(row)

    print()
    print("💡 INTERPRETATION:")
    print("   With α=0.8: After 10 steps, only 10% of original info remains")
    print("   With α=0.8: After 50 steps, virtually nothing remains!")
    print("   This explains why RNNs forget long-term context!")


def practical_implications():
    """
    Discuss practical implications of RNN memory limitations
    """
    print("\n\n🎯 PRACTICAL IMPLICATIONS")
    print("=" * 30)

    print("✅ WHAT RNNs ARE GOOD AT:")
    print("   - Short sequences (< 10-20 steps)")
    print("   - Local dependencies")
    print("   - Recent context matters most")
    print()
    print("   Examples:")
    print("   📝 'The cat is very happy' → sentiment analysis ✅")
    print("   🎵 Next note in a short melody ✅")
    print("   📈 Stock price prediction (short-term) ✅")
    print()

    print("❌ WHAT RNNs STRUGGLE WITH:")
    print("   - Long sequences (> 50 steps)")
    print("   - Long-range dependencies")
    print("   - Information from early context")
    print()
    print("   Examples:")
    print("   📚 'The cat, which was born in France and lived there")
    print("       for many years before moving to Spain where it")
    print("       learned to hunt mice, was sleeping.' ❌")
    print("       (RNN forgets 'cat' by the time it sees 'was sleeping')")
    print()
    print("   💬 Long conversations ❌")
    print("   📖 Document summarization ❌")
    print("   🔄 Long-term memory tasks ❌")


def the_lstm_motivation():
    """
    Explain why this led to LSTMs
    """
    print("\n\n🚀 THIS LED TO LSTMs!")
    print("=" * 25)

    print("🤔 THE CRITICAL QUESTIONS:")
    print("   - How can we fix the vanishing gradient problem?")
    print("   - How can we maintain long-term memory?")
    print("   - How can we selectively forget/remember?")
    print()

    print("💡 THE LSTM BREAKTHROUGH (1997):")
    print("   Hochreiter & Schmidhuber's solution:")
    print("   'What if memory had GATES?'")
    print()

    print("🚪 THE GATE CONCEPT:")
    print("   - FORGET GATE: 'What should I forget from old memory?'")
    print("   - INPUT GATE: 'What new information should I store?'")
    print("   - OUTPUT GATE: 'What should I output from memory?'")
    print()

    print("🔄 LSTM vs RNN COMPARISON:")
    print("   RNN:  h_t = tanh(W_hh × h_{t-1} + W_xh × x_t)")
    print("         ↑ Fixed mixing of old and new")
    print()
    print("   LSTM: Uses gates to control information flow")
    print("         ↑ Selective forgetting and remembering")
    print()

    print("📈 THE RESULTS:")
    print("   ✅ Can remember information for 100+ time steps")
    print("   ✅ Solves vanishing gradient problem")
    print("   ✅ Enabled machine translation, chatbots, etc.")
    print("   ✅ Foundation for modern language models!")


def key_mathematical_insights():
    """
    Summarize the key mathematical insights
    """
    print("\n\n🎯 KEY MATHEMATICAL INSIGHTS")
    print("=" * 35)

    print("1️⃣ MEMORY IS THE HIDDEN STATE:")
    print("   h_t encodes all past information")
    print("   Updated each time step with new input")
    print()

    print("2️⃣ MEMORY UPDATE FORMULA:")
    print("   h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print("   Combines: old memory + new input → new memory")
    print()

    print("3️⃣ SAME WEIGHTS ACROSS TIME:")
    print("   W_hh, W_xh, W_hy shared across all time steps")
    print("   This enables learning sequential patterns")
    print()

    print("4️⃣ GRADIENT CHAINS THROUGH TIME:")
    print("   ∂Error/∂W = Σ(gradients from all time steps)")
    print("   Enables learning from entire sequence")
    print()

    print("5️⃣ VANISHING GRADIENT MATHEMATICS:")
    print("   |∂h_T/∂h_1| ≤ |W_hh|^{T-1}")
    print("   Exponential decay limits long-term learning")
    print()

    print("6️⃣ MEMORY CAPACITY BOTTLENECK:")
    print("   All sequence information compressed into h_t")
    print("   Fixed-size memory creates forgetting")
    print()

    print("💡 THE FUNDAMENTAL TRADE-OFF:")
    print("   RNNs can handle sequences BUT have memory limitations")
    print("   This limitation drove the need for LSTM innovation!")


if __name__ == "__main__":
    print("🧠 RNN MEMORY MATHEMATICS: The Complete Mechanism")
    print("=" * 60)
    print("Understanding exactly how RNNs manage memory!")
    print()

    # Core equations
    rnn_core_equations()

    # Step by step breakdown
    memory_update_step_by_step()

    # Numerical example
    concrete_numerical_example()

    # Memory evolution
    memory_visualization()

    # Gradient flow
    the_gradient_flow()

    # Vanishing gradient math
    vanishing_gradient_mathematics()

    # Memory capacity
    memory_capacity_analysis()

    # Practical implications
    practical_implications()

    # LSTM motivation
    the_lstm_motivation()

    # Key insights
    key_mathematical_insights()

    print("\n🌟 RNN MEMORY MATHEMATICS MASTERED!")
    print("You now understand:")
    print("- Exactly how RNNs manage memory mathematically")
    print("- Why gradients vanish over long sequences")
    print("- Memory capacity limitations")
    print("- Why LSTMs were desperately needed!")
    print()
    print("Ready to see how LSTMs solve these problems? 🔒✨")
