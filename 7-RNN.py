"""
🧠 RNNs: HOW AI GOT MEMORY!
===========================

The Big Problem: Regular neural networks are AMNESIACS!
They see each input independently, with NO memory of what came before.

But what if we need to understand SEQUENCES?
- "I love this movie, it was..." → Need to remember "love" to understand "it"
- Stock prices: Today's price depends on yesterday's trend
- Language: "The cat that I saw yesterday was..." → Remember which "cat"!

Enter RNNs: Neural networks with MEMORY! 🧠💭
"""

import numpy as np
import matplotlib.pyplot as plt


def the_memory_problem():
    """
    Show why regular neural networks fail at sequences
    """
    print("🤔 THE MEMORY PROBLEM")
    print("=" * 25)

    print("🧠 REGULAR NEURAL NETWORKS:")
    print("   Each input is processed INDEPENDENTLY")
    print("   No memory of previous inputs")
    print("   Like having amnesia every second! 🤕")
    print()

    print("📝 EXAMPLE: Movie Review Analysis")
    print("   Sentence: 'I love this movie, it was amazing!'")
    print()
    print("   Regular Network processes word by word:")
    print("   Word 1: 'I' → Network: 'Neutral word'")
    print("   Word 2: 'love' → Network: 'Positive word'")
    print("   Word 3: 'this' → Network: 'Neutral word'")
    print("   Word 4: 'movie' → Network: 'Neutral word'")
    print("   Word 5: 'it' → Network: 'What does 'it' refer to??' 😵")
    print()
    print("   ❌ PROBLEM: By word 5, network forgot about 'love' and 'movie'!")
    print("   ❌ Can't understand that 'it' refers to the loved movie!")
    print()

    print("💡 WHAT WE NEED:")
    print("   Memory to connect: 'I love this movie' + 'it was amazing'")
    print("   Understanding: 'it' = the movie that was loved")
    print("   Result: POSITIVE review! ✅")


def the_rnn_breakthrough():
    """
    Introduce the RNN solution
    """
    print("\n\n🚀 THE RNN BREAKTHROUGH!")
    print("=" * 30)

    print("💡 THE GENIUS IDEA:")
    print("   'What if the network could remember what it saw before?'")
    print("   'What if it had a MEMORY that updates with each input?'")
    print()

    print("🧠 RNN = Regular Neural Network + MEMORY!")
    print()
    print("   📥 INPUT: Current word")
    print("   🧠 MEMORY: What I've seen so far")
    print("   🔄 PROCESS: Combine current input + memory")
    print("   📤 OUTPUT: Prediction + Updated memory")
    print()

    print("🔄 THE RNN LOOP:")
    print("   Step 1: See 'I' → Memory: 'Someone is speaking'")
    print("   Step 2: See 'love' → Memory: 'Someone loves something'")
    print("   Step 3: See 'this' → Memory: 'Someone loves this thing'")
    print("   Step 4: See 'movie' → Memory: 'Someone loves this movie'")
    print("   Step 5: See 'it' → Memory knows 'it' = the loved movie!")
    print("   Step 6: See 'amazing' → CONCLUSION: Very positive review! 🎉")


def rnn_architecture_simple():
    """
    Show the basic RNN architecture
    """
    print("\n\n🏗️ RNN ARCHITECTURE: The Memory Machine")
    print("=" * 45)

    print("📐 BASIC RNN STRUCTURE:")
    print()
    print("   Input(t) ────┐")
    print("               ↓")
    print("   Memory(t-1) → [RNN CELL] → Output(t)")
    print("               ↓")
    print("               Memory(t) ──┐")
    print("                          ↓")
    print("                     (saves for next step)")
    print()

    print("💡 KEY INSIGHT:")
    print("   The SAME cell processes each time step")
    print("   But it REMEMBERS what it learned from previous steps!")
    print()

    print("🔄 UNFOLDED THROUGH TIME:")
    print()
    print("   Input(1)    Input(2)    Input(3)    Input(4)")
    print("      ↓           ↓           ↓           ↓")
    print("   [RNN] ────> [RNN] ────> [RNN] ────> [RNN]")
    print("      ↓           ↓           ↓           ↓")
    print("  Output(1)   Output(2)   Output(3)   Output(4)")
    print()
    print("   Memory flows left to right →→→→")
    print("   Same weights used at each step!")


def rnn_math_simple():
    """
    Show the simple math behind RNNs
    """
    print("\n\n🧮 RNN MATH: Surprisingly Simple!")
    print("=" * 40)

    print("🎯 THE CORE FORMULA:")
    print()
    print("   Memory(t) = tanh(Input(t) × W_input + Memory(t-1) × W_memory + bias)")
    print("   Output(t) = Memory(t) × W_output")
    print()

    print("🔍 BREAKING IT DOWN:")
    print()
    print("   📥 Input(t): Current word (as numbers)")
    print("   🧠 Memory(t-1): What we remembered from before")
    print("   ⚖️ W_input: How much to trust current input")
    print("   ⚖️ W_memory: How much to trust old memory")
    print("   ➕ bias: Adjustment factor")
    print("   📈 tanh: Keeps values between -1 and +1")
    print()

    print("💡 INTUITION:")
    print("   'Combine what I see NOW with what I REMEMBER'")
    print("   'Create new memory that includes both!'")
    print("   'Use this new memory to make prediction!'")


def step_by_step_example():
    """
    Walk through a concrete example
    """
    print("\n\n👣 STEP-BY-STEP EXAMPLE")
    print("=" * 30)

    print("📝 SENTENCE: 'Cat is happy'")
    print("🎯 TASK: Predict sentiment (positive/negative)")
    print()

    # Simplified example with made-up numbers
    print("🔢 SIMPLIFIED NUMBERS:")
    print("   Word embeddings: Cat=0.2, is=0.1, happy=0.8")
    print("   W_input = 0.5, W_memory = 0.3, bias = 0.1")
    print("   Initial memory = 0.0")
    print()

    memory = 0.0
    W_input = 0.5
    W_memory = 0.3
    bias = 0.1

    words = [("Cat", 0.2), ("is", 0.1), ("happy", 0.8)]

    print("🔄 PROCESSING EACH WORD:")
    print()

    for step, (word, input_val) in enumerate(words, 1):
        print(f"📍 STEP {step}: '{word}'")
        print(f"   Input: {input_val}")
        print(f"   Previous memory: {memory:.3f}")

        # Calculate new memory (simplified tanh ≈ just the linear part)
        new_memory = input_val * W_input + memory * W_memory + bias
        # Simple activation (instead of tanh)
        new_memory = max(-1, min(1, new_memory))  # Clip between -1 and 1

        print(
            f"   New memory = {input_val} × {W_input} + {memory:.3f} × {W_memory} + {bias}"
        )
        print(f"   New memory = {new_memory:.3f}")

        # Sentiment prediction (positive if > 0.5)
        sentiment = (
            "POSITIVE"
            if new_memory > 0.3
            else "NEGATIVE" if new_memory < -0.3 else "NEUTRAL"
        )
        print(f"   Current sentiment prediction: {sentiment}")
        print()

        memory = new_memory

    print("🎉 FINAL RESULT:")
    print(f"   Final memory: {memory:.3f}")
    print(
        f"   Final prediction: {'POSITIVE! 😊' if memory > 0.3 else 'NEGATIVE 😢' if memory < -0.3 else 'NEUTRAL 😐'}"
    )
    print()
    print("💡 THE MAGIC:")
    print("   RNN remembered 'Cat' and 'is' while processing 'happy'")
    print("   Built up understanding: 'A cat is in a happy state'")
    print("   Final judgment: POSITIVE sentiment! ✅")


def rnn_applications():
    """
    Show what RNNs can do
    """
    print("\n\n🌟 WHAT RNNs REVOLUTIONIZED")
    print("=" * 35)

    applications = [
        (
            "📝 Language Translation",
            "Input: 'Hello world' → Output: 'Hola mundo'",
            "Remembers context while translating",
        ),
        (
            "🎵 Music Generation",
            "Input: Previous notes → Output: Next note",
            "Remembers melody and rhythm patterns",
        ),
        (
            "📈 Stock Prediction",
            "Input: Historical prices → Output: Future price",
            "Remembers price trends and patterns",
        ),
        (
            "🗣️ Speech Recognition",
            "Input: Audio sequence → Output: Text words",
            "Remembers partial words and context",
        ),
        (
            "💬 Chatbots",
            "Input: User message → Output: Response",
            "Remembers conversation history",
        ),
        (
            "📚 Text Summarization",
            "Input: Long document → Output: Key points",
            "Remembers important themes throughout",
        ),
    ]

    print("🚀 RNN APPLICATIONS:")
    print()
    for app, example, how in applications:
        print(f"{app}")
        print(f"   Example: {example}")
        print(f"   How: {how}")
        print()

    print("🎯 THE COMMON PATTERN:")
    print("   All involve processing SEQUENCES where ORDER MATTERS!")
    print("   Memory allows understanding of context and dependencies!")


def the_vanishing_gradient_problem():
    """
    Introduce the limitation that led to LSTMs
    """
    print("\n\n⚠️ THE VANISHING GRADIENT PROBLEM")
    print("=" * 40)

    print("🤔 BASIC RNNs HAVE A MEMORY PROBLEM:")
    print("   They can remember recent things well")
    print("   But they FORGET things from long ago!")
    print()

    print("📝 EXAMPLE PROBLEM:")
    print("   Sentence: 'The cat, which was black and white and very fluffy,")
    print("             and lived in Paris for many years, was sleeping.'")
    print()
    print("   Word 1: 'The cat' → RNN: 'There's a cat' ✅")
    print("   Word 15: 'was sleeping' → RNN: 'Something was sleeping... but what??' ❌")
    print()
    print("   💔 PROBLEM: RNN forgot about the cat!")
    print("   Memory decayed over the long sentence!")
    print()

    print("🧠 WHY THIS HAPPENS:")
    print("   During backpropagation through time:")
    print("   Gradients get smaller and smaller as they flow backward")
    print("   Early words get almost no learning signal!")
    print("   Result: RNN can't learn long-term dependencies 😢")
    print()

    print("💡 THE SOLUTION PREVIEW:")
    print("   This limitation led to LSTM networks!")
    print("   LSTMs have special 'gates' to maintain long-term memory")
    print("   Coming up next in our journey! 🚀")


def comparison_with_previous_networks():
    """
    Compare RNNs with what we learned before
    """
    print("\n\n🔗 CONNECTION TO PREVIOUS NETWORKS")
    print("=" * 40)

    print("📊 NETWORK COMPARISON:")
    print()

    networks = [
        ("Perceptron", "Single decision", "AND/OR gates", "No memory"),
        ("Multi-layer", "Complex decisions", "XOR, classification", "No memory"),
        ("CNN", "Spatial patterns", "Image recognition", "No sequence memory"),
        ("RNN", "Sequential patterns", "Language, time series", "Has memory! 🧠"),
    ]

    print("   Network      | Purpose           | Example Use       | Memory")
    print("   -------------|-------------------|-------------------|----------")
    for name, purpose, example, memory in networks:
        print(f"   {name:<12} | {purpose:<17} | {example:<17} | {memory}")

    print()
    print("🎯 THE PROGRESSION:")
    print("   Perceptron: Learn simple patterns")
    print("   Multi-layer: Learn complex patterns")
    print("   CNN: Learn spatial patterns")
    print("   RNN: Learn temporal/sequential patterns")
    print()
    print("💡 SAME FOUNDATION:")
    print("   All use the same backpropagation learning!")
    print("   All build on McCulloch-Pitts neurons!")
    print("   RNNs just add the MEMORY component! 🔄")


def whats_next():
    """
    Preview what's coming
    """
    print("\n\n🔜 WHAT'S NEXT?")
    print("=" * 15)

    print("✅ YOU NOW UNDERSTAND:")
    print("   🧠 Regular networks have no memory")
    print("   💭 RNNs add memory to process sequences")
    print("   🔄 Memory updates with each time step")
    print("   📝 Enables language understanding, prediction")
    print("   ⚠️ But basic RNNs forget long-term info")
    print()

    print("🚀 COMING UP - LSTMs:")
    print("   🔒 Gated memory that doesn't vanish")
    print("   🧠 Long-term AND short-term memory")
    print("   🎯 Solve the vanishing gradient problem")
    print("   💪 Remember context across long sequences")
    print()

    print("🎉 THEN WE'LL SEE:")
    print("   How LSTMs enabled Google Translate")
    print("   How they powered early chatbots")
    print("   How they led to modern Transformers!")


if __name__ == "__main__":
    print("🧠 RNNs: How AI Got Memory!")
    print("=" * 35)
    print("The breakthrough that made AI understand sequences!")
    print()

    # The memory problem
    the_memory_problem()

    # RNN breakthrough
    the_rnn_breakthrough()

    # Architecture
    rnn_architecture_simple()

    # Simple math
    rnn_math_simple()

    # Step by step example
    step_by_step_example()

    # Applications
    rnn_applications()

    # Vanishing gradient problem
    the_vanishing_gradient_problem()

    # Comparison with previous
    comparison_with_previous_networks()

    # What's next
    whats_next()

    print("\n🌟 RNN BREAKTHROUGH UNDERSTOOD!")
    print("You now know how AI learned to remember!")
    print("Ready to see how LSTMs solved the long-term memory problem? 🔒✨")
