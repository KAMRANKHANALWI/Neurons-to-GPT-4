"""
🧠 RNN FORMULA: Complete Symbol Breakdown
=========================================

Let's break down the RNN formula EXACTLY like we did for LSTM!
Every symbol explained simply, step by step!

h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)

What the heck is all this?? Let's decode it! 🕵️
"""

import numpy as np


def break_down_rnn_formula():
    """
    Break down every single symbol in the RNN formula
    """
    print("🔍 BREAKING DOWN THE RNN FORMULA")
    print("=" * 35)

    print("😱 THE RNN FORMULA:")
    print("   h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print()

    print("🕵️ LET'S DECODE EVERY SYMBOL:")
    print()

    print("📌 h_t:")
    print("   = 'hidden state at time t'")
    print("   = THE MEMORY! (what RNN remembers)")
    print("   = A list of numbers representing current understanding")
    print("   Example: h_t = [0.7, 0.3, 0.8] (remembering: happy, cat, sleeping)")
    print()

    print("📌 tanh:")
    print("   = 'hyperbolic tangent function' (the squashing function)")
    print("   = Takes any number and squashes it between -1 and +1")
    print("   = tanh(big positive) ≈ +1")
    print("   = tanh(big negative) ≈ -1")
    print("   = tanh(0) = 0")
    print()

    print("📌 W_hh:")
    print("   = 'hidden-to-hidden weights'")
    print("   = 'How much should I trust my previous memory?'")
    print("   = Numbers that control memory influence")
    print("   = Learned automatically during training!")
    print()

    print("📌 h_{t-1}:")
    print("   = 'hidden state at time t-1'")
    print("   = PREVIOUS MEMORY (what RNN remembered before)")
    print("   = The memory from the last time step")
    print("   Example: h_{t-1} = [0.5, 0.8, 0.2] (previous understanding)")
    print()

    print("📌 W_xh:")
    print("   = 'input-to-hidden weights'")
    print("   = 'How much should I trust the new input?'")
    print("   = Numbers that control input influence")
    print("   = Also learned automatically!")
    print()

    print("📌 x_t:")
    print("   = 'input at time t'")
    print("   = NEW INFORMATION coming in right now")
    print("   = Current word, current data point, etc.")
    print("   Example: x_t = [1, 0, 0] (word 'cat' in one-hot encoding)")
    print()

    print("📌 b_h:")
    print("   = 'hidden bias'")
    print("   = Adjustment number (fine-tuning)")
    print("   = Like adding a constant to help decision")
    print("   = Learned automatically too!")


def translate_rnn_to_simple_english():
    """
    Translate the RNN formula to simple English
    """
    print("\n\n🗣️ TRANSLATING RNN TO SIMPLE ENGLISH")
    print("=" * 45)

    print("😱 COMPLEX RNN VERSION:")
    print("   h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print()

    print("😊 SIMPLE ENGLISH VERSION:")
    print(
        "   new_memory = squash(memory_weights × old_memory + input_weights × new_input + adjustment)"
    )
    print()

    print("🎯 EVEN SIMPLER:")
    print(
        "   new_memory = squash(trusted_old_memory + trusted_new_input + fine_tuning)"
    )
    print()

    print("📝 STEP BY STEP:")
    print("   Step 1: Take previous memory and current input")
    print("   Step 2: Multiply previous memory by 'trust weights'")
    print("   Step 3: Multiply new input by 'importance weights'")
    print("   Step 4: Add both together + fine-tuning bias")
    print("   Step 5: Squash result between -1 and +1 (tanh)")
    print("   Step 6: This becomes the NEW MEMORY!")


def compare_rnn_to_rosenblatt():
    """
    Compare RNN formula to Rosenblatt's formula
    """
    print("\n\n🔗 COMPARE RNN TO ROSENBLATT'S FORMULA")
    print("=" * 45)

    print("🎯 ROSENBLATT'S FORMULA:")
    print("   new_weight = old_weight + (learning_rate × error × input)")
    print()
    print("   What each part means:")
    print("   new_weight = result we want")
    print("   old_weight = what we had before")
    print("   learning_rate = how much to change")
    print("   error × input = direction and amount to change")
    print()

    print("🧠 RNN MEMORY UPDATE FORMULA:")
    print("   h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print()
    print("   What each part means:")
    print("   h_t = result we want (new memory)")
    print("   h_{t-1} = what we had before (old memory)")
    print("   W_hh × h_{t-1} = how much old memory to keep")
    print("   W_xh × x_t = how much new input to add")
    print("   b_h = fine-tuning adjustment")
    print("   tanh = activation function (like step function)")
    print()

    print("💡 THE BEAUTIFUL SIMILARITY:")
    print("   Rosenblatt: new = old + change")
    print("   RNN: new_memory = mix(old_memory + new_input)")
    print("   Both combine old and new information!")


def tanh_function_explained():
    """
    Explain the tanh function simply
    """
    print("\n\n📈 THE TANH FUNCTION EXPLAINED")
    print("=" * 35)

    print("🤔 WHAT IS TANH?")
    print("   tanh = 'hyperbolic tangent' = S-shaped curve")
    print("   Job: Take ANY number and squash it between -1 and +1")
    print("   Perfect for memory! (negative = inhibition, positive = activation)")
    print()

    print("🧮 THE MATH:")
    print("   tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))")
    print("   Don't worry about the formula, just know what it does!")
    print()

    print("📊 EXAMPLES:")
    test_values = [-3, -1, 0, 1, 3]
    print("   Input | Tanh Output | Meaning")
    print("   ------|-------------|------------------------")

    for x in test_values:
        tanh_output = np.tanh(x)
        if tanh_output < -0.5:
            meaning = "Strong INHIBITION"
        elif tanh_output < 0:
            meaning = "Weak inhibition"
        elif tanh_output < 0.5:
            meaning = "Weak activation"
        else:
            meaning = "Strong ACTIVATION"
        print(f"   {x:5} | {tanh_output:11.3f} | {meaning}")

    print()
    print("💡 WHY TANH FOR RNN MEMORY:")
    print("   Negative values = 'Forget' or 'Inhibit'")
    print("   Zero = 'Neutral'")
    print("   Positive values = 'Remember' or 'Activate'")
    print("   Range -1 to +1 = Perfect for memory updates!")


def rnn_matrix_multiplication():
    """
    Explain RNN matrix multiplication simply
    """
    print("\n\n🔢 RNN MATRIX MULTIPLICATION MADE SIMPLE")
    print("=" * 50)

    print("😰 WHAT IS W_hh × h_{t-1} + W_xh × x_t?")
    print("   It's two separate weighted sums added together!")
    print("   Just like Rosenblatt, but with TWO sources of information!")
    print()

    print("🎯 SIMPLE EXAMPLE:")
    print("   Previous memory: h_{t-1} = [0.8, 0.3]")
    print("   New input: x_t = [1, 0]")
    print("   Memory weights: W_hh = [[0.5, 0.2], [0.3, 0.7]]")
    print("   Input weights: W_xh = [[0.4, 0.1], [0.6, 0.2]]")
    print()

    print("🧮 STEP 1 - Process old memory:")
    print("   W_hh × h_{t-1} = memory_contribution")
    print("   Row 1: (0.5 × 0.8) + (0.2 × 0.3) = 0.4 + 0.06 = 0.46")
    print("   Row 2: (0.3 × 0.8) + (0.7 × 0.3) = 0.24 + 0.21 = 0.45")
    print("   memory_contribution = [0.46, 0.45]")
    print()

    print("🧮 STEP 2 - Process new input:")
    print("   W_xh × x_t = input_contribution")
    print("   Row 1: (0.4 × 1) + (0.1 × 0) = 0.4 + 0 = 0.4")
    print("   Row 2: (0.6 × 1) + (0.2 × 0) = 0.6 + 0 = 0.6")
    print("   input_contribution = [0.4, 0.6]")
    print()

    print("🧮 STEP 3 - Add them together:")
    print("   combined = memory_contribution + input_contribution")
    print("   combined = [0.46, 0.45] + [0.4, 0.6] = [0.86, 1.05]")
    print()

    print("💡 IT'S LIKE MIXING TWO INGREDIENTS:")
    print("   Ingredient 1: Processed old memory")
    print("   Ingredient 2: Processed new input")
    print("   Recipe: Add them together for final result!")


def complete_rnn_example():
    """
    Work through complete RNN example with real numbers
    """
    print("\n\n🔢 COMPLETE RNN EXAMPLE WITH REAL NUMBERS")
    print("=" * 50)

    print("🎯 SCENARIO: Processing word 'happy' after remembering 'cat'")
    print("   Previous memory: [0.8, 0.3] (remembering: positive sentiment about cat)")
    print("   New input: [0, 1] (word 'happy')")
    print()

    print("📊 GIVEN VALUES:")
    h_prev = np.array([0.8, 0.3])
    x_current = np.array([0, 1])
    W_hh = np.array([[0.5, 0.2], [0.3, 0.7]])
    W_xh = np.array([[0.4, 0.6], [0.2, 0.8]])
    b_h = np.array([0.1, -0.1])

    print(f"   h_{{t-1}} (previous memory): {h_prev}")
    print(f"   x_t (current input 'happy'): {x_current}")
    print(f"   W_hh (memory weights): \\n{W_hh}")
    print(f"   W_xh (input weights): \\n{W_xh}")
    print(f"   b_h (bias): {b_h}")
    print()

    print("🔄 STEP-BY-STEP RNN CALCULATION:")
    print()

    print("STEP 1: Process previous memory")
    memory_contribution = np.dot(W_hh, h_prev)
    print(f"   W_hh × h_{{t-1}} = {memory_contribution}")
    print("   This captures what we should remember from before")
    print()

    print("STEP 2: Process new input")
    input_contribution = np.dot(W_xh, x_current)
    print(f"   W_xh × x_t = {input_contribution}")
    print("   This processes the new word 'happy'")
    print()

    print("STEP 3: Combine information")
    combined = memory_contribution + input_contribution + b_h
    print(f"   Combined = {memory_contribution} + {input_contribution} + {b_h}")
    print(f"   Combined = {combined}")
    print()

    print("STEP 4: Apply tanh activation")
    new_memory = np.tanh(combined)
    print(f"   h_t = tanh({combined}) = {new_memory}")
    print()

    print("🎉 FINAL RESULT:")
    print(f"   Previous memory: {h_prev}")
    print(f"   New memory: {new_memory}")
    print("   Interpretation:")
    print(
        f"   Memory[0]: {h_prev[0]:.1f} → {new_memory[0]:.3f} (sentiment got stronger!)"
    )
    print(f"   Memory[1]: {h_prev[1]:.1f} → {new_memory[1]:.3f} (context updated)")
    print()
    print("   RNN successfully combined 'cat' memory with 'happy' input!")
    print("   New memory captures: 'happy cat' concept! 🐱😊")


def rnn_vs_lstm_formulas():
    """
    Compare RNN and LSTM formulas side by side
    """
    print("\n\n🆚 RNN vs LSTM FORMULAS SIDE BY SIDE")
    print("=" * 45)

    print("🧠 RNN FORMULA (Simple):")
    print("   h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print("   Translation: new_memory = squash(old_memory + new_input)")
    print("   Gates: NONE - everything mixes together")
    print("   Memory control: BASIC - no selective forget/remember")
    print()

    print("🔒 LSTM FORMULAS (Smart):")
    print("   f_t = σ(W_f × [h_{t-1}, x_t] + b_f)    ← Forget gate")
    print("   i_t = σ(W_i × [h_{t-1}, x_t] + b_i)    ← Input gate")
    print("   C̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c) ← Candidate values")
    print("   C_t = f_t * C_{t-1} + i_t * C̃_t        ← Cell state update")
    print("   o_t = σ(W_o × [h_{t-1}, x_t] + b_o)    ← Output gate")
    print("   h_t = o_t * tanh(C_t)                  ← Final output")
    print()
    print("   Gates: THREE - smart control of information")
    print("   Memory control: ADVANCED - selective operations")
    print()

    print("💡 THE KEY DIFFERENCE:")
    print("   RNN: 1 formula, everything mixed")
    print("   LSTM: 6 formulas, smart gates controlling everything")
    print("   RNN: Simple but forgetful")
    print("   LSTM: Complex but powerful memory!")


def formula_evolution_story():
    """
    Tell the story of formula evolution
    """
    print("\n\n📈 THE FORMULA EVOLUTION STORY")
    print("=" * 35)

    print("🎭 ACT 1: McCulloch-Pitts (1943)")
    print("   Formula: output = step(Σ(weights × inputs) - threshold)")
    print("   Innovation: First mathematical neuron")
    print("   Limitation: No learning, no memory")
    print()

    print("🎭 ACT 2: Perceptron (1957)")
    print("   Formula: new_weight = old_weight + (rate × error × input)")
    print("   Innovation: Learning!")
    print("   Limitation: Only linear problems")
    print()

    print("🎭 ACT 3: Multi-layer + Backprop (1986)")
    print("   Formula: h = activation(W × x + b) (multiple layers)")
    print("   Innovation: Non-linear problem solving")
    print("   Limitation: No memory across time")
    print()

    print("🎭 ACT 4: RNN (1990s)")
    print("   Formula: h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print("   Innovation: Memory! Time sequences!")
    print("   Limitation: Forgets long-term information")
    print()

    print("🎭 ACT 5: LSTM (1997)")
    print("   Formula: 6 equations with smart gates")
    print("   Innovation: Long-term memory with selective control")
    print("   Achievement: Enabled modern language AI!")
    print()

    print("🌟 THE PATTERN:")
    print("   Each formula builds on the previous one")
    print("   Each solves the limitations of the previous")
    print("   All use the same core: weights × inputs + bias")
    print("   Evolution: Simple → Learning → Non-linear → Memory → Smart Memory")


def naming_conventions():
    """
    Explain naming conventions for RNN formula
    """
    print("\n\n🏷️ WHAT TO CALL THE RNN FORMULA")
    print("=" * 35)

    print("🎯 DIFFERENT NAMES FOR THE SAME THING:")
    print()

    print("📚 SIMPLE NAMES:")
    print("   - 'RNN memory update formula'")
    print("   - 'Recurrent hidden state equation'")
    print("   - 'Memory mixing formula'")
    print()

    print("🎓 TECHNICAL NAMES:")
    print("   - 'Tanh-activated recurrent transformation'")
    print("   - 'Hidden state recurrence relation'")
    print("   - 'Temporal state transition function'")
    print()

    print("🔬 ACADEMIC NAMES:")
    print("   - 'Elman network update rule'")
    print("   - 'Simple recurrent unit (SRU) equation'")
    print("   - 'Vanilla RNN dynamics'")
    print()

    print("💡 WHAT IT REALLY IS:")
    print("   At its core: A neuron that feeds its output back to itself")
    print("   Purpose: Maintain memory across time steps")
    print("   Method: Mix old memory with new input")
    print()

    print("🎭 LIKE A CONVERSATION:")
    print("   Each time step: 'Here's what I remember + here's what's new'")
    print("   Formula: 'Let me update my understanding'")
    print("   Result: 'This is my new understanding for next time'")


def key_takeaways_rnn():
    """
    Summarize key takeaways about RNN formula
    """
    print("\n\n🎯 KEY TAKEAWAYS ABOUT RNN FORMULA")
    print("=" * 45)

    print("✅ EVERY SYMBOL DECODED:")
    print("   h_t = new memory (the result)")
    print("   tanh = squashing function (-1 to +1)")
    print("   W_hh = memory trust weights (learned)")
    print("   h_{t-1} = previous memory")
    print("   W_xh = input importance weights (learned)")
    print("   x_t = new input")
    print("   b_h = fine-tuning bias (learned)")
    print()

    print("✅ SIMPLE TRANSLATION:")
    print("   Complex: h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print("   Simple: new_memory = squash(trusted_old + important_new + adjustment)")
    print()

    print("✅ SAME PATTERN AS ALWAYS:")
    print("   All formulas: result = activation(weights × inputs + bias)")
    print("   RNN just has TWO types of inputs: memory + new_data")
    print()

    print("✅ THE MEMORY MAGIC:")
    print("   RNN formula creates a 'memory loop'")
    print("   Output becomes input for next time step")
    print("   This enables understanding sequences!")
    print()

    print("💡 THE BREAKTHROUGH UNDERSTANDING:")
    print("   RNN = Neuron with memory feedback loop")
    print("   Formula = How to update memory each step")
    print("   Simple but powerful for sequence learning!")
    print()

    print("🚀 READY FOR ANYTHING:")
    print("   Now you understand BOTH RNN and LSTM formulas!")
    print("   Same principles, different complexity levels!")
    print("   Foundation for understanding ANY neural network! ✨")


if __name__ == "__main__":
    print("🧠 RNN FORMULA: Complete Symbol Breakdown!")
    print("=" * 50)
    print("Breaking down h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)")
    print("EXACTLY like we did for LSTM! 🕵️")
    print()

    # Break down every symbol
    break_down_rnn_formula()

    # Translate to simple English
    translate_rnn_to_simple_english()

    # Compare to Rosenblatt
    compare_rnn_to_rosenblatt()

    # Tanh function explained
    tanh_function_explained()

    # Matrix multiplication
    rnn_matrix_multiplication()

    # Complete example
    complete_rnn_example()

    # RNN vs LSTM formulas
    rnn_vs_lstm_formulas()

    # Evolution story
    formula_evolution_story()

    # Naming conventions
    naming_conventions()

    # Key takeaways
    key_takeaways_rnn()

    print("\n🌟 RNN FORMULA DECODED!")
    print("Now you understand every single symbol!")
    print("h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h) = crystal clear! ✅")
    print()
    print("You've mastered BOTH RNN and LSTM formulas! 🧠🔒")
    print("Ready for the TRANSFORMER revolution? ⚡🤖")
