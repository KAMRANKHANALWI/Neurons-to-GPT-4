"""
🏊‍♂️ POOLING: The Smart AI Summarizer!
=====================================

After filters detect patterns, we need to ask:
"Was the pattern found in this REGION?" (not exact pixel location)

Pooling = Smart way to summarize: "Yes, there's an edge SOMEWHERE here!"
This makes AI robust to small movements and variations!
"""

import numpy as np


def what_is_pooling():
    """
    The most intuitive explanation of pooling!
    """
    print("🤔 WHAT IS POOLING?")
    print("=" * 20)

    print("🎯 POOLING = SMART SUMMARIZATION!")
    print()
    print("Think of it like this:")
    print("   🔍 Filter says: 'I found edges at pixels (5,3), (5,4), (6,3), (6,4)'")
    print(
        "   🧠 Brain asks: 'Do I care about EXACT pixel? Or just that edge exists in that AREA?'"
    )
    print("   💡 Answer: 'Just that it exists in that area!'")
    print()

    print("🏊‍♂️ POOLING OPERATION:")
    print("   Takes a REGION of detected features")
    print("   Summarizes: 'What's the MAIN signal here?'")
    print("   Result: ONE number representing the whole region")
    print()

    print("🎯 WHY THIS IS GENIUS:")
    print("   Makes AI say: 'There's an eye SOMEWHERE in this region'")
    print("   Instead of: 'There's an eye at EXACTLY pixel (47, 23)'")
    print("   Result: AI works even if face moves 1-2 pixels! 🎉")


def max_pooling_example():
    """
    Show max pooling with real numbers
    """
    print("\n\n📊 MAX POOLING: 'What's the STRONGEST signal?'")
    print("=" * 50)

    print("🔍 After edge filter processed a region, we got:")
    feature_map = np.array([[1, 3, 2, 8], [0, 5, 1, 7], [2, 1, 4, 3], [6, 0, 2, 1]])

    print("Feature Map (4x4):")
    print("   [1, 3, 2, 8]")
    print("   [0, 5, 1, 7]")
    print("   [2, 1, 4, 3]")
    print("   [6, 0, 2, 1]")
    print()
    print("💡 High numbers = Strong pattern detected!")
    print("💡 Low numbers = Weak or no pattern")
    print()

    print("🏊‍♂️ MAX POOLING with 2x2 windows:")
    print("Let's summarize each 2x2 region with its MAXIMUM value")
    print()

    # Process each 2x2 region
    regions = [
        ("Top-Left", feature_map[0:2, 0:2], (0, 1)),
        ("Top-Right", feature_map[0:2, 2:4], (0, 1)),
        ("Bottom-Left", feature_map[2:4, 0:2], (2, 3)),
        ("Bottom-Right", feature_map[2:4, 2:4], (2, 3)),
    ]

    pooled_result = np.zeros((2, 2))

    for name, region, (i, j) in regions:
        max_val = np.max(region)
        pooled_result[i // 2, j // 2] = max_val

        print(f"🔍 {name} region:")
        for row in region:
            print(f"   {row}")
        print(f"   MAX value: {max_val}")
        print(f"   💡 Meaning: 'Strongest pattern in this region = {max_val}'")
        print()

    print("📊 FINAL POOLED RESULT (2x2):")
    for row in pooled_result:
        print(f"   {row}")
    print()
    print("🎉 We summarized 4x4 → 2x2 while keeping the important info!")

    return pooled_result


def average_pooling_example():
    """
    Show average pooling as alternative
    """
    print("\n\n📊 AVERAGE POOLING: 'What's the OVERALL signal?'")
    print("=" * 50)

    print("🎯 Different question: 'What's the AVERAGE strength in this region?'")
    print()

    feature_map = np.array([[1, 3, 2, 8], [0, 5, 1, 7], [2, 1, 4, 3], [6, 0, 2, 1]])

    print("Same Feature Map (4x4):")
    print("   [1, 3, 2, 8]")
    print("   [0, 5, 1, 7]")
    print("   [2, 1, 4, 3]")
    print("   [6, 0, 2, 1]")
    print()

    print("🏊‍♂️ AVERAGE POOLING with 2x2 windows:")

    regions = [
        ("Top-Left", feature_map[0:2, 0:2]),
        ("Top-Right", feature_map[0:2, 2:4]),
        ("Bottom-Left", feature_map[2:4, 0:2]),
        ("Bottom-Right", feature_map[2:4, 2:4]),
    ]

    avg_pooled_result = np.zeros((2, 2))

    for idx, (name, region) in enumerate(regions):
        avg_val = np.mean(region)
        row, col = idx // 2, idx % 2
        avg_pooled_result[row, col] = avg_val

        print(f"🔍 {name} region:")
        for r in region:
            print(f"   {r}")
        print(f"   AVERAGE: {avg_val:.1f}")
        print(f"   💡 Meaning: 'Overall pattern strength = {avg_val:.1f}'")
        print()

    print("📊 AVERAGE POOLED RESULT:")
    for row in avg_pooled_result:
        print(f"   {row}")
    print()
    print("🤔 Compare with Max Pooling - different information captured!")


def translation_invariance_demo():
    """
    Show how pooling makes CNNs robust to position changes
    """
    print("\n\n🎯 THE MAGIC: TRANSLATION INVARIANCE!")
    print("=" * 45)

    print("🚗 SCENARIO: Detecting a car in different positions")
    print()

    # Car in position 1
    print("📍 POSITION 1: Car on the left")
    car_left = np.array(
        [
            [8, 7, 0, 0],  # Strong car signal on left
            [9, 6, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    print("Feature Map:")
    for row in car_left:
        print(f"   {row}")

    # Max pool
    left_pooled = np.array(
        [
            [np.max(car_left[0:2, 0:2]), np.max(car_left[0:2, 2:4])],
            [np.max(car_left[2:4, 0:2]), np.max(car_left[2:4, 2:4])],
        ]
    )

    print("After Max Pooling:")
    for row in left_pooled:
        print(f"   {row}")
    print("🎯 Result: [9, 0, 0, 0] - Strong signal in top-left")
    print()

    # Car in position 2
    print("📍 POSITION 2: Car on the right")
    car_right = np.array(
        [
            [0, 0, 8, 7],  # Strong car signal on right
            [0, 0, 9, 6],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    print("Feature Map:")
    for row in car_right:
        print(f"   {row}")

    # Max pool
    right_pooled = np.array(
        [
            [np.max(car_right[0:2, 0:2]), np.max(car_right[0:2, 2:4])],
            [np.max(car_right[2:4, 0:2]), np.max(car_right[2:4, 2:4])],
        ]
    )

    print("After Max Pooling:")
    for row in right_pooled:
        print(f"   {row}")
    print("🎯 Result: [0, 9, 0, 0] - Strong signal in top-right")
    print()

    print("🤯 THE MAGIC INSIGHT:")
    print("   Both positions show STRONG CAR SIGNAL (9)!")
    print("   Network learns: 'Car detected SOMEWHERE in top region'")
    print("   Doesn't care about EXACT pixel position!")
    print("   Result: Robust to small movements! 🎉")


def pooling_benefits():
    """
    Explain the key benefits of pooling
    """
    print("\n\n✨ WHY POOLING IS GENIUS!")
    print("=" * 30)

    print("1️⃣ TRANSLATION INVARIANCE:")
    print("   'I found a face SOMEWHERE in this region'")
    print("   vs 'I found a face at EXACTLY pixel (23, 47)'")
    print("   Result: Works with moved/shifted objects! 📍")
    print()

    print("2️⃣ REDUCES COMPUTATION:")
    print("   4x4 feature map → 2x2 pooled map")
    print("   75% reduction in data!")
    print("   Network runs FASTER! ⚡")
    print()

    print("3️⃣ REDUCES OVERFITTING:")
    print("   Less precise = more generalizable")
    print("   'Face detector' vs 'This exact face at this exact position'")
    print("   Works on NEW images better! 🎯")
    print()

    print("4️⃣ HIERARCHICAL ABSTRACTION:")
    print("   Layer 1: Pixel-level details")
    print("   Layer 2: Small region summaries")
    print("   Layer 3: Larger region summaries")
    print("   Result: From details to big picture! 🏗️")


def real_world_analogy():
    """
    Real-world analogy for pooling
    """
    print("\n\n🌍 REAL-WORLD ANALOGY")
    print("=" * 25)

    print("🏢 COMPANY REPORTING ANALOGY:")
    print()

    print("🔍 FILTERS = DETAILED EMPLOYEES:")
    print("   'I saw suspicious activity at desk 23 at 2:47 PM'")
    print("   'I saw suspicious activity at desk 24 at 2:48 PM'")
    print("   'I saw suspicious activity at desk 25 at 2:49 PM'")
    print()

    print("🏊‍♂️ POOLING = SMART MANAGER:")
    print("   'There was suspicious activity in the EAST WING around 2:45-3:00 PM'")
    print("   (Summarizes details into actionable insight)")
    print()

    print("👔 WHY MANAGERS ARE USEFUL:")
    print("   CEO doesn't need: 'Desk 23, 2:47 PM, Employee wore blue shirt'")
    print("   CEO needs: 'Security issue in East Wing, afternoon shift'")
    print("   Result: Right level of detail for decision making! 🎯")
    print()

    print("🧠 SAME WITH CNN:")
    print("   Next layer doesn't need exact pixel positions")
    print("   Next layer needs: 'Eye detected in upper-left region'")
    print("   Result: Better for high-level pattern recognition! 👁️")


def stride_and_size_explanation():
    """
    Explain stride and pool size parameters
    """
    print("\n\n⚙️ POOLING PARAMETERS")
    print("=" * 25)

    print("🎛️ TWO KEY SETTINGS:")
    print()

    print("1️⃣ POOL SIZE:")
    print("   2x2 pool: Summarize every 2x2 region")
    print("   3x3 pool: Summarize every 3x3 region")
    print("   💡 Bigger pool = More aggressive summarization")
    print()

    print("2️⃣ STRIDE:")
    print("   Stride 2: Move 2 pixels between pools (non-overlapping)")
    print("   Stride 1: Move 1 pixel between pools (overlapping)")
    print("   💡 Bigger stride = More size reduction")
    print()

    print("📊 COMMON COMBINATIONS:")
    print("   2x2 pool, stride 2: Most common (non-overlapping)")
    print("   Input 4x4 → Output 2x2 (4x reduction)")
    print()
    print("   3x3 pool, stride 3: Aggressive reduction")
    print("   Input 6x6 → Output 2x2 (9x reduction)")
    print()
    print("   2x2 pool, stride 1: Overlapping (less common)")
    print("   Input 4x4 → Output 3x3 (less reduction)")


def connection_to_full_cnn():
    """
    Show how pooling fits in the full CNN pipeline
    """
    print("\n\n🔗 POOLING IN FULL CNN PIPELINE")
    print("=" * 35)

    print("🏗️ COMPLETE CNN LAYER:")
    print()
    print("   Input Image (28x28)")
    print("        ↓")
    print("   🔍 CONVOLUTION (apply filters)")
    print("        ↓")
    print("   Feature Maps (26x26) - one per filter")
    print("        ↓")
    print("   🏊‍♂️ POOLING (summarize regions)")
    print("        ↓")
    print("   Pooled Maps (13x13) - reduced size")
    print("        ↓")
    print("   🔄 Repeat: More Conv + Pool layers")
    print("        ↓")
    print("   Final Classification")
    print()

    print("🎯 THE PROGRESSION:")
    print("   Layer 1: 28x28 → Conv → 26x26 → Pool → 13x13")
    print("   Layer 2: 13x13 → Conv → 11x11 → Pool → 5x5")
    print("   Layer 3: 5x5 → Conv → 3x3 → Pool → 1x1")
    print("   Result: From detailed image to single classification!")


def whats_next():
    """
    Preview the complete CNN architecture
    """
    print("\n\n🚀 WHAT'S NEXT?")
    print("=" * 15)

    print("✅ YOU NOW UNDERSTAND:")
    print("   🔍 Filters: Pattern detectors")
    print("   🏊‍♂️ Pooling: Smart summarization")
    print("   🎯 Translation invariance: Robust to position")
    print("   ⚡ Computation reduction: Faster networks")
    print()

    print("🔜 COMPLETE CNN ARCHITECTURE:")
    print("   🏗️ How to stack Conv + Pool layers")
    print("   🧠 How to connect to final classifier")
    print("   💻 Build complete digit recognizer")
    print("   🎨 Visualize what each layer sees")
    print()

    print("🎉 You're almost ready to build your first CNN!")


if __name__ == "__main__":
    print("🏊‍♂️ POOLING: The Smart AI Summarizer!")
    print("=" * 45)
    print("How AI learns 'approximately where' instead of 'exactly where'!")
    print()

    # What is pooling
    what_is_pooling()

    # Max pooling example
    max_pooling_example()

    # Average pooling
    average_pooling_example()

    # Translation invariance demo
    translation_invariance_demo()

    # Pooling benefits
    pooling_benefits()

    # Real-world analogy
    real_world_analogy()

    # Parameters explanation
    stride_and_size_explanation()

    # Connection to full CNN
    connection_to_full_cnn()

    # What's next
    whats_next()

    print("\n🌟 POOLING MASTERED!")
    print("You understand how AI becomes robust to variations!")
    print("Ready to build the COMPLETE CNN ARCHITECTURE? 🏗️✨")
