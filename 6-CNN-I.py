"""
🔍 CNN FILTERS: How AI Learned to See Patterns!
===============================================

Think of filters as TINY PATTERN DETECTORS that slide across images
looking for specific features like edges, curves, and shapes!

Let's see EXACTLY how this works with simple examples!
"""

import numpy as np
import matplotlib.pyplot as plt


def what_is_a_filter():
    """
    The most intuitive explanation of CNN filters!
    """
    print("👁️ WHAT IS A CNN FILTER?")
    print("=" * 30)

    print("🎯 Think of a filter as a TINY PATTERN TEMPLATE!")
    print()
    print("Example: Vertical Edge Detector")
    print("Filter (3x3):")
    print("   [-1, 0, +1]")
    print("   [-1, 0, +1]")
    print("   [-1, 0, +1]")
    print()
    print("💡 This filter says:")
    print("   'I'm looking for dark-to-light transitions!'")
    print("   'Dark on left (-1), neutral middle (0), bright on right (+1)'")
    print()

    print("🔍 How it works:")
    print("   1. Place filter on image patch")
    print("   2. Multiply corresponding pixels")
    print("   3. Add up all results")
    print("   4. High sum = pattern found!")
    print("   5. Slide to next position")


def simple_edge_detection_example():
    """
    Show edge detection with actual numbers!
    """
    print("\n\n🧮 EDGE DETECTION: STEP BY STEP")
    print("=" * 40)

    print("🖼️ Simple image (5x5):")
    # Create simple image: dark left, bright right
    image = np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ]
    )

    print("   [0, 0, 1, 1, 1]")
    print("   [0, 0, 1, 1, 1]")
    print("   [0, 0, 1, 1, 1]")
    print("   [0, 0, 1, 1, 1]")
    print("   [0, 0, 1, 1, 1]")
    print("   (Dark left, bright right - perfect vertical edge!)")
    print()

    print("🔍 Vertical edge filter (3x3):")
    filter_edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    print("   [-1,  0, +1]")
    print("   [-1,  0, +1]")
    print("   [-1,  0, +1]")
    print()

    print("📊 CONVOLUTION PROCESS:")
    print("Let's place filter at position (1,1) - right at the edge!")
    print()

    # Extract 3x3 patch at position (1,1)
    patch = image[0:3, 0:3]
    print("Image patch:")
    print(f"   {patch}")
    print()

    print("Filter:")
    print(f"   {filter_edge}")
    print()

    print("🧮 Element-wise multiplication:")
    result_matrix = patch * filter_edge
    print(f"   {result_matrix}")
    print()

    print("➕ Sum all values:")
    result = np.sum(result_matrix)
    print(f"   Sum = {result}")
    print()

    if result > 2:
        print("🎉 HIGH VALUE! EDGE DETECTED! ✅")
    else:
        print("😐 Low value, no strong edge here")

    return image, filter_edge


def convolution_animation():
    """
    Show how filter slides across the image
    """
    print("\n\n🎬 FILTER SLIDING ANIMATION (Text Version)")
    print("=" * 50)

    print("Watch the filter slide across our image...")
    print()

    # Simple 4x4 image for easier visualization
    image = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])

    filter_edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    print("Original image (4x4):")
    for row in image:
        print(f"   {row}")
    print()

    print("Filter (3x3):")
    for row in filter_edge:
        print(f"   {row}")
    print()

    print("🎬 SLIDING PROCESS:")
    output = np.zeros((2, 2))  # Output will be 2x2

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    position_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]

    for idx, (i, j) in enumerate(positions):
        print(f"\n📍 Position {idx+1}: {position_names[idx]} ({i},{j})")

        # Extract patch
        patch = image[i : i + 3, j : j + 3]
        print("   Image patch:")
        for row in patch:
            print(f"      {row}")

        # Apply filter
        result_matrix = patch * filter_edge
        result = np.sum(result_matrix)
        output[i, j] = result

        print(f"   Convolution result: {result}")

        if result > 2:
            print("   🔥 STRONG EDGE DETECTED!")
        elif result > 0:
            print("   ⚡ Weak edge detected")
        else:
            print("   😐 No edge here")

    print(f"\n📊 FINAL OUTPUT (Feature Map):")
    print("   (Shows where edges were detected)")
    for row in output:
        print(f"   {row}")

    return output


def different_filter_types():
    """
    Show different types of filters and what they detect
    """
    print("\n\n🎨 DIFFERENT FILTER TYPES")
    print("=" * 30)

    filters = {
        "Vertical Edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        "Horizontal Edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        "Diagonal Edge": np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
        "Blur Filter": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,  # Normalize
        "Sharpen Filter": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    }

    for name, filter_array in filters.items():
        print(f"🔍 {name}:")
        for row in filter_array:
            print(f"   {row}")

        print("   💡 Detects:", end=" ")
        if "Vertical" in name:
            print("Vertical lines and edges")
        elif "Horizontal" in name:
            print("Horizontal lines and edges")
        elif "Diagonal" in name:
            print("Diagonal lines and edges")
        elif "Blur" in name:
            print("Smooths/blurs the image")
        elif "Sharpen" in name:
            print("Enhances edges and details")
        print()


def the_magic_insight():
    """
    The key insight about CNN filters
    """
    print("\n\n✨ THE MAGIC INSIGHT!")
    print("=" * 25)

    print("🎯 WHY FILTERS ARE REVOLUTIONARY:")
    print()

    print("1️⃣ PATTERN RECOGNITION:")
    print("   Each filter learns to detect ONE specific pattern")
    print("   'I'm the vertical edge detector!'")
    print("   'I'm the curve detector!'")
    print("   'I'm the corner detector!'")
    print()

    print("2️⃣ TRANSLATION INVARIANCE:")
    print("   Same filter works EVERYWHERE in the image")
    print("   Edge in top-left? ✅ Detected!")
    print("   Edge in bottom-right? ✅ Detected!")
    print("   'I find my pattern no matter where it is!'")
    print()

    print("3️⃣ PARAMETER SHARING:")
    print("   ONE filter = 9 numbers (3x3)")
    print("   Can process ENTIRE image!")
    print("   Regular neural net would need millions of weights!")
    print()

    print("4️⃣ HIERARCHICAL LEARNING:")
    print("   Layer 1: Simple patterns (edges)")
    print("   Layer 2: Complex patterns (shapes)")
    print("   Layer 3: Objects (faces, cars)")
    print("   'Building complexity step by step!'")


def real_world_analogy():
    """
    Real-world analogy for understanding filters
    """
    print("\n\n🌍 REAL-WORLD ANALOGY")
    print("=" * 25)

    print("🔍 CNN Filters = Specialized Detectives!")
    print()

    print("🕵️ Detective Smith (Vertical Edge Filter):")
    print("   'I only look for vertical lines!'")
    print("   'Tall buildings? I'll find them!'")
    print("   'Tree trunks? I'll spot them!'")
    print()

    print("🕵️ Detective Jones (Horizontal Edge Filter):")
    print("   'I only look for horizontal lines!'")
    print("   'Horizons? That's my specialty!'")
    print("   'Table edges? I'm on it!'")
    print()

    print("🕵️ Detective Brown (Curve Filter):")
    print("   'I only look for curves!'")
    print("   'Circles? I'll find them!'")
    print("   'Smiles? That's my job!'")
    print()

    print("👥 THE TEAM APPROACH:")
    print("   Each detective scans the ENTIRE scene")
    print("   Each reports what THEY found")
    print("   Combined reports = complete understanding!")
    print("   'Together, we see everything!' 🎯")


def connection_to_backprop():
    """
    Connect this back to what we learned about backpropagation
    """
    print("\n\n🔗 CONNECTION TO BACKPROPAGATION")
    print("=" * 35)

    print("🧠 Remember our network learning?")
    print("   Weights adjust based on errors...")
    print()

    print("🎯 CNN FILTERS ARE LEARNED WEIGHTS!")
    print("   Filter values = weights that get adjusted!")
    print("   Backpropagation teaches filters what patterns to detect!")
    print()

    print("📚 LEARNING PROCESS:")
    print("   1. Show CNN image of 'cat'")
    print("   2. CNN guesses 'dog' (wrong!)")
    print("   3. Error flows backward")
    print("   4. Filters adjust: 'We need to detect cat features better!'")
    print("   5. Repeat thousands of times")
    print("   6. Filters automatically learn:")
    print("      - Cat ear detector")
    print("      - Cat whisker detector")
    print("      - Cat eye detector")
    print()

    print("🤯 THE MIND-BLOWING TRUTH:")
    print("   We don't DESIGN the filters!")
    print("   The network LEARNS them automatically!")
    print("   Just like perceptron learned weights!")
    print("   But now weights detect visual patterns! 👁️✨")


def whats_next():
    """
    Preview what's coming next
    """
    print("\n\n🚀 WHAT'S NEXT?")
    print("=" * 15)

    print("✅ NOW YOU UNDERSTAND:")
    print("   🔍 Filters = Pattern detectors")
    print("   🧮 Convolution = Sliding window operation")
    print("   🎯 Each filter specializes in one pattern")
    print("   🧠 Filters are learned, not designed!")
    print()

    print("🔜 COMING UP:")
    print("   📊 POOLING: Smart summarization")
    print("   🏗️ CNN ARCHITECTURE: Complete system")
    print("   💻 CODE: Build digit recognizer")
    print("   🎨 VISUALIZATION: See what CNNs see")
    print()

    print("🎉 You're building the foundation of computer vision!")


if __name__ == "__main__":
    print("🔍 CNN FILTERS: The Eyes of Artificial Intelligence!")
    print("=" * 60)
    print("Discover how AI learned to see patterns!")
    print()

    # What is a filter?
    what_is_a_filter()

    # Simple edge detection
    simple_edge_detection_example()

    # Convolution process
    convolution_animation()

    # Different filter types
    different_filter_types()

    # The magic insight
    the_magic_insight()

    # Real-world analogy
    real_world_analogy()

    # Connection to backprop
    connection_to_backprop()

    # What's next
    whats_next()

    print("\n🌟 BREAKTHROUGH ACHIEVED!")
    print("You now understand the building blocks of computer vision!")
    print("Ready for POOLING and complete CNN architecture? 🏊‍♂️🏗️")
