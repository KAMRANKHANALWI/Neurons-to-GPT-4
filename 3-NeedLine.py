"""
🎯 WHY Do We Need Lines? The Real Purpose Explained!
===================================================

You asked the PERFECT question! Let's understand WHY we draw lines
and what they're actually FOR in real life!

The answer: To CLASSIFY things! To separate GOOD from BAD!
"""

import matplotlib.pyplot as plt
import numpy as np


def the_real_world_purpose():
    """
    Explain why we need to separate things in real life
    """
    print("🌍 THE REAL-WORLD PURPOSE")
    print("=" * 30)

    print("🎯 The fundamental problem AI solves:")
    print("   'Given some information, make a DECISION!'")
    print()

    print("📊 Examples of decisions we make daily:")
    print("   🏥 Doctor: 'Is this patient SICK or HEALTHY?'")
    print("   📧 Email: 'Is this email SPAM or NOT SPAM?'")
    print("   💰 Bank: 'Should we APPROVE or REJECT this loan?'")
    print("   🚗 Self-driving car: 'Is that object a PEDESTRIAN or POLE?'")
    print("   🎬 Netflix: 'Will this person LIKE or DISLIKE this movie?'")
    print()

    print("💡 In ALL cases, we're separating things into TWO groups:")
    print("   Group A: ✅ (Good/Yes/Approve/Like)")
    print("   Group B: ❌ (Bad/No/Reject/Dislike)")
    print()

    print("🧠 The AI's job:")
    print("   'Draw a line to separate Group A from Group B!'")


def email_spam_example():
    """
    Show a concrete example: Email spam detection
    """
    print("\n\n📧 CONCRETE EXAMPLE: Email Spam Detection")
    print("=" * 45)

    print("🎯 Goal: Automatically detect spam emails")
    print()

    print("📊 We look at 2 features of each email:")
    print("   Feature 1: Number of words like 'FREE', 'WIN', 'MONEY'")
    print("   Feature 2: Number of exclamation marks!!!")
    print()

    print("📨 Training data (emails we already know):")

    # Sample email data
    emails = [
        ([0, 1], 0, "Normal: 'Hi, how are you?'"),
        ([1, 2], 0, "Normal: 'Free parking available!'"),
        ([2, 8], 1, "SPAM: 'FREE MONEY!!! WIN NOW!!!'"),
        ([5, 12], 1, "SPAM: 'FREE!!! WIN!!! MONEY!!! CLICK NOW!!!'"),
        ([0, 0], 0, "Normal: 'Meeting at 3pm'"),
        ([1, 1], 0, "Normal: 'Thanks! See you tomorrow'"),
        ([8, 15], 1, "SPAM: 'FREE WIN MONEY!!! URGENT!!!'"),
        ([3, 6], 1, "SPAM: 'FREE CASH!!! WIN BIG!!!'"),
    ]

    for features, is_spam, description in emails:
        spam_words, exclamations = features
        label = "SPAM" if is_spam else "NORMAL"
        print(f"      {spam_words} spam words, {exclamations} exclamations → {label}")
        print(f"         '{description}'")
    print()

    print("🤔 The challenge:")
    print("   New email arrives: [4, 9] (4 spam words, 9 exclamations)")
    print("   Is it SPAM or NORMAL?")
    print()

    print("🧠 Perceptron's solution:")
    print("   'Let me draw a LINE to separate spam from normal!'")
    print("   If new email is ABOVE the line → SPAM")
    print("   If new email is BELOW the line → NORMAL")


def visualize_email_classification():
    """
    Create a visual of email classification
    """
    print("\n\n📊 VISUALIZING EMAIL CLASSIFICATION")
    print("=" * 40)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Email Spam Classification: Before and After Learning", fontsize=14)

    # Email data points
    normal_emails = np.array([[0, 1], [1, 2], [0, 0], [1, 1]])
    spam_emails = np.array([[2, 8], [5, 12], [8, 15], [3, 6]])

    # Plot 1: Raw data (before learning)
    ax1.scatter(
        normal_emails[:, 0],
        normal_emails[:, 1],
        s=150,
        c="green",
        marker="o",
        label="Normal Email ✅",
        edgecolors="black",
    )
    ax1.scatter(
        spam_emails[:, 0],
        spam_emails[:, 1],
        s=150,
        c="red",
        marker="s",
        label="Spam Email ❌",
        edgecolors="black",
    )

    # Add labels to points
    for i, point in enumerate(normal_emails):
        ax1.annotate(
            f"Normal\n{point}",
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    for i, point in enumerate(spam_emails):
        ax1.annotate(
            f"Spam\n{point}",
            point,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax1.set_xlabel("Number of Spam Words")
    ax1.set_ylabel("Number of Exclamation Marks")
    ax1.set_title("Before Learning: Mixed Up Data")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 9)
    ax1.set_ylim(-1, 16)

    # Plot 2: With decision boundary (after learning)
    ax2.scatter(
        normal_emails[:, 0],
        normal_emails[:, 1],
        s=150,
        c="green",
        marker="o",
        label="Normal Email ✅",
        edgecolors="black",
    )
    ax2.scatter(
        spam_emails[:, 0],
        spam_emails[:, 1],
        s=150,
        c="red",
        marker="s",
        label="Spam Email ❌",
        edgecolors="black",
    )

    # Draw decision line (learned by perceptron)
    # Let's say perceptron learned: 2×spam_words + 1×exclamations = 6
    x_line = np.linspace(0, 8, 100)
    y_line = 6 - 2 * x_line  # Rearranged: exclamations = 6 - 2×spam_words
    ax2.plot(x_line, y_line, "blue", linewidth=3, label="Decision Line")

    # Color regions
    xx, yy = np.meshgrid(np.linspace(-1, 9, 50), np.linspace(-1, 16, 50))
    Z = 2 * xx + yy - 6  # The learned formula
    ax2.contourf(xx, yy, Z >= 0, levels=[0.5, 1.5], colors=["lightcoral"], alpha=0.3)
    ax2.contourf(xx, yy, Z < 0, levels=[-1.5, -0.5], colors=["lightgreen"], alpha=0.3)

    # Test new email
    new_email = [4, 9]
    ax2.scatter(
        new_email[0],
        new_email[1],
        s=200,
        c="yellow",
        marker="*",
        label="New Email (Unknown)",
        edgecolors="black",
        linewidth=2,
    )
    ax2.annotate(
        "New Email\n[4, 9]\nSPAM!",
        new_email,
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"),
    )

    ax2.set_xlabel("Number of Spam Words")
    ax2.set_ylabel("Number of Exclamation Marks")
    ax2.set_title("After Learning: Line Separates Spam from Normal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 9)
    ax2.set_ylim(-1, 16)

    plt.tight_layout()
    plt.show()

    print("👁️ WHAT YOU'RE SEEING:")
    print("   Left: Raw data - spam and normal emails mixed together")
    print("   Right: Perceptron drew a line to separate them!")
    print("   Blue line: The decision boundary")
    print("   Green region: 'Normal email' region")
    print("   Red region: 'Spam email' region")
    print("   Yellow star: New email → Falls in red region → SPAM!")


def medical_diagnosis_example():
    """
    Another example: Medical diagnosis
    """
    print("\n\n🏥 ANOTHER EXAMPLE: Medical Diagnosis")
    print("=" * 40)

    print("🎯 Goal: Diagnose heart disease")
    print()

    print("📊 We measure 2 things:")
    print("   Feature 1: Resting heart rate")
    print("   Feature 2: Blood pressure")
    print()

    print("👥 Training data (patients we already diagnosed):")
    patients = [
        ([65, 120], 0, "Healthy patient"),
        ([70, 130], 0, "Healthy patient"),
        ([95, 180], 1, "Heart disease"),
        ([100, 190], 1, "Heart disease"),
        ([60, 110], 0, "Healthy patient"),
        ([88, 175], 1, "Heart disease"),
    ]

    for features, has_disease, description in patients:
        heart_rate, blood_pressure = features
        diagnosis = "DISEASE" if has_disease else "HEALTHY"
        print(
            f"   Heart rate: {heart_rate}, Blood pressure: {blood_pressure} → {diagnosis}"
        )

    print()
    print("🏥 New patient arrives:")
    print("   Heart rate: 85, Blood pressure: 165")
    print("   Diagnosis: ???")
    print()

    print("🧠 Perceptron draws a line:")
    print("   Patients ABOVE line → Heart disease")
    print("   Patients BELOW line → Healthy")
    print("   New patient → Above line → HEART DISEASE!")


def why_coordinate_geometry_matters():
    """
    Explain why we use coordinate geometry
    """
    print("\n\n📐 WHY COORDINATE GEOMETRY IN CODING?")
    print("=" * 40)

    print("🤔 You asked: 'Why coordinate geometry in coding?'")
    print()

    print("💡 Because EVERYTHING can be plotted as points!")
    print()

    print("🏠 House price prediction:")
    print("   x-axis: Size of house (sq ft)")
    print("   y-axis: Number of bedrooms")
    print("   Decision: Expensive or Cheap?")
    print()

    print("🎵 Music recommendation:")
    print("   x-axis: How much bass?")
    print("   y-axis: How fast is tempo?")
    print("   Decision: Will user like it?")
    print()

    print("📈 Stock trading:")
    print("   x-axis: Stock price change (%)")
    print("   y-axis: Trading volume")
    print("   Decision: Buy or Sell?")
    print()

    print("🎯 The pattern:")
    print("   1. Convert real-world problem → numbers")
    print("   2. Plot numbers as points on graph")
    print("   3. Draw line to separate good from bad")
    print("   4. Use line to classify new points!")


def the_fundamental_insight():
    """
    The key insight about classification
    """
    print("\n\n🧠 THE FUNDAMENTAL INSIGHT")
    print("=" * 30)

    print("🎯 Every AI classification problem is:")
    print("   'Given some measurements, put things into groups'")
    print()

    print("📊 The steps:")
    print("   1. MEASURE things (features)")
    print("   2. PLOT them as points")
    print("   3. SEPARATE groups with a line")
    print("   4. CLASSIFY new things using the line")
    print()

    print("🤯 This is why geometry matters in coding:")
    print("   Geometry = The language of separation!")
    print("   Lines = The tools of classification!")
    print()

    print("💡 Real insight:")
    print("   We're not 'doing math for fun'")
    print("   We're 'using math to make smart decisions!'")


def why_xor_breaks_this():
    """
    Connect back to why XOR breaks this system
    """
    print("\n\n💥 WHY XOR BREAKS THIS BEAUTIFUL SYSTEM")
    print("=" * 45)

    print("🔶 XOR pattern in real world:")
    print("   'Take umbrella if it's raining OR sunny, but NOT both'")
    print("   [Not raining, Not sunny] → No umbrella")
    print("   [Not raining, Sunny] → Take umbrella")
    print("   [Raining, Not sunny] → Take umbrella")
    print("   [Raining, Sunny] → No umbrella (weird weather!)")
    print()

    print("📊 When plotted:")
    print("   ❌ No umbrella: [0,0] and [1,1]")
    print("   ✅ Take umbrella: [0,1] and [1,0]")
    print()

    print("🚫 The problem:")
    print("   They're in a checkerboard pattern!")
    print("   No single line can separate them!")
    print()

    print("💡 This is why we needed Multi-Layer Networks:")
    print("   Some real-world patterns are too complex for one line!")
    print("   We needed multiple lines working together!")


def key_takeaways():
    """
    Summarize the key points
    """
    print("\n\n🎯 KEY TAKEAWAYS")
    print("=" * 20)

    print("✅ WHY LINES?")
    print("   To separate GOOD from BAD in real decisions!")
    print()

    print("✅ WHY COORDINATE GEOMETRY?")
    print("   Because everything can be measured and plotted!")
    print()

    print("✅ WHY CLASSIFICATION?")
    print("   Every smart decision is about grouping things!")
    print()

    print("✅ WHY PERCEPTRONS MATTER?")
    print("   They automatically find the best separating line!")
    print()

    print("✅ WHY XOR IS IMPORTANT?")
    print("   It showed us that some patterns need multiple lines!")
    print()

    print("🚀 THE BIG PICTURE:")
    print("   AI = Teaching computers to make smart decisions")
    print("   Lines = The simplest way to separate things")
    print("   Multiple lines = Can separate anything!")


if __name__ == "__main__":
    print("🎯 WHY Do We Need Lines? The Real Purpose!")
    print("=" * 50)
    print("Understanding the REAL reason behind all this math...")
    print()

    # The real-world purpose
    the_real_world_purpose()

    # Email spam example
    email_spam_example()

    # Visualize it
    visualize_email_classification()

    # Medical example
    medical_diagnosis_example()

    # Why coordinate geometry
    why_coordinate_geometry_matters()

    # The fundamental insight
    the_fundamental_insight()

    # Why XOR breaks this
    why_xor_breaks_this()

    # Key takeaways
    key_takeaways()

    print("\n🌟 NOW YOU UNDERSTAND:")
    print("Lines aren't just math - they're DECISION MAKERS!")
    print("Every line answers: 'Should I say YES or NO?'")
    print()
    print("🚀 Ready to see how multiple lines solve everything?")
