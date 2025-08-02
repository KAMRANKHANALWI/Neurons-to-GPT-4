"""
🧠 McCulloch-Pitts Neuron: Explained Like You're 5!
==================================================

Imagine you're deciding whether to go outside or not.
You look at TWO things:
1. Is it sunny? (Yes=1, No=0)
2. Is it warm? (Yes=1, No=0)

The THRESHOLD is like your personal rule:
"I'll only go outside if I have AT LEAST 2 good reasons"

Let's see what happens...
"""


def simple_decision_example():
    """
    The neuron is like making a simple decision!
    """
    print("🌤️  SHOULD I GO OUTSIDE? (Simple Decision Making)")
    print("=" * 55)

    print("My rule: I need AT LEAST 2 points to go outside")
    print("- Sunny weather = 1 point")
    print("- Warm weather = 1 point")
    print("- Threshold = 2 points (minimum needed)")
    print()

    # Test all weather combinations
    print("Let's check different weather conditions:")
    print("-" * 40)

    # Case 1: Not sunny, not warm
    sunny = 0
    warm = 0
    total_points = sunny + warm
    decision = "YES" if total_points >= 2 else "NO"

    print(f"🌧️  Cloudy and Cold:")
    print(f"   Sunny? {sunny} point")
    print(f"   Warm?  {warm} point")
    print(f"   Total: {total_points} points")
    print(f"   Need:  2 points minimum")
    print(f"   Go outside? {decision} (not enough points!)")
    print()

    # Case 2: Sunny but not warm
    sunny = 1
    warm = 0
    total_points = sunny + warm
    decision = "YES" if total_points >= 2 else "NO"

    print(f"☀️  Sunny but Cold:")
    print(f"   Sunny? {sunny} point")
    print(f"   Warm?  {warm} point")
    print(f"   Total: {total_points} points")
    print(f"   Need:  2 points minimum")
    print(f"   Go outside? {decision} (not enough points!)")
    print()

    # Case 3: Not sunny but warm
    sunny = 0
    warm = 1
    total_points = sunny + warm
    decision = "YES" if total_points >= 2 else "NO"

    print(f"🌤️  Cloudy but Warm:")
    print(f"   Sunny? {sunny} point")
    print(f"   Warm?  {warm} point")
    print(f"   Total: {total_points} points")
    print(f"   Need:  2 points minimum")
    print(f"   Go outside? {decision} (not enough points!)")
    print()

    # Case 4: Sunny AND warm
    sunny = 1
    warm = 1
    total_points = sunny + warm
    decision = "YES" if total_points >= 2 else "NO"

    print(f"☀️🌡️ Sunny AND Warm:")
    print(f"   Sunny? {sunny} point")
    print(f"   Warm?  {warm} point")
    print(f"   Total: {total_points} points")
    print(f"   Need:  2 points minimum")
    print(f"   Go outside? {decision} (enough points! 🎉)")


def coffee_shop_example():
    """
    Another simple example - deciding whether to visit a coffee shop
    """
    print("\n\n☕ SHOULD I GO TO THIS COFFEE SHOP?")
    print("=" * 45)

    print("My decision rule: I need AT LEAST 3 good things")
    print("Let me check 3 things:")
    print("- Good coffee? (Yes=2 points, No=0)")
    print("- Close to me? (Yes=1 point, No=0)")
    print("- Cheap prices? (Yes=1 point, No=0)")
    print("- Threshold = 3 points minimum")
    print()

    # Different coffee shops
    shops = [
        ("☕ Starbucks", [2, 0, 0], "Good coffee, but far and expensive"),
        ("☕ Local Cafe", [0, 1, 1], "Bad coffee, but close and cheap"),
        ("☕ Perfect Place", [2, 1, 1], "Good coffee, close, and cheap!"),
    ]

    for shop_name, scores, description in shops:
        good_coffee, close, cheap = scores
        total_points = good_coffee + close + cheap
        decision = "YES" if total_points >= 3 else "NO"

        print(f"{shop_name} - {description}")
        print(f"   Good coffee? {good_coffee} points")
        print(f"   Close to me? {close} point")
        print(f"   Cheap prices? {cheap} point")
        print(f"   Total: {total_points} points")
        print(f"   Need: 3 points minimum")
        print(f"   Visit? {decision}")
        if decision == "YES":
            print("   🎉 This place meets my standards!")
        else:
            print("   ❌ Not good enough for me")
        print()


def explain_the_brain_connection():
    """
    Connect this back to how brain neurons work
    """
    print("\n🧠 HOW THIS RELATES TO YOUR BRAIN")
    print("=" * 35)

    print(
        """
    Your brain neurons work EXACTLY like this!
    
    Real Example: Deciding to eat something
    
    Your "eating neuron" gets signals:
    👃 Smell sensor: "Smells good!" = 1 point
    👁️  Vision sensor: "Looks tasty!" = 1 point  
    🕐 Time sensor: "It's lunch time!" = 1 point
    
    Your brain's threshold: "Need 2 good signals to eat"
    
    If you get 2 or more signals → You eat! (neuron fires = 1)
    If you get less than 2 → You don't eat (neuron stays quiet = 0)
    
    That's ALL a McCulloch-Pitts neuron does:
    1. Add up the input points
    2. Check if it's enough (threshold)
    3. Say YES (1) or NO (0)
    
    Simple counting and decision making!
    """
    )


def the_big_limitation():
    """
    Explain why this was limited and led to the next breakthrough
    """
    print("\n⚠️  THE BIG PROBLEM (Why This Wasn't Enough)")
    print("=" * 50)

    print(
        """
    The problem: These "points" (weights) are FIXED forever!
    
    Imagine if you could NEVER change your decision rules:
    - You decide "good coffee = 2 points" when you're 5 years old
    - You can NEVER change this rule your entire life!
    - Even if you try different coffee shops and learn
    - Even if your taste changes as you grow up
    
    That's the McCulloch-Pitts limitation:
    ❌ No learning
    ❌ No adapting  
    ❌ Rules are fixed forever
    
    The breakthrough question in 1957:
    💡 "What if the neuron could LEARN better point values?"
    💡 "What if it could change its rules based on experience?"
    
    This led to the PERCEPTRON - the first learning machine! 🚀
    """
    )


if __name__ == "__main__":
    print("🎯 Let's Understand the McCulloch-Pitts Neuron!")
    print("(No technical background needed!)")
    print("=" * 50)

    # Simple decision examples
    simple_decision_example()
    coffee_shop_example()

    # Connect to brain
    explain_the_brain_connection()

    # Explain the limitation
    the_big_limitation()

    print("🔜 NEXT: The Perceptron - A Neuron That Can LEARN!")
    print("(Finally, a machine that gets smarter with practice!)")
