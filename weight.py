"""
🎯 WHAT ARE WEIGHTS? The "Importance" Factor!
============================================

Think of weights like this:
Some things matter MORE to you than others when making decisions!

Example: Choosing a restaurant
- Good food: VERY important to you
- Close location: A little important
- Cheap price: Not very important (you have money!)

The WEIGHT tells the neuron: "How much should I care about this input?"
"""


def restaurant_example():
    """
    Perfect example of how weights work in real life
    """
    print("🍕 CHOOSING A RESTAURANT: Weight = Importance")
    print("=" * 50)

    print("You're choosing between restaurants.")
    print("You care about 3 things, but NOT EQUALLY:")
    print()
    print("📊 Your personal importance levels (weights):")
    print("   🍽️  Good food:     Weight = 3 (VERY important!)")
    print("   📍 Close to me:    Weight = 1 (a little important)")
    print("   💰 Cheap price:    Weight = 1 (a little important)")
    print("   🎯 Threshold:      5 points (minimum to choose)")
    print()

    restaurants = [
        ("🍔 Fast Food Place", [1, 1, 1], "Bad food, close, cheap"),
        ("🍝 Fancy Restaurant", [1, 0, 0], "Great food, far, expensive"),
        ("🍕 Perfect Pizza", [1, 1, 0], "Great food, close, expensive"),
    ]

    weights = [3, 1, 1]  # [food_weight, location_weight, price_weight]

    for name, inputs, description in restaurants:
        good_food, close, cheap = inputs

        print(f"{name} - {description}")
        print(
            f"   Good food? {good_food} × {weights[0]} = {good_food * weights[0]} points"
        )
        print(f"   Close?     {close} × {weights[1]} = {close * weights[1]} points")
        print(f"   Cheap?     {cheap} × {weights[2]} = {cheap * weights[2]} points")

        total = good_food * weights[0] + close * weights[1] + cheap * weights[2]
        decision = "YES! 🎉" if total >= 5 else "NO ❌"

        print(f"   Total: {total} points")
        print(f"   Choose it? {decision}")
        print()


def dating_example():
    """
    Another relatable example - what you look for in dating
    """
    print("\n💕 DATING EXAMPLE: What Matters Most to You?")
    print("=" * 45)

    print("You're on a dating app. You care about:")
    print("   😊 Funny:        Weight = 2 (pretty important)")
    print("   🎓 Smart:        Weight = 3 (VERY important!)")
    print("   😍 Attractive:   Weight = 1 (nice, but not everything)")
    print("   🎯 Threshold:    4 points (minimum to swipe right)")
    print()

    profiles = [
        ("Profile A", [1, 0, 1], "Funny and attractive, but not smart"),
        ("Profile B", [0, 1, 1], "Smart and attractive, but not funny"),
        ("Profile C", [1, 1, 0], "Funny and smart, but not attractive"),
    ]

    weights = [2, 3, 1]  # [funny_weight, smart_weight, attractive_weight]

    for name, inputs, description in profiles:
        funny, smart, attractive = inputs

        print(f"{name} - {description}")
        print(f"   Funny?      {funny} × {weights[0]} = {funny * weights[0]} points")
        print(f"   Smart?      {smart} × {weights[1]} = {smart * weights[1]} points")
        print(
            f"   Attractive? {attractive} × {weights[2]} = {attractive * weights[2]} points"
        )

        total = funny * weights[0] + smart * weights[1] + attractive * weights[2]
        decision = "Swipe RIGHT! 💕" if total >= 4 else "Swipe left 👈"

        print(f"   Total: {total} points")
        print(f"   Decision: {decision}")
        print()


def job_example():
    """
    Job selection example
    """
    print("\n💼 JOB SELECTION: What Do You Value Most?")
    print("=" * 45)

    print("You're choosing between job offers:")
    print("   💰 Good salary:    Weight = 2 (important)")
    print("   🏠 Work from home: Weight = 3 (VERY important to you!)")
    print("   👥 Nice team:      Weight = 1 (nice to have)")
    print("   🎯 Threshold:      4 points (minimum to accept)")
    print()

    jobs = [
        ("Tech Startup", [1, 0, 1], "Great salary, office only, nice team"),
        ("Remote Company", [0, 1, 1], "Okay salary, fully remote, nice team"),
        ("Dream Job", [1, 1, 0], "Great salary, fully remote, difficult team"),
    ]

    weights = [2, 3, 1]  # [salary_weight, remote_weight, team_weight]

    for name, inputs, description in jobs:
        salary, remote, team = inputs

        print(f"{name} - {description}")
        print(f"   Good salary? {salary} × {weights[0]} = {salary * weights[0]} points")
        print(f"   Remote work? {remote} × {weights[1]} = {remote * weights[1]} points")
        print(f"   Nice team?   {team} × {weights[2]} = {team * weights[2]} points")

        total = salary * weights[0] + remote * weights[1] + team * weights[2]
        decision = "ACCEPT! 🎉" if total >= 4 else "REJECT ❌"

        print(f"   Total: {total} points")
        print(f"   Decision: {decision}")
        print()


def simple_math_explanation():
    """
    Break down the math in the simplest way
    """
    print("\n🧮 THE SIMPLE MATH BEHIND IT")
    print("=" * 35)

    print("It's just like calculating your grade in school!")
    print()
    print("Your final grade = (Homework × 20%) + (Midterm × 30%) + (Final × 50%)")
    print()
    print("In neuron terms:")
    print("Output = (Input1 × Weight1) + (Input2 × Weight2) + (Input3 × Weight3)")
    print()
    print("Example:")
    print("   Input1 = 1 (yes), Weight1 = 3 (very important)")
    print("   Input2 = 0 (no),  Weight2 = 2 (important)")
    print("   Input3 = 1 (yes), Weight3 = 1 (a little important)")
    print()
    print("   Calculation: (1 × 3) + (0 × 2) + (1 × 1) = 3 + 0 + 1 = 4 points")
    print("   If threshold = 3, then 4 ≥ 3, so output = 1 (YES!)")


def the_key_insight():
    """
    The main point about weights
    """
    print("\n💡 THE KEY INSIGHT ABOUT WEIGHTS")
    print("=" * 35)

    print(
        """
    Weights = Your personal preferences!
    
    👤 Person A might care most about money (high weight on salary)
    👤 Person B might care most about family time (high weight on work-life balance)
    👤 Person C might care most about adventure (high weight on travel opportunities)
    
    Same inputs, different weights = different decisions!
    
    🤖 In AI: The neuron's "personality" is defined by its weights
    
    🚨 THE PROBLEM: In McCulloch-Pitts neurons, these weights are FIXED!
    You can never change what you care about, even if you learn from experience.
    
    🎯 That's why we needed the PERCEPTRON: A neuron that can LEARN and 
    update its weights based on experience!
    """
    )


if __name__ == "__main__":
    print("🎯 Understanding WEIGHTS: The Importance Factor")
    print("=" * 50)

    # Real-life examples
    restaurant_example()
    dating_example()
    job_example()

    # Simple math explanation
    simple_math_explanation()

    # Key insight
    the_key_insight()

    print("\n🔜 NEXT: The Perceptron - A Neuron That Can Change Its Mind!")
    print("(Finally! A neuron that can learn what's REALLY important)")
