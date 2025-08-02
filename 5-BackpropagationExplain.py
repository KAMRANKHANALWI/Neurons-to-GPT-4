"""
🎓 BACKPROPAGATION: The School Grade Analogy
===========================================

Imagine you got a BAD GRADE on your final report card.
Your parents want to know: "WHO'S TO BLAME?"

Let's see how blame flows backwards through your "learning network"!
"""


def school_grade_analogy():
    """
    The PERFECT analogy for understanding backpropagation!
    """
    print("🎓 THE SCHOOL GRADE DISASTER!")
    print("=" * 35)

    print("Your final grade: D- (2.0/4.0) 😱")
    print("Your parents expected: A (4.0/4.0) 📚")
    print("ERROR = Expected - Actual = 4.0 - 2.0 = 2.0 points too low!")
    print()

    print("🤔 Your parents ask: 'WHO'S TO BLAME for this disaster?'")
    print()

    print("🏗️ YOUR 'LEARNING NETWORK':")
    print("   📊 FINAL GRADE (Output Layer)")
    print("        ↑")
    print("   📚 SUBJECTS (Hidden Layer)")
    print("        ↑")
    print("   🎯 STUDY HABITS (Input Layer)")
    print()


def forward_pass_school():
    """
    How your grade was calculated (forward pass)
    """
    print("📈 FORWARD PASS: How Your Grade Was Calculated")
    print("=" * 50)

    print("🎯 INPUTS (Your study habits):")
    study_time = 2  # hours per day (should be more!)
    attend_class = 1  # do you attend? (1=yes, 0=no)
    do_homework = 0  # do you do homework? (0=rarely!)

    print(f"   Study time: {study_time} hours/day")
    print(f"   Attend class: {attend_class} (yes)")
    print(f"   Do homework: {do_homework} (rarely!)")
    print()

    print("📚 HIDDEN LAYER (Subject grades):")
    print("Each subject gets influenced by your study habits...")

    # Math grade calculation
    math_influence = study_time * 0.3 + attend_class * 0.4 + do_homework * 0.8
    math_grade = min(4.0, max(0.0, math_influence))  # cap between 0-4

    # Science grade calculation
    science_influence = study_time * 0.5 + attend_class * 0.3 + do_homework * 0.6
    science_grade = min(4.0, max(0.0, science_influence))

    print(f"   📐 Math grade: {math_grade:.1f}/4.0")
    print(
        f"      Calculation: {study_time}×0.3 + {attend_class}×0.4 + {do_homework}×0.8 = {math_influence:.1f}"
    )
    print(f"   🔬 Science grade: {science_grade:.1f}/4.0")
    print(
        f"      Calculation: {study_time}×0.5 + {attend_class}×0.3 + {do_homework}×0.6 = {science_influence:.1f}"
    )
    print()

    print("📊 OUTPUT LAYER (Final grade):")
    final_grade = math_grade * 0.6 + science_grade * 0.4  # Math weighted more
    print(f"   Final Grade = Math×0.6 + Science×0.4")
    print(
        f"   Final Grade = {math_grade:.1f}×0.6 + {science_grade:.1f}×0.4 = {final_grade:.1f}/4.0"
    )
    print(f"   Result: {final_grade:.1f}/4.0 = D- 😱")

    return {
        "inputs": [study_time, attend_class, do_homework],
        "hidden": [math_grade, science_grade],
        "output": final_grade,
        "target": 4.0,
        "error": 4.0 - final_grade,
    }


def backward_pass_school(results):
    """
    How blame flows backwards (the magic!)
    """
    print("\n🔙 BACKWARD PASS: The Blame Game!")
    print("=" * 40)

    error = results["error"]
    print(f"📊 STEP 1: Output Layer Error")
    print(f"   Total disappointment: {error:.1f} points too low!")
    print("   Your parents: 'This is unacceptable! 😤'")
    print()

    print(f"📚 STEP 2: Hidden Layer Errors (Blame the subjects)")
    print("   Now we ask: 'How much is EACH SUBJECT to blame?'")
    print()

    # Distribute blame to subjects based on their weights
    math_weight = 0.6  # Math counts 60% of final grade
    science_weight = 0.4  # Science counts 40% of final grade

    math_blame = error * math_weight
    science_blame = error * science_weight

    print(f"   📐 Math's blame: {error:.1f} × {math_weight} = {math_blame:.1f}")
    print(f"      'Math hurt your grade MORE because it's weighted more!'")
    print(
        f"   🔬 Science's blame: {error:.1f} × {science_weight} = {science_blame:.1f}"
    )
    print(f"      'Science hurt less because it's weighted less'")
    print()

    print(f"🎯 STEP 3: Input Layer Errors (Blame your habits)")
    print("   Now we ask: 'Which STUDY HABITS caused each subject to fail?'")
    print()

    # Math subject blames your habits
    print(f"   📐 Math subject says:")
    study_time_blame_math = math_blame * 0.3
    attend_blame_math = math_blame * 0.4
    homework_blame_math = math_blame * 0.8

    print(
        f"      Study time blame: {math_blame:.1f} × 0.3 = {study_time_blame_math:.1f}"
    )
    print(f"      Attendance blame: {math_blame:.1f} × 0.4 = {attend_blame_math:.1f}")
    print(f"      Homework blame: {math_blame:.1f} × 0.8 = {homework_blame_math:.1f}")
    print(f"      'Homework hurt me the MOST!' 📝")
    print()

    # Science subject blames your habits
    print(f"   🔬 Science subject says:")
    study_time_blame_science = science_blame * 0.5
    attend_blame_science = science_blame * 0.3
    homework_blame_science = science_blame * 0.6

    print(
        f"      Study time blame: {science_blame:.1f} × 0.5 = {study_time_blame_science:.1f}"
    )
    print(
        f"      Attendance blame: {science_blame:.1f} × 0.3 = {attend_blame_science:.1f}"
    )
    print(
        f"      Homework blame: {science_blame:.1f} × 0.6 = {homework_blame_science:.1f}"
    )
    print(f"      'Study time hurt me the MOST!' ⏰")
    print()

    # Total blame for each habit
    total_study_blame = study_time_blame_math + study_time_blame_science
    total_attend_blame = attend_blame_math + attend_blame_science
    total_homework_blame = homework_blame_math + homework_blame_science

    print(f"🎯 TOTAL BLAME FOR EACH HABIT:")
    print(f"   Study time total blame: {total_study_blame:.1f}")
    print(f"   Attendance total blame: {total_attend_blame:.1f}")
    print(f"   Homework total blame: {total_homework_blame:.1f}")
    print()

    return {
        "study_blame": total_study_blame,
        "attend_blame": total_attend_blame,
        "homework_blame": total_homework_blame,
    }


def learning_improvements(blame_results):
    """
    How you improve based on the blame (weight updates)
    """
    print("🚀 STEP 4: LEARNING IMPROVEMENTS!")
    print("=" * 35)

    print("Based on the blame analysis, here's how you should change:")
    print()

    learning_rate = 0.1  # Don't change TOO drastically

    current_study = 2
    new_study = current_study + (learning_rate * blame_results["study_blame"])

    current_homework = 0
    new_homework = min(
        1, current_homework + (learning_rate * blame_results["homework_blame"])
    )

    print(f"📚 STUDY TIME:")
    print(f"   Current: {current_study} hours/day")
    print(f"   Adjustment: +{learning_rate * blame_results['study_blame']:.1f} hours")
    print(f"   New target: {new_study:.1f} hours/day")
    print(f"   'I need to study MORE!' 📖")
    print()

    print(f"📝 HOMEWORK:")
    print(f"   Current: {current_homework} (rarely do it)")
    print(f"   Adjustment: +{learning_rate * blame_results['homework_blame']:.2f}")
    print(f"   New target: {new_homework:.2f} (start doing it!)")
    print(f"   'Homework has the BIGGEST impact!' ✍️")
    print()

    print(f"🏫 ATTENDANCE:")
    print(f"   Current: 1 (already attending)")
    print(f"   'Keep attending class - that's not the problem!' ✅")


def the_magic_revealed():
    """
    The key insight about backpropagation
    """
    print("\n\n✨ THE MAGIC REVEALED!")
    print("=" * 25)

    print("🎯 WHAT JUST HAPPENED:")
    print("   1. Started with a BAD RESULT (low grade)")
    print("   2. AUTOMATICALLY figured out what caused it")
    print("   3. AUTOMATICALLY calculated how much each factor contributed")
    print("   4. AUTOMATICALLY determined how to improve")
    print()

    print("🧠 THIS IS EXACTLY HOW BACKPROPAGATION WORKS:")
    print("   1. Forward pass: Calculate prediction")
    print("   2. Calculate error: How wrong were we?")
    print("   3. Backward pass: Distribute blame to each layer")
    print("   4. Update weights: Improve based on blame")
    print()

    print("🎭 THE BEAUTIFUL INSIGHT:")
    print("   💡 Each layer gets blamed PROPORTIONALLY to its influence!")
    print("   💡 Bigger influence = bigger blame = bigger change needed!")
    print("   💡 The network automatically learns what to fix!")
    print()

    print("🚀 REAL NEURAL NETWORKS:")
    print("   Instead of 'study habits → subjects → final grade'")
    print("   We have 'inputs → hidden neurons → output'")
    print("   But the blame distribution works EXACTLY THE SAME!")


def xor_connection():
    """
    Connect this back to solving XOR
    """
    print("\n\n🔶 HOW THIS SOLVES XOR!")
    print("=" * 25)

    print("Remember XOR was impossible for single perceptrons?")
    print("Now with backpropagation:")
    print()

    print("🎯 XOR Network Structure:")
    print("   Inputs: [A, B] → Hidden neurons → Output")
    print()

    print("🧠 Learning Process:")
    print("   1. Try [0,1] → should output 1, but outputs 0")
    print("   2. Error = 1 - 0 = 1 (we're wrong!)")
    print("   3. Blame flows backward:")
    print("      - Output layer: 'I need to be more positive!'")
    print("      - Hidden layer: 'What should WE change?'")
    print("      - Input weights: 'Adjust us to help!'")
    print("   4. Network automatically adjusts ALL weights")
    print("   5. Try again... getting better each time!")
    print()

    print("🎉 RESULT:")
    print("   The network DISCOVERS the XOR pattern by itself!")
    print("   No human hand-designing weights!")
    print("   Pure automatic learning! 🤖✨")


if __name__ == "__main__":
    print("🎓 BACKPROPAGATION: The School Grade Disaster!")
    print("=" * 55)
    print("The most intuitive explanation you'll ever see!")
    print()

    # Forward pass
    results = forward_pass_school()

    # Backward pass
    blame_results = backward_pass_school(results)

    # Learning improvements
    learning_improvements(blame_results)

    # The magic revealed
    the_magic_revealed()

    # XOR connection
    xor_connection()

    print("\n🌟 NOW YOU UNDERSTAND BACKPROPAGATION!")
    print("It's just automatic blame distribution + learning!")
    print("Ready to see it solve XOR automatically? 🚀")
