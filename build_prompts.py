import csv
import json
import random
from pathlib import Path

BATCH_SIZE = 6   # prompts per chat message
SEED = 448       # fixed so everyone gets the same shuffle
MODELS = ["claude", "chatgpt", "gemini"]

PROMPTS = {
    "coding": [
        "Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case.",
        "Write a Python function that merges two sorted lists into one sorted list.",
        "Implement binary search in Python and explain its time complexity.",
        "Write a Python function that returns the n-th Fibonacci number efficiently.",
        "Write a Python class implementing a stack with push, pop, and peek.",
        "Write a JavaScript function that debounces another function.",
        "Write a SQL query that finds the second-highest salary from an Employees table.",
        "Write a Python function that counts word frequencies in a text file.",
        "Implement a queue using two stacks in Python.",
        "Write a Python function that detects whether a linked list has a cycle.",
        "Write a bash one-liner to find the 10 largest files in a directory, and explain it.",
        "Write a Python function that flattens an arbitrarily nested list.",
        "This function crashes when the list is empty. Explain why and fix it: def avg(xs): return sum(xs)/len(xs)",
        "Write a Python decorator that times how long a function takes to run.",
        "Implement quicksort in Python.",
        "Write a regular expression that validates a simple email address and explain each part.",
        "Write a Python function that finds all primes up to n using the Sieve of Eratosthenes.",
        "Write a C function that reverses a string in place.",
        "Write a Python function that converts a Roman numeral to an integer.",
        "Write a Java method that checks whether two strings are anagrams.",
        "Implement an LRU cache in Python.",
        "Write a Python script that reads a CSV file and prints the average of a numeric column.",
        "Write a function that checks whether the parentheses in a string are balanced.",
        "Explain the difference between a list and a tuple in Python and when to use each.",
        "Write a Python function that performs depth-first search on a graph given as an adjacency list.",
        "Write a Python function that computes the greatest common divisor of two numbers.",
        "Write a React component that shows a counter with increment and decrement buttons.",
        "Write a Python function that removes duplicates from a list while preserving order.",
        "Write a Python function that rotates a square matrix 90 degrees clockwise.",
        "Explain how a hash table works and implement a simple one in Python.",
    ],
    "math": [
        "Solve for x: 3x + 7 = 25.",
        "What is the derivative of x^3 * sin(x)?",
        "Evaluate the integral of 2x * e^(x^2) dx.",
        "A train travels 120 miles in 1.5 hours. What is its average speed in miles per hour?",
        "How many ways can 5 people be arranged in a row?",
        "Find the sum of the first 50 positive integers and show your reasoning.",
        "Solve the quadratic equation x^2 - 5x + 6 = 0.",
        "What is the probability of getting a sum of 8 when rolling two fair dice?",
        "Prove that the square root of 2 is irrational.",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "A shirt costs $40 after a 20% discount. What was its original price?",
        "What is the limit of sin(x)/x as x approaches 0, and why?",
        "Compute 17 x 23 without a calculator and show your steps.",
        "A rectangle has perimeter 36 and its length is twice its width. Find its area.",
        "How many positive divisors does 360 have?",
        "Solve the system: 2x + y = 7 and x - y = 2.",
        "What is the expected value of a fair six-sided die roll?",
        "Find the area under y = x^2 from x = 0 to x = 3.",
        "Explain the difference between permutations and combinations with an example.",
        "Simplify (x^2 - 9)/(x - 3).",
        "What is the sum of the infinite geometric series 1 + 1/2 + 1/4 + ...?",
        "Use induction to prove that 1 + 2 + ... + n = n(n+1)/2.",
        "A bag has 3 red and 5 blue marbles. Two are drawn without replacement. What is the probability both are red?",
        "Convert 0.375 to a fraction in lowest terms.",
        "Find the inverse of the function f(x) = 2x + 3.",
        "What is the remainder when 2^100 is divided by 7?",
        "If log base 2 of x equals 5, what is x?",
        "Find the distance between the points (1, 2) and (4, 6).",
        "Explain what standard deviation measures and compute it for the data 2, 4, 4, 4, 5, 5, 7, 9.",
        "What is the dot product of the vectors (1, 2, 3) and (4, 5, 6)?",
    ],
    "factual": [
        "Why is the sky blue?",
        "What caused the fall of the Western Roman Empire?",
        "How does photosynthesis work?",
        "What is the difference between weather and climate?",
        "Explain how vaccines train the immune system.",
        "What were the main causes of World War I?",
        "How does a blockchain work?",
        "Why do we have seasons?",
        "What is the function of mitochondria?",
        "Explain the difference between a virus and a bacterium.",
        "How do airplanes stay in the air?",
        "What is inflation and what causes it?",
        "Who was Ada Lovelace and why is she important?",
        "How does GPS determine your location?",
        "What is the greenhouse effect?",
        "Explain how the electoral college works in the United States.",
        "What is the difference between DNA and RNA?",
        "How do ocean tides work?",
        "What was the Industrial Revolution and how did it change society?",
        "Explain how a neural network learns.",
        "What causes earthquakes?",
        "How does the stock market work?",
        "What is the difference between a democracy and a republic?",
        "How do noise-cancelling headphones work?",
        "What is the theory of evolution by natural selection?",
        "Why is the ocean salty?",
        "What is the Pythagorean theorem and where is it used?",
        "How does the internet send data between computers?",
        "What is the water cycle?",
        "What is the difference between machine learning and traditional programming?",
    ],
    "writing": [
        "Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle.",
        "Write three original haikus about autumn.",
        "Write a professional email asking my professor for an extension on an assignment.",
        "Write a persuasive paragraph arguing that students should have later school start times.",
        "Write a product description for a reusable water bottle.",
        "Write a short poem about the first snowfall of winter.",
        "Write the opening paragraph of a cover letter for a software engineering internship.",
        "Write a dialogue between a robot and a child meeting for the first time.",
        "Describe a bustling night market in vivid detail in one paragraph.",
        "Write a motivational speech of about 150 words for a team before a big game.",
        "Write a limerick about a cat who loves coffee.",
        "Write a short review of a fictional restaurant called The Copper Spoon.",
        "Write a thank-you note to a neighbor who watched my dog.",
        "Write the opening paragraph of a mystery novel set in a small coastal town.",
        "Write a LinkedIn post announcing that I finished my first machine learning course.",
        "Write a short bedtime story about a dragon who is afraid of fire.",
        "Write an apology message to a friend for missing their birthday party.",
        "Write an argumentative paragraph for or against homework, your choice.",
        "Write a brief toast for my sister's wedding.",
        "Write a short sci-fi scene where an astronaut hears a knock on the airlock.",
        "Write a polite complaint to a landlord about a broken heater.",
        "Write a two-paragraph travel blog intro about a weekend in Lisbon.",
        "Write a short tribute to a beloved old car.",
        "Write a funny out-of-office auto-reply message.",
        "Write a short speech introducing a guest lecturer on climate science.",
        "Write a free-verse poem about the feeling of finishing an exam.",
        "Write a short original fable with a moral about a fox and a crow.",
        "Write a job posting for a part-time barista.",
        "Write a short diary entry from the perspective of a houseplant.",
        "Write a pitch for a mobile app that helps people find study partners.",
    ],
}

HEADER = (
    "I'm going to give you {n} separate prompts. Answer each one independently, "
    "exactly as you normally would if it were the only message in our conversation. "
    "Begin each answer with a line containing exactly === ANSWER k === "
    "(where k is the prompt number), and write nothing before the first marker.\n\n"
)


def main():
    data = Path("data")
    (data / "batches").mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        (data / "raw" / m).mkdir(parents=True, exist_ok=True)

    rows, pid = [], 0
    for cat, plist in PROMPTS.items():
        for p in plist:
            pid += 1
            rows.append({"prompt_id": pid, "category": cat, "prompt": p})

    with open(data / "prompts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "category", "prompt"])
        w.writeheader()
        w.writerows(rows)

    random.Random(SEED).shuffle(rows)  # mix categories inside each batch
    batches = {}
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i:i + BATCH_SIZE]
        name = f"{i // BATCH_SIZE + 1:02d}"
        batches[name] = [r["prompt_id"] for r in chunk]
        body = HEADER.format(n=len(chunk))
        body += "\n\n".join(f"PROMPT {k}: {r['prompt']}" for k, r in enumerate(chunk, 1))
        (data / "batches" / f"batch_{name}.txt").write_text(body + "\n", encoding="utf-8")

    (data / "batches.json").write_text(json.dumps(batches, indent=1), encoding="utf-8")
    print(f"{len(rows)} prompts -> {len(batches)} batches in data/batches/")
    print("Now paste each batch into each chatbot (see README.md).")


if __name__ == "__main__":
    main()
