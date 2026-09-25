# -*- coding: utf-8 -*-
# ==================================================
#  NEXTGEN DISTILL — build YOUR OWN model (nextgen-trained)
#  Teachers: gpt-oss:20b + qwen3:14b  ->  generate data
#  Student:  Mistral-7B-Instruct-v0.3 (LoRA fine-tune)
#  Output:   Q4_K_M GGUF uploaded to Kaggle dataset
#  Runs on:  Kaggle GPU (T4) notebook/script kernel
# ==================================================
import json, os, re, subprocess, time, urllib.request, shutil, sys

VERCEL_URL = "https://nextgen-web-eta.vercel.app"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_BIN = "/usr/local/bin/ollama"
TEACHERS   = ["gpt-oss:20b", "qwen3:14b"]
OUT_JSONL  = "/kaggle/working/train.jsonl"
CKPT       = "/kaggle/working/gen_progress.json"
DS_DATA_ID = "kingking1111/nextgen-distill-data"
DS_MODEL_ID = "kingking1111/nextgen-model"
N_PROMPTS  = 440

# Parallel data-gen: Kaggle does CHUNK=1, each Colab account does CHUNK=2..6
N_CHUNKS = 6                      # total machines generating data
CHUNK = 1                         # this machine's chunk index (1..N_CHUNKS)
MERGE_PARTS = ["kingking1111/nextgen-distill-part2",
               "kingking1111/nextgen-distill-part3",
               "kingking1111/nextgen-distill-part4",
               "kingking1111/nextgen-distill-part5",
               "kingking1111/nextgen-distill-part6"]   # datasets from the Colab accounts
MERGE_WAIT_MIN = 420              # how many minutes to wait for the other chunks before training

# Explicit Kaggle credentials so uploads/downloads always hit the right account
# (kernel auto-auth is unreliable on some runtimes).
os.environ["KAGGLE_USERNAME"] = "kingking1111"
os.environ["KAGGLE_KEY"] = "57ffdad5a85b0d71d90ce867951a8a55"

def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

def sh(cmd, silent=True, timeout=3600):
    if not silent: print(">", cmd[:140], flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 and not silent:
            print(r.stderr[-400:], flush=True)
        return r
    except Exception as e:
        print("cmd failed:", e, flush=True); return None

def http(url, data=None, timeout=600):
    h = {"User-Agent": "Mozilla/5.0"}
    if data is not None: h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
        return json.loads(b) if b else {}

def ollama_chat(messages, model, max_tokens=240, temperature=0.7):
    data = {"model": model, "messages": messages, "stream": False,
            "options": {"num_ctx": 4096, "temperature": temperature, "num_predict": max_tokens}}
    r = http(OLLAMA_URL + "/api/chat", data)
    msg = r.get("message") or {}
    content = msg.get("content") or ""
    if not content and msg.get("reasoning"):
        content = msg["reasoning"]
    return content.strip()

# --------------------------------------------------
# 1) Install Ollama
# --------------------------------------------------
log("Installing Ollama...")
if not os.path.exists(OLLAMA_BIN):
    sh("apt-get update -qq && apt-get install -y -qq zstd", silent=False, timeout=600)
    sh("curl -fsSL -o /tmp/ollama.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst", silent=False, timeout=3600)
    sh("zstd -d -f /tmp/ollama.tar.zst -o /tmp/ollama.tar", silent=False, timeout=600)
    sh("tar -xf /tmp/ollama.tar -C /usr/local", silent=False, timeout=600)
    sh("chmod +x " + OLLAMA_BIN)
sh(OLLAMA_BIN + " --version", silent=False)
os.environ["PATH"] = "/usr/local/bin:" + os.environ.get("PATH", "")
serv_env = dict(os.environ); serv_env["OLLAMA_NUM_PARALLEL"] = "1"
subprocess.Popen([OLLAMA_BIN, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=serv_env)
for _ in range(30):
    try:
        http(OLLAMA_URL + "/api/version", timeout=3); break
    except Exception:
        time.sleep(2)
log("Ollama ready.")

# --------------------------------------------------
# 2) Pull teacher models
# --------------------------------------------------
for m in TEACHERS:
    log("Pulling teacher", m, "...")
    sh(OLLAMA_BIN + " pull " + m, silent=False)

# --------------------------------------------------
# 3) Build prompt list
# --------------------------------------------------
def build_prompts():
    P = []
    cur = [
        "Explain the water cycle to a 10 year old.",
        "What is the difference between HTTP and HTTPS?",
        "Write a Python function that checks if a string is a palindrome.",
        "Explain how a neural network learns.",
        "Translate 'Good morning, how are you today?' into Tamil.",
        "Write a short poem about the ocean.",
        "Give me a step-by-step plan to start a small online business.",
        "What are the symptoms of vitamin D deficiency?",
        "Explain what a database index is and why it matters.",
        "Write a SQL query to find duplicate emails in a users table.",
        "Debug this Python code and explain the bug: def f(): return x + 1",
        "How do I calm my mind before an exam?",
        "Explain compound interest with a simple example.",
        "Write a professional email to a client about a delayed delivery.",
        "What is the capital of Australia and what is it known for?",
        "Explain blockchain in simple terms.",
        "Write JavaScript to fetch data from an API and render a list.",
        "What are the best practices for a secure REST API?",
        "Summarize the plot of Romeo and Juliet in 3 sentences.",
        "Give me 5 tips to improve my English speaking.",
        "Explain the difference between TCP and UDP.",
        "Write a React component that shows a counter with buttons.",
        "How does a refrigerator keep food cold?",
        "What is machine learning overfitting?",
        "Create a weekly workout plan for beginners.",
        "Write a Python script to read a CSV file and print statistics.",
        "Explain inflation and why prices rise.",
        "How do I write a good resume summary?",
        "Translate 'I love programming' into Hindi.",
        "What is photosynthesis?",
        "Write an essay outline about the importance of reading.",
        "Explain Git branches and merge conflicts.",
        "Give me ideas for a birthday party for a 12 year old.",
        "What is the fastest way to learn programming basics?",
        "Write a SQL statement to create a users table with email and age.",
        "Explain the concept of karma simply.",
        "How do vaccines work?",
        "Write a bash command to find the largest file in a folder.",
        "What are the differences between Python lists and tuples?",
        "Explain how to use the Django ORM to query a model.",
        "Write a function to sort a dictionary by its values.",
        "How do I improve my memory and concentration?",
        "Explain the stock market to a beginner.",
        "Write a Python program that prints the Fibonacci series.",
        "What is the rule of thirds in photography?",
        "Translate 'Where is the nearest hospital?' into Telugu.",
        "Explain why the sky is blue.",
        "Give me a recipe for a simple tomato pasta.",
        "Write a Node.js Express route that returns JSON.",
        "How does two-factor authentication work?",
        "Explain what a VPN does.",
        "Write a persuasive paragraph about recycling.",
        "What is the difference between a job and a career?",
        "Explain how solar panels work.",
        "Write a Python function to reverse a linked list.",
        "How do I set up a home network securely?",
        "Explain mindfulness in 3 sentences.",
        "Write an HTML page with a form and a submit button.",
        "What are macros and micros in nutrition?",
        "Explain the water purification process.",
        "Write a C program to add two numbers.",
        "How does search engine ranking work?",
        "Explain the difference between empathy and sympathy.",
        "Write a TypeScript function with type annotations.",
        "How do I save money as a student?",
        "Explain what an API is with an analogy.",
        "Write a Java method that reverses a string.",
        "What is the greenhouse effect?",
        "Explain the 80/20 rule.",
        "Write a Python decorator that times a function.",
        "How do I prepare for a job interview?",
        "Explain recursion with a simple example.",
        "Write an SQL query to join two tables.",
        "What causes seasons on Earth?",
        "Explain the difference between Save and Save As.",
        "Write a responsive CSS button with hover effect.",
        "How does a search engine crawler work?",
        "Explain the terms debit and credit.",
        "Write a Go program that prints hello world.",
        "What are the benefits of drinking water?",
        "Explain how SSL certificates work.",
        "Write a Python class for a bank account with deposit and withdraw.",
        "How do I build a personal brand?",
        "Explain the big bang theory simply.",
        "Write a regex to validate an email address.",
        "What is the difference between a switch and a router?",
        "Explain how to make a good first impression.",
        "Write a bash loop that renames all .txt files to .md.",
        "What is the Pythagorean theorem?",
        "Explain the difference between AI and ML.",
        "Write a Flask app with a /hello route.",
        "How do I stop procrastinating?",
        "Explain the 5-second rule myth.",
        "Write a Python function that counts word frequencies.",
        "What is a metaphor? Give 3 examples.",
        "Explain how airplanes fly.",
        "Write a SQL query for top 5 highest paid employees.",
        "How does cloud storage work?",
        "Explain the difference between wants and needs.",
        "Write a Kotlin program that prints numbers 1 to 10.",
        "What is the speed of light?",
        "Explain how to use CSS flexbox for a simple layout.",
        "Write a Python script to download an image from a URL.",
        "How do I write a thank-you note?",
        "Explain the concept of supply and demand.",
        "Write a shell command to show disk usage per folder.",
        "What is a vector?",
        "Explain how a QR code works.",
        "Write a Django model for a blog post with title and body.",
        "How do I get better sleep?",
        "Explain the term 'opportunity cost'.",
        "Write a JavaScript function to debounce input.",
        "What is the difference between RAM and ROM?",
        "Explain how to write a business plan summary.",
        "Write a Python program using threading.",
        "How do search suggestions work?",
        "Explain the importance of saving for emergencies.",
        "Write a React hook that fetches data.",
        "What is the theory of evolution?",
        "Explain how to make an effective presentation.",
        "Write a bash script that backs up a directory.",
        "How do I reduce screen time?",
        "Explain what a firewall does.",
        "Write a Python lambda that squares a number.",
        "What are the 7 wonders of the modern world?",
        "Explain how to negotiate a salary.",
        "Write a CSS animation for a bouncing ball.",
        "How does a touchscreen work?",
        "Explain the difference between a virus and bacteria.",
        "Write a FastAPI endpoint with a GET request.",
        "How do I stay motivated while studying?",
        "Explain the principle of least privilege.",
        "Write a Python generator that yields even numbers.",
        "What is the function of the heart?",
        "Explain how to use map in JavaScript.",
        "Write a Dockerfile for a Python app.",
        "How do credit scores work?",
        "Explain the term 'liquidity'.",
        "Write a C++ program with a class and a method.",
        "What is the difference between climate and weather?",
        "Explain how to make homemade lemonade.",
        "Write a Python function that finds the max in a list.",
        "How do I learn a new language quickly?",
        "Explain the concept of variables in programming.",
        "Write an HTML table with 3 rows and 2 columns.",
        "What is the meaning of life in one paragraph?",
        "Explain how GPS works.",
        "Write a SQL query that groups sales by month.",
        "How do I organize my study notes?",
        "Explain the difference between a byte and a bit.",
        "Write a JavaScript promise example.",
        "What are good habits for a healthy lifestyle?",
        "Explain how to use reduce in JavaScript.",
        "Write a Python script that checks if a number is prime.",
        "How does online banking encryption work?",
        "Explain the term 'compound sentence' with examples.",
        "Write a bash command to count lines in a file.",
        "What is artificial general intelligence?",
        "Explain how to cook perfect rice.",
        "Write a Rust program that prints hello world.",
        "How do I build confidence?",
        "Explain the difference between a class and an object.",
        "Write a Vue component that shows a message.",
        "What is the most efficient way to study for exams?",
        "Explain how batteries work.",
        "Write a Python function to merge two sorted lists.",
        "How does a neural network recognize images?",
        "Explain the term 'Escape velocity'.",
        "Write an SQL query to update a user's email.",
        "What are the benefits of exercise?",
        "Explain how to handle errors in Python with try/except.",
        "Write a simple chatbot logic in Python.",
        "How do I create a budget?",
        "Explain the difference between data and information.",
        "Write a JavaScript function to shuffle an array.",
        "What is the best way to learn web development?",
        "Explain how rainbows form.",
        "Write a Python script to send an email using SMTP.",
        "How do I improve my typing speed?",
        "Explain the concept of time zones.",
        "Write a Laravel route example.",
        "What is the difference between equality and equity?",
        "Explain how to use git reset safely.",
        "Write a C# program that calculates the area of a circle.",
        "How does memory work in a computer?",
        "Explain the term 'burnout' and how to avoid it.",
        "Write a Python function using functools.lru_cache.",
        "What are the principles of good UI design?",
        "Explain how to make a paper airplane.",
        "Write a SQL query with a WHERE and ORDER BY.",
        "How do I deal with stress at work?",
        "Explain the difference between synchronous and asynchronous code.",
        "Write a Node.js script that reads a file.",
        "What is the fastest animal on land?",
        "Explain how to use environment variables in Python.",
        "Write a bash script that loops over files.",
        "How do I choose a career path?",
        "Explain the term 'blockchain' again in one sentence.",
        "Write a React form with validation.",
        "What are the effects of sleep deprivation?",
        "Explain how a microphone works.",
        "Write a Python program to compute factorial.",
        "How do I write cleaner code?",
        "Explain the difference between a bug and a glitch.",
        "Write a SQL query to delete old records.",
        "What is the boiling point of water and why?",
        "Explain how to give constructive feedback.",
        "Write a JavaScript class with a constructor.",
        "How do I use semicolons correctly?",
        "Explain the term 'tax' simply.",
        "Write a Python script that parses JSON.",
        "What are the best free resources to learn Python?",
        "Explain how airplanes stay in the air.",
        "Write an Arduino blink sketch.",
        "How do I improve communication skills?",
        "Explain the difference between a compiler and an interpreter.",
        "Write a Go function that returns a greeting.",
        "What is a black hole?",
        "Explain how to make tea properly.",
        "Write a Python function that removes duplicates.",
        "How do I set SMART goals?",
        "Explain the term 'AI hallucination'.",
        "Write a bash command to find a string in files.",
        "What are the stages of grief?",
        "Explain how to use CSS grid.",
        "Write a SQL query for a join with three tables.",
        "How do I practice typing with 10 fingers?",
        "Explain the difference between input and output devices.",
        "Write a Python decorator for authentication.",
        "What is the square root of 144?",
        "Explain how to keep a conversation going.",
        "Write a JavaScript function that returns the second largest number in an array.",
        "How do I choose between MongoDB and PostgreSQL?",
        "Explain the term 'serverless'.",
        "Write a Python program that generates a random password.",
        "What is the difference between a city and a town?",
        "Explain how to write a clear instruction.",
        "Write a shell script to monitor CPU usage.",
        "How do I learn from my mistakes?",
        "Explain the concept of time management.",
        "Write a Python function to flatten a nested list.",
        "What are the benefits of journaling?",
        "Explain how e-commerce payment gateways work.",
        "Write a SQL query to compute average salary per department.",
        "How do I make a good YouTube thumbnail?",
        "Explain the term 'deadlock' in databases.",
        "Write a Java program using a HashMap.",
        "What is the normal body temperature?",
        "Explain how to build a habit.",
        "Write a Python script that scrapes a web page title.",
        "How do I set up VS Code for Python?",
        "Explain the difference between static and dynamic typing.",
        "Write a React component with useState.",
        "What is the tallest mountain in the world?",
        "Explain how to write an apology email.",
        "Write a bash command to zip a folder.",
        "How do I avoid distractions while working?",
        "Explain the term 'unit test' with an example.",
        "Write a Python function that checks if two strings are anagrams.",
        "What is the difference between organic and inorganic food?",
        "Explain how to use async/await in Python.",
        "Write a CSS centered flexbox layout.",
        "How do I speak confidently in public?",
        "Explain the concept of a 'safe haven' in finance.",
        "Write a SQL query to count users by country.",
        "What are the basics of investing for beginners?",
        "Explain how a car engine works in simple terms.",
        "Write a Python script using pandas to show a dataframe head.",
        "How do I manage multiple deadlines?",
        "Explain the term 'sidebar' in web design.",
        "Write a JavaScript loop that prints odd numbers.",
        "What is the difference between a plan and a strategy?",
        "Explain how to create a strong password.",
        "Write a FastAPI app with a POST endpoint.",
        "How do I overcome fear of failure?",
        "Explain the term 'bandwidth'.",
        "Write a Python function that returns the most common word in a text.",
        "What are the keys to a long friendship?",
        "Explain how to debug code effectively.",
        "Write a bash script to check if a file exists.",
        "How do I plan a trip on a budget?",
        "Explain the difference between print and return in Python.",
        "Write a React app with a simple todo list.",
        "What is the largest ocean on Earth?",
        "Explain how to write a cover letter.",
        "Write a SQL query to add a column to a table.",
        "How do I improve my handwriting?",
        "Explain the term 'open source'.",
        "Write a Python program that simulates a dice roll.",
        "What are the health benefits of meditation?",
        "Explain how to use pointers in C.",
        "Write a JavaScript function to convert Celsius to Fahrenheit.",
        "How do I ask better questions?",
        "Explain the difference between a feature and a bug.",
        "Write a Django view that returns a template.",
        "What is the smallest country in the world?",
        "Explain how to use a map for navigation.",
        "Write a Python function that validates a phone number with regex.",
        "How do I set up two monitors?",
        "Explain the term 'customer retention'.",
        "Write a CSS card component.",
        "What are the three states of matter?",
        "Explain how to use breakpoints while debugging.",
        "Write a Node.js API with a GET and POST route.",
        "How do I make a good first website?",
        "Explain the difference between heat and temperature.",
        "Write a Python script that prints the current date and time.",
        "What is the purpose of a resume objective?",
        "Explain how to use decorators in practice.",
        "Write a SQL query for a full outer join.",
        "How do I keep plants alive indoors?",
        "Explain the term 'machine translation'.",
        "Write a Java program that reads user input.",
        "What are the signs of dehydration?",
        "Explain how to write effective meeting notes.",
        "Write a bash script that greets the user by name.",
        "How do I handle constructive criticism?",
        "Explain the concept of 'practicing gratitude'.",
        "Write a Python function to transpose a matrix.",
        "What is the difference between a course and a workshop?",
        "Explain how to use a breadcrumb in web design.",
        "Write a JavaScript function to format a date.",
        "How do I make studying fun?",
        "Explain the term 'data privacy'.",
        "Write a FastAPI model with Pydantic.",
        "What is the most spoken language in the world?",
        "Explain how to write a punchy social media caption.",
        "Write a Python generator expression example.",
        "How do I choose a laptop for programming?",
        "Explain the difference between a lie and a mistake.",
        "Write a SQL query to find the longest name in a table.",
        "What are the advantages of reading daily?",
        "Explain how to use the zip function in Python.",
        "Write a React Router setup with two pages.",
        "How do I negotiate a deadline?",
        "Explain the term 'impression management'.",
        "Write a C program using a struct.",
        "What is the difference between jogging and running?",
        "Explain how to test a Python function.",
        "Write a bash one-liner to kill a process by name.",
        "How do I make friends in a new city?",
        "Explain the concept of 'opportunity cost' again.",
        "Write a Python function to check if a year is a leap year.",
        "What is the best time to exercise?",
        "Explain how to write a good README.",
        "Write a SQL query to prevent duplicate rows.",
        "How do I keep a clean inbox?",
        "Explain the term 'gamification'.",
        "Write a JavaScript async function example.",
        "What are the basic sewing skills?",
        "Explain how to use the reduce function with an object.",
        "Write a Python script to rename files in a folder.",
        "How do I improve my listening skills?",
        "Explain the difference between motivation and discipline.",
        "Write a Kotlin function with a default parameter.",
        "What is the purpose of a firewall rule?",
        "Explain how to give a toast at a wedding.",
        "Write a bash script with a case statement.",
        "How do I track my expenses?",
        "Explain the term 'cognitive load'.",
        "Write a Python function using a dictionary comprehension.",
        "What is the difference between a hobby and a passion?",
        "Explain how to create a simple game in Python with pygame.",
        "Write a SQL query to show table structure.",
        "How do I respond to a rejection?",
        "Explain the concept of 'habit stacking'.",
        "Write a JavaScript function that deep clones an object.",
        "What are the best books for beginners?",
        "Explain how to use sets in Python.",
        "Write a Django template with a for loop.",
        "How do I arrange furniture in a small room?",
        "Explain the term 'burn rate' in startups.",
        "Write a Go program with a goroutine.",
        "What is the difference between a legend and a myth?",
        "Explain how to write a progress report.",
        "Write a Python function that reads a file line by line.",
        "How do I avoid phone addiction?",
        "Explain the term 'user experience'.",
        "Write a CSS transition on hover.",
        "What is the most efficient way to commute?",
        "Explain how to use environment files in Node.js.",
        "Write a Python script to convert a list to a string.",
        "How do I build resilience?",
        "Explain the difference between copyright and trademark.",
        "Write a SQL query using a subquery.",
        "What are the basics of public speaking?",
        "Explain how to make a study timetable.",
        "Write a Java function that returns a boolean.",
        "How do I choose healthy snacks?",
        "Explain the term 'influencer marketing'.",
        "Write a Python function that checks email format.",
        "What is the difference between a goal and a dream?",
        "Explain how to use yield in Python.",
        "Write a React component with useEffect.",
        "How do I clean my keyboard?",
        "Explain the concept of 'digital footprint'.",
        "Write a bash script to display system info.",
        "What are the benefits of walking daily?",
        "Explain how to write a short story.",
        "Write a Python program that finds duplicates in a list.",
        "How do I organize my desktop files?",
        "Explain the term 'MVP' in business.",
        "Write a SQL query to rank users by points.",
        "What is the difference between a fact and an opinion?",
        "Explain how to use list comprehension in Python.",
        "Write a JavaScript timer using setInterval.",
        "How do I prepare for a marathon?",
        "Explain the concept of 'good debt'.",
        "Write a FastAPI dependency example.",
        "What are the essentials for a home office?",
        "Explain how to use the argparse module in Python.",
        "Write a C# method with a return value.",
        "How do I deal with difficult people?",
        "Explain the term 'gaslighting'.",
        "Write a Python script that checks disk space with shutil.",
        "What is the difference between a summary and a review?",
        "Explain how to use custom fonts in CSS.",
        "Write a SQL query to find the second highest value.",
        "How do I set boundaries at work?",
        "Explain the concept of 'compound growth'.",
        "Write a Python function to shuffle a string.",
        "What are the key nutrients for brain health?",
        "Explain how to do a SWOT analysis.",
        "Write a JavaScript arrow function example.",
        "How do I improve my posture?",
        "Explain the term 'API rate limiting'.",
        "Write a Django REST framework serializer.",
        "What is the difference between a synonym and an antonym?",
        "Explain how to use git stash.",
        "Write a Python function that uses logging.",
        "How do I plan a content calendar?",
        "Explain the concept of 'emotional intelligence'.",
        "Write a bash script that creates directories.",
        "What are the best exercises for back pain?",
        "Explain how to write a clear error message.",
        "Write a React button that increments a value.",
        "How do I reduce food waste?",
        "Explain the term 'seed funding'.",
        "Write a Python program that draws a circle with turtle.",
        "What is the difference between a rule and a law?",
        "Explain how to use defaultdict in Python.",
        "Write a SQL query with CASE WHEN.",
        "How do I keep my phone battery healthy?",
        "Explain the concept of 'work-life balance'.",
        "Write a Node.js middleware example.",
        "What are the signs of good leadership?",
        "Explain how to make a decision matrix.",
        "Write a Python function that generates a UUID.",
        "How do I write a to-do list that works?",
        "Explain the term 'freemium'.",
        "Write a CSS media query for mobile.",
        "What is the difference between a proverb and a quote?",
        "Explain how to use the time module in Python.",
        "Write a FastAPI endpoint with path parameters.",
        "How do I practice active listening?",
        "Explain the concept of 'minimum viable product'.",
        "Write a Python script to split a string by commas.",
        "What are the benefits of learning history?",
        "Explain how to use try/except else in Python.",
        "Write a bash script that reads a config file.",
        "How do I build a morning routine?",
        "Explain the term 'net neutrality'.",
        "Write a Java program with a for-each loop.",
        "What is the difference between a hobby income and a business?",
        "Explain how to write a mission statement.",
        "Write a Python function that checks if a number is even.",
        "How do I improve my English writing?",
        "Explain the concept of 'opportunity recognition'.",
        "Write a SQL query to delete a table.",
        "What are the basics of bicycle maintenance?",
        "Explain how to use tuples vs lists in Python again.",
        "Write a React component that shows the current time.",
        "How do I make my code faster?",
        "Explain the term 'sunk cost'.",
        "Write a Python script that merges two JSON files.",
        "What is the difference between a mentor and a coach?",
        "Explain how to write a cold email.",
        "Write a bash command to sort a file alphabetically.",
        "How do I stay consistent with workouts?",
        "Explain the concept of 'critical thinking'.",
        "Write a JavaScript function to sum an array.",
        "What are the best apps for note-taking?",
        "Explain how to use exceptions in Python.",
        "Write a Django model with a ForeignKey.",
        "How do I create a study group?",
        "Explain the term 'digital detox'.",
        "Write a Python program that prints a multiplication table.",
        "What is the difference between a team and a group?",
        "Explain how to use the requests library in Python.",
        "Write a SQL query to backup a table.",
        "How do I prepare for an exam in one week?",
        "Explain the concept of 'positive thinking'.",
        "Write a Go program that reads a file.",
        "What are the keys to effective delegation?",
        "Explain how to write a thesis statement.",
        "Write a Python function that checks password strength.",
        "How do I set up a morning walk habit?",
        "Explain the term 'digital literacy'.",
        "Write a CSS button with a gradient.",
        "What is the difference between a prediction and a guarantee?",
        "Explain how to use filter in Python.",
        "Write a JavaScript fetch POST example.",
        "How do I negotiate a good price?",
        "Explain the concept of 'active income'.",
        "Write a FastAPI file upload endpoint.",
        "What are the best habits for productivity?",
        "Explain how to use kwargs in Python.",
        "Write a bash script that checks internet connectivity.",
        "How do I make a budget spreadsheet?",
        "Explain the term 'social proof'.",
        "Write a Python class with a static method.",
        "What is the difference between a leader and a manager?",
        "Explain how to write a personal statement.",
        "Write a SQL query using JOIN with alias.",
        "How do I declutter my home?",
        "Explain the concept of 'financial freedom'.",
        "Write a JavaScript function to flatten an array.",
        "What are the best ways to relax?",
        "Explain how to use enumerate in Python.",
        "Write a Node.js script that uses environment variables.",
        "How do I keep a gratitude journal?",
        "Explain the term 'value proposition'.",
        "Write a Python function that returns unique items in order.",
        "What is the difference between a goal and a milestone?",
        "Explain how to make a personal development plan.",
        "Write a CSS sticky header example.",
        "How do I respond to a compliment?",
        "Explain the concept of 'compound learning'.",
        "Write a Django command example.",
        "What are the best study techniques for math?",
        "Explain how to use slicing in Python.",
        "Write a JavaScript reduce to count occurrences.",
        "How do I improve my willpower?",
        "Explain the term 'brainstorming'.",
        "Write a Python script that generates a multiplication quiz.",
        "What is the difference between a question and a problem?",
        "Explain how to write a conclusion.",
        "Write a bash command to find empty directories.",
        "How do I set priorities?",
        "Explain the concept of 'continuous improvement'.",
        "Write a React component with props.",
        "What are the basics of time tracking?",
        "Explain how to use f-strings in Python.",
        "Write a SQL query to combine two tables.",
        "How do I make healthier meal choices?",
        "Explain the term 'customer journey'.",
        "Write a Python function that swaps two variables.",
        "What is the difference between training and learning?",
        "Explain how to write an introduction paragraph.",
        "Write a Go program that does arithmetic.",
        "How do I create a vision board?",
        "Explain the concept of 'focus'.",
        "Write a JavaScript function that removes falsy values.",
        "What are the best tools for remote work?",
        "Explain how to use the Counter class in Python.",
        "Write a bash script that prints a welcome message.",
        "How do I manage my energy during the day?",
        "Explain the term 'scalability'.",
        "Write a Python function that reads environment variables.",
        "What is the difference between a slogan and a tagline?",
        "Explain how to write a simple plan.",
        "Write a SQL query with GROUP BY and HAVING.",
        "How do I practice keyboard shortcuts?",
        "Explain the concept of 'random acts of kindness'.",
        "Write a React component that filters a list.",
        "What are the best ways to learn a new skill?",
        "Explain how to use lambda with sorted in Python.",
        "Write a Node.js server that returns JSON.",
        "How do I make a daily schedule?",
        "Explain the term 'domain knowledge'.",
        "Write a Python script that checks internet speed with ping.",
        "What is the difference between a vision and a mission?",
        "Explain how to write a friendly reply.",
        "Write a CSS animated spinner.",
        "How do I choose a good password manager?",
        "Explain the concept of 'delayed gratification'.",
        "Write a JavaScript function to get unique values.",
        "What are the best habits for mental health?",
        "Explain how to use the any() and all() functions in Python.",
        "Write a Django URL pattern example.",
        "How do I stay calm in a crisis?",
        "Explain the term 'brand loyalty'.",
        "Write a Python function that converts markdown headings to HTML.",
        "What is the difference between a task and a project?",
        "Explain how to write a follow-up email.",
        "Write a bash script that appends a timestamp to a log.",
        "How do I choose a topic for a blog?",
        "Explain the concept of 'persistence'.",
        "Write a JavaScript function that capitalizes each word.",
        "What are the best ways to celebrate small wins?",
        "Explain how to use the math module in Python.",
        "Write a SQL query to copy data between tables.",
        "How do I build a reading habit?",
        "Explain the term 'workflow'.",
        "Write a Python program that simulates a coin flip.",
        "What is the difference between a tip and advice?",
        "Explain how to write a thank-you email.",
        "Write a React hook that uses localStorage.",
        "How do I improve my focus during deep work?",
        "Explain the concept of 'accountability'.",
        "Write a FastAPI health check endpoint.",
        "What are the best foods for energy?",
        "Explain how to use the os module in Python.",
        "Write a bash script that sets an alias.",
        "How do I create a simple portfolio website?",
        "Explain the term 'outsourcing'.",
        "Write a Python function that finds the index of a value.",
        "What is the difference between a certification and a degree?",
        "Explain how to write a weekly review.",
        "Write a SQL query to find rows with missing values.",
        "How do I balance work and study?",
        "Explain the concept of 'mentorship'.",
        "Write a JavaScript function to sort dates.",
        "What are the best practices for email etiquette?",
        "Explain how to use break and continue in Python.",
        "Write a Django view with a redirect.",
        "How do I make a good first website footer?",
        "Explain the term 'pivot' in business.",
        "Write a Python script that counts characters in a string.",
        "What is the difference between a guarantee and a warranty?",
        "Explain how to write a short report.",
        "Write a bash command to show running processes.",
        "How do I improve my reading speed?",
        "Explain the concept of 'lifelong learning'.",
        "Write a React component that toggles dark mode.",
        "What are the basics of personal finance?",
        "Explain how to use sys.argv in Python.",
        "Write a SQL query to pivot data.",
        "How do I set up a study corner?",
        "Explain the term 'onboarding'.",
        "Write a Python function that returns the mode of a list.",
        "What is the difference between an idea and an opportunity?",
        "Explain how to write a professional bio.",
        "Write a JavaScript function that parses query params.",
        "How do I make better decisions?",
        "Explain the concept of 'flow state'.",
        "Write a FastAPI error handler.",
        "What are the best ways to network online?",
        "Explain how to use pathlib in Python.",
        "Write a bash script that greps and counts.",
        "How do I create a content strategy?",
        "Explain the term 'churn'.",
        "Write a Python script that sorts a JSON list by a key.",
        "What is the difference between a course and a tutorial?",
        "Explain how to write a project proposal.",
        "Write a Django form with validation.",
        "How do I keep my code DRY?",
        "Explain the concept of 'scenario planning'.",
        "Write a JavaScript Promise.all example.",
        "What are the best ways to avoid burnout?",
        "Explain how to use the datetime module in Python.",
        "Write a SQL query to find users who logged in this month.",
        "How do I organize my Google Drive?",
        "Explain the term 'market research'.",
        "Write a Python function that converts camelCase to snake_case.",
        "What is the difference between a habit and a routine?",
        "Explain how to write a survey question.",
        "Write a CSS grid gallery layout.",
        "How do I improve my negotiation skills?",
        "Explain the concept of 'iteration'.",
        "Write a Go program with a map.",
        "What are the best ways to relax after work?",
        "Explain how to use filter with a lambda in Python.",
        "Write a Node.js script with an interval.",
        "How do I set financial goals?",
        "Explain the term 'upselling'.",
        "Write a Python function that checks if a string contains only digits.",
        "What is the difference between a deadline and a timeline?",
        "Explain how to write a change request.",
        "Write a bash script that compares two files.",
        "How do I build a support network?",
        "Explain the concept of 'sustainable growth'.",
        "Write a React component with conditional rendering.",
        "What are the best habits for coding daily?",
        "Explain how to use the json module in Python.",
        "Write a SQL query that uses DISTINCT.",
        "How do I create a personal brand on social media?",
        "Explain the term 'customer support'.",
        "Write a Python script that downloads a file.",
        "What is the difference between a roadmap and a plan?",
        "Explain how to write a compelling headline.",
        "Write a Django test example.",
        "How do I improve my email subject lines?",
        "Explain the concept of 'active recall'.",
        "Write a JavaScript function that debounces a search.",
        "What are the best ways to learn data science?",
        "Explain how to use context managers in Python.",
        "Write a FastAPI response model example.",
        "How do I handle feedback on my work?",
        "Explain the term 'conversion rate'.",
        "Write a Python function that returns the file extension.",
        "What is the difference between a target and an outcome?",
        "Explain how to write a kickoff message.",
        "Write a bash script that waits for a process.",
        "How do I choose between two job offers?",
        "Explain the concept of 'shared goals'.",
        "Write a React component that uses context.",
        "What are the best practices for version control?",
        "Explain how to use the random module in Python.",
        "Write a SQL query to shuffle results.",
        "How do I improve my decision speed?",
        "Explain the term 'retention rate'.",
        "Write a Python script that parses CSV with csv module.",
        "What is the difference between a milestone and a deliverable?",
        "Explain how to write a status update.",
        "Write a JavaScript function to check if an array is sorted.",
        "How do I create a learning plan?",
        "Explain the concept of 'psychological safety'.",
        "Write a FastAPI WebSocket example.",
        "What are the best ways to use breaks at work?",
        "Explain how to use iterators in Python.",
        "Write a Django API with token auth.",
        "How do I make my writing clearer?",
        "Explain the term 'key performance indicator'.",
        "Write a Python function that truncates a string to N words.",
        "What is the difference between an audit and a review?",
        "Explain how to write a meeting agenda.",
        "Write a bash script that rotates logs.",
        "How do I stay organized with many projects?",
        "Explain the concept of 'quality over quantity'.",
        "Write a JavaScript function that converts an object to an array.",
        "What are the best resources to learn SQL?",
        "Explain how to use the collections module in Python.",
        "Write a SQL query for a self join.",
        "How do I build an online presence?",
        "Explain the term 'data-driven'.",
        "Write a Python script that plots a simple chart with matplotlib.",
        "What is the difference between a brief and a spec?",
        "Explain how to write a user guide.",
        "Write a React component that validates a form.",
        "How do I improve my handwriting speed?",
        "Explain the concept of 'incremental progress'.",
        "Write a Go program with a slice.",
        "What are the best habits for success?",
        "Explain how to use the re module for replacement in Python.",
        "Write a Django admin customization example.",
        "How do I prepare a presentation deck?",
        "Explain the term 'market fit'.",
        "Write a Python function that returns the longest word in a string.",
        "What is the difference between a role and a responsibility?",
        "Explain how to write an elevator pitch.",
        "Write a bash script that checks disk health.",
        "How do I manage remote meetings?",
        "Explain the concept of 'constructive conflict'.",
        "Write a JavaScript function to group an array by key.",
        "What are the best ways to reduce stress?",
        "Explain how to use the itertools module in Python.",
        "Write a SQL query with window functions.",
        "How do I create a habit tracker?",
        "Explain the term 'turnaround time'.",
        "Write a FastAPI dependency injection example.",
        "What is the difference between a synonym and a similar word?",
        "Explain how to write a cold outreach message.",
        "Write a Python script that checks date validity.",
        "How do I build a daily reading routine?",
        "Explain the concept of 'accountability partner'.",
        "Write a React list with keys.",
        "What are the best ways to learn by doing?",
        "Explain how to use the typing module in Python.",
        "Write a Django query optimization tip.",
        "How do I set up notifications that help?",
        "Explain the term 'resource allocation'.",
        "Write a Python function that checks anagram pairs in a list.",
        "What is the difference between a job description and a spec?",
        "Explain how to write a personal growth plan.",
        "Write a bash script that renames files with a pattern.",
        "How do I improve my team communication?",
        "Explain the concept of 'structured thinking'.",
        "Write a JavaScript function that throttles events.",
        "What are the best ways to end the day well?",
        "Explain how to use generators vs lists in Python.",
        "Write a SQL query to update multiple rows.",
        "How do I plan a product launch?",
        "Explain the term 'burnout recovery'.",
        "Write a Python script that checks email regex.",
        "What is the difference between a lesson and a lecture?",
        "Explain how to write a weekly goal review.",
        "Write a FastAPI CRUD example.",
        "How do I make my home more organized?",
        "Explain the concept of 'visible progress'.",
        "Write a JavaScript function that maps an array to objects.",
        "What are the best ways to learn quickly?",
        "Explain how to use the glob module in Python.",
        "Write a Django pagination example.",
        "How do I set up a daily standup?",
        "Explain the term 'actionable feedback'.",
        "Write a Python function that finds the factorial with recursion.",
        "What is the difference between a bug fix and a feature?",
        "Explain how to write a clear disclaimer.",
        "Write a bash script that checks the date.",
        "How do I create a content pillar?",
        "Explain the concept of 'deep work'.",
        "Write a React component that uses refs.",
        "What are the best practices for code reviews?",
        "Explain how to use the timeit module in Python.",
        "Write a SQL query to archive old records.",
        "How do I improve my memory for facts?",
        "Explain the term 'feedback loop'.",
        "Write a Python script that reads an Excel file with openpyxl.",
        "What is the difference between a draft and a final version?",
        "Explain how to write a negotiation email.",
        "Write a Go program with a struct method.",
        "How do I make better use of my weekends?",
        "Explain the concept of 'margin of safety'.",
        "Write a JavaScript function that checks balanced parentheses.",
        "What are the best ways to practice English daily?",
        "Explain how to use the hashlib module in Python.",
        "Write a Django signal example.",
        "How do I set a realistic deadline?",
        "Explain the term 'benchmarking'.",
        "Write a Python function that converts a list to a dictionary.",
        "What is the difference between a review and a reflection?",
        "Explain how to write a weekly summary.",
        "Write a bash script that prints a calendar.",
        "How do I keep up with technology news?",
        "Explain the concept of 'continuous learning'.",
        "Write a React component with a modal.",
        "What are the best ways to plan a day?",
        "Explain how to use the csv module in Python.",
        "Write a SQL query to find the median.",
        "How do I improve my spoken English?",
        "Explain the term 'workload management'.",
        "Write a Python script that generates QR codes with a library.",
        "What is the difference between a task list and a plan?",
        "Explain how to write a project timeline.",
        "Write a Django middleware example.",
        "How do I create a weekly meal plan?",
        "Explain the concept of 'buffer time'.",
        "Write a JavaScript function that finds the longest substring.",
        "What are the best ways to learn from books?",
        "Explain how to use the functools module in Python.",
        "Write a FastAPI background task example.",
        "How do I make a work progress tracker?",
        "Explain the term 'quick win'.",
        "Write a Python function that merges two dicts.",
        "What is the difference between a promise and a commitment?",
        "Explain how to write a user story.",
        "Write a bash script that archives old logs.",
        "How do I build a personal wiki?",
        "Explain the concept of 'single source of truth'.",
        "Write a React component with a search box.",
        "What are the best ways to take notes?",
        "Explain how to use the os.path module in Python.",
        "Write a SQL query to add an index.",
        "How do I create a team norms doc?",
        "Explain the term 'cycle time'.",
        "Write a Python script that finds common elements between two lists.",
        "What is the difference between an expert and a learner?",
        "Explain how to write a closing summary.",
        "Write a Go program that counts words in a string.",
        "How do I prepare for a certification exam?",
        "Explain the concept of 'learning by teaching'.",
        "Write a JavaScript function that checks palindrome words.",
        "What are the best ways to manage anger?",
        "Explain how to use the subprocess module in Python.",
        "Write a Django view that exports CSV.",
        "How do I set up a home gym?",
        "Explain the term 'bandwidth' in a project context.",
        "Write a Python function that calculates a moving average.",
        "What is the difference between a fact check and a review?",
        "Explain how to write a project charter.",
        "Write a bash script that monitors a log file.",
        "How do I reduce decision fatigue?",
        "Explain the concept of 'minimalism'.",
        "Write a React component that uses useReducer.",
        "What are the best ways to build a habit chain?",
        "Explain how to use the string module in Python.",
        "Write a SQL query to detect anomalies.",
        "How do I create a skill inventory?",
        "Explain the term 'employee engagement'.",
        "Write a Python script that checks if a file is empty.",
        "What is the difference between a walkthrough and a demo?",
        "Explain how to write a retrospective.",
        "Write a Django serializers example.",
        "How do I keep my notes organized?",
        "Explain the concept of '80/20 rule' in business.",
        "Write a JavaScript function to merge objects.",
        "What are the best ways to learn a framework?",
        "Explain how to use the ast module in Python.",
        "Write a FastAPI middleware example.",
        "How do I make a project status report?",
        "Explain the term 'standard operating procedure'.",
        "Write a Python function that returns unique pairs summing to a target.",
        "What is the difference between a manual and a guide?",
        "Explain how to write an FAQ.",
        "Write a bash script that checks network speed.",
        "How do I build a customer survey?",
        "Explain the concept of 'user stories'.",
        "Write a React component that uses memo.",
        "What are the best ways to schedule deep work?",
        "Explain how to use the built-in sorted() with a key in Python.",
        "Write a SQL query to find consecutive records.",
        "How do I create a task breakdown?",
        "Explain the term 'throughput'.",
        "Write a Python script that finds the longest common prefix.",
        "What is the difference between a process and a procedure?",
        "Explain how to write a lesson plan.",
        "Write a Go program with a switch statement.",
        "How do I handle competing priorities?",
        "Explain the concept of 'kaizen'.",
        "Write a JavaScript function that converts a number to words.",
        "What are the best ways to give feedback?",
        "Explain how to use the dataclasses module in Python.",
        "Write a Django template filter example.",
        "How do I create a personal OKR?",
        "Explain the term 'exit criteria'.",
        "Write a Python function that checks if a path exists.",
        "What is the difference between a result and an outcome?",
        "Explain how to write a handover document.",
        "Write a bash script that greps multiple patterns.",
        "How do I make a study summary?",
        "Explain the concept of 'critical path'.",
        "Write a React component with error boundary.",
        "What are the best ways to improve concentration?",
        "Explain how to use the zipfile module in Python.",
        "Write a SQL query to compare two tables.",
        "How do I build a career plan?",
        "Explain the term 'stakeholder'.",
        "Write a Python script that finds the top N items in a list.",
        "What is the difference between a review meeting and a planning meeting?",
        "Explain how to write a daily log.",
        "Write a Django model manager example.",
        "How do I improve my vocabulary?",
        "Explain the concept of 'mind mapping'.",
        "Write a JavaScript function that reverses words in a sentence.",
        "What are the best ways to end procrastination?",
        "Explain how to use the sys module in Python.",
        "Write a FastAPI pagination example.",
        "How do I create a personal dashboard?",
        "Explain the term 'run rate'.",
        "Write a Python function that checks balanced brackets.",
        "What is the difference between a test and a quiz?",
        "Explain how to write a release notes draft.",
        "Write a bash script that parses arguments.",
        "How do I set up weekly goals?",
        "Explain the concept of 'incremental delivery'.",
        "Write a React component with portals.",
        "What are the best ways to practice mindfulness?",
        "Explain how to use the built-in max with key in Python.",
        "Write a SQL query to fill missing dates.",
        "How do I create a meeting template?",
        "Explain the term 'upskilling'.",
        "Write a Python script that splits a large file.",
        "What is the difference between a rubric and a checklist?",
        "Explain how to write a product requirements doc.",
        "Write a Django queryset optimization example.",
        "How do I make a decision when uncertain?",
        "Explain the concept of 'design thinking'.",
        "Write a JavaScript function to detect duplicates.",
        "What are the best ways to keep a team informed?",
        "Explain how to use the contextlib module in Python.",
        "Write a FastAPI custom exception example.",
        "How do I create a content calendar template?",
        "Explain the term 'dependency'.",
        "Write a Python function that validates a credit card number.",
        "What is the difference between a sprint and a milestone?",
        "Explain how to write a test plan.",
        "Write a bash script that shows open ports.",
        "How do I plan a workshop?",
        "Explain the concept of 'focused attention'.",
        "Write a React component with lazy loading.",
        "What are the best ways to learn grammar?",
        "Explain how to use the re module for search in Python.",
        "Write a SQL query to normalize a table.",
        "How do I create a job description?",
        "Explain the term 'acceptance criteria'.",
        "Write a Python script that extracts URLs from text.",
        "What is the difference between a plan review and a plan update?",
        "Explain how to write a risk register.",
        "Write a Go program that reads user input.",
        "How do I handle unexpected changes?",
        "Explain the concept of 'prioritization matrix'.",
        "Write a JavaScript function that counts vowels.",
        "What are the best ways to celebrate success?",
        "Explain how to use the calendar module in Python.",
        "Write a Django cache example.",
        "How do I create a daily journal template?",
        "Explain the term 'lead time'.",
        "Write a Python function that finds pairs with a given sum.",
        "What is the difference between a draft plan and a final plan?",
        "Explain how to write a kickoff checklist.",
        "Write a bash script that splits a log by date.",
        "How do I build a reading list?",
        "Explain the concept of 'structured procrastination'.",
        "Write a React component with a dropdown.",
        "What are the best ways to retain information?",
        "Explain how to use the statistics module in Python.",
        "Write a SQL query to compute cumulative totals.",
        "How do I create a team schedule?",
        "Explain the term 'benchmark'.",
        "Write a Python script that checks if a string is a palindrome ignoring spaces.",
        "What is the difference between a template and an example?",
        "Explain how to write a debrief.",
        "Write a Django test client example.",
        "How do I set up a personal finance tracker?",
        "Explain the concept of 'goal cascade'.",
        "Write a JavaScript function that validates an email.",
        "What are the best ways to reduce email overload?",
        "Explain how to use the unittest module in Python.",
        "Write a FastAPI WebSocket broadcast example.",
        "How do I create a task priority list?",
        "Explain the term 'scope creep'.",
        "Write a Python function that returns the kth largest element.",
        "What is the difference between a metric and an indicator?",
        "Explain how to write a weekly activity log.",
        "Write a bash script that backs up files older than a day.",
        "How do I make a habit of journaling?",
        "Explain the concept of 'smart goals'.",
        "Write a React component that uses Suspense.",
        "What are the best ways to handle interruptions?",
        "Explain how to use the warnings module in Python.",
        "Write a SQL query to find top sellers per region.",
        "How do I create a project folder structure?",
        "Explain the term 'velocity'.",
        "Write a Python script that checks odd/even without modulo.",
        "What is the difference between a walkthrough and a code review?",
        "Explain how to write a handover email.",
        "Write a Django celery example.",
        "How do I improve my morning routine?",
        "Explain the concept of 'parkinson's law'.",
        "Write a JavaScript function that finds the missing number in an array.",
        "What are the best ways to learn from failures?",
        "Explain how to use the pprint module in Python.",
        "Write a FastAPI static file serving example.",
        "How do I create a communication plan?",
        "Explain the term 'cycle time' in agile.",
        "Write a Python function that checks if a string is a valid ISBN.",
        "What is the difference between a milestone chart and a gantt chart?",
        "Explain how to write a sprint retrospective.",
        "Write a bash script that counts words in a file.",
        "How do I build a study schedule?",
        "Explain the concept of 'deliberate practice'.",
        "Write a React component with a tooltip.",
        "What are the best ways to track habits?",
        "Explain how to use the setdefault method in Python.",
        "Write a SQL query to flag duplicates.",
        "How do I create a team wiki?",
        "Explain the term 'risk appetite'.",
        "Write a Python script that finds the shortest word.",
        "What is the difference between a budget and a forecast?",
        "Explain how to write a status email.",
        "Write a Django signals best practice example.",
        "How do I improve my estimation skills?",
        "Explain the concept of 'compound interest' in skills.",
        "Write a JavaScript function that checks leap year.",
        "What are the best ways to plan a month?",
        "Explain how to use the importlib module in Python.",
        "Write a FastAPI rate limit example.",
        "How do I create a personal project tracker?",
        "Explain the term 'milestone'.",
        "Write a Python function that returns the median of a list.",
        "What is the difference between a walkthrough and a tutorial?",
        "Explain how to write a daily standup message.",
        "Write a bash script that finds recently modified files.",
        "How do I build a learning journal?",
        "Explain the concept of 'growth mindset'.",
        "Write a React component that uses forwardRef.",
        "What are the best ways to avoid information overload?",
        "Explain how to use the pickle module in Python.",
        "Write a SQL query to find the longest streak.",
        "How do I create a yearly plan?",
        "Explain the term 'sprint goal'.",
        "Write a Python script that counts words in a text file.",
        "What is the difference between a plan and a strategy session?",
        "Explain how to write a project closure report.",
        "Write a Django view with pagination.",
        "How do I improve my focus with music?",
        "Explain the concept of 'eat the frog'.",
        "Write a JavaScript function that checks if two arrays are equal.",
        "What are the best ways to handle constructive feedback?",
        "Explain how to use the built-in isinstance in Python.",
        "Write a FastAPI login endpoint example.",
        "How do I create a folder naming convention?",
        "Explain the term 'definition of done'.",
        "Write a Python function that checks for duplicates in a string.",
        "What is the difference between a hack and a solution?",
        "Explain how to write a knowledge base article.",
        "Write a bash script that compresses a directory.",
        "How do I build a project roadmap?",
        "Explain the concept of 'time blocking'.",
        "Write a React component with a progress bar.",
        "What are the best ways to end a meeting?",
        "Explain how to use the built-in property in Python.",
        "Write a SQL query to find the most active users.",
        "How do I create a skill matrix?",
        "Explain the term 'cross-training'.",
        "Write a Python script that converts a string to title case.",
        "What is the difference between a reference and a citation?",
        "Explain how to write a thank-you after a meeting.",
        "Write a Django email example.",
        "How do I make a weekly planner?",
        "Explain the concept of 'spaced repetition'.",
        "Write a JavaScript function that detects anagram.",
        "What are the best ways to clear your mind?",
        "Explain how to use the built-in enumerate in a loop.",
        "Write a FastAPI upload with progress example.",
        "How do I create a decision log?",
        "Explain the term 'dependencies' in project planning.",
        "Write a Python function that checks if a list is sorted.",
        "What is the difference between a report and a dashboard?",
        "Explain how to write a meeting minutes template.",
        "Write a bash script that finds duplicate files.",
        "How do I build a personal brand statement?",
        "Explain the concept of 'first principles'.",
        "Write a React component that uses useMemo.",
        "What are the best ways to practice coding?",
        "Explain how to use the built-in any with a list comprehension.",
        "Write a SQL query to identify gaps in sequence.",
        "How do I create a reading tracker?",
        "Explain the term 'alignment'.",
        "Write a Python script that merges lists and removes duplicates.",
        "What is the difference between a summary and a conclusion?",
        "Explain how to write a retrospective report.",
        "Write a Django filter example.",
        "How do I improve my speed of work?",
        "Explain the concept of 'maximum effort'.",
        "Write a JavaScript function that trims whitespace.",
        "What are the best ways to prepare for a meeting?",
        "Explain how to use the built-in reversed in Python.",
        "Write a FastAPI OAuth2 example.",
        "How do I create a daily checklist?",
        "Explain the term 'knowledge transfer'.",
        "Write a Python function that validates JSON.",
        "What is the difference between a milestone report and a progress report?",
        "Explain how to write a launch checklist.",
        "Write a bash script that finds files by size.",
        "How do I build a study group plan?",
        "Explain the concept of 'structured review'.",
        "Write a React component that uses useCallback.",
        "What are the best ways to learn a tool?",
        "Explain how to use the built-in sorted with reverse in Python.",
        "Write a SQL query to compute running totals.",
        "How do I create a personal OKR tracker?",
        "Explain the term 'sprint planning'.",
        "Write a Python script that detects the language of a short text.",
        "What is the difference between a guide and a checklist?",
        "Explain how to write a weekly team update.",
        "Write a Django REST framework permissions example.",
        "How do I improve my time estimates?",
        "Explain the concept of 'backlog'.",
        "Write a JavaScript function that gets today's date.",
        "What are the best ways to organize a study desk?",
        "Explain how to use the built-in zip_longest in Python.",
        "Write a FastAPI exception handling example.",
        "How do I create a project risk log?",
        "Explain the term 'velocity' in scrum.",
        "Write a Python function that checks if a number is perfect.",
        "What is the difference between a document and a record?",
        "Explain how to write a training plan.",
        "Write a bash script that sorts files into folders by extension.",
        "How do I build a coding challenge routine?",
        "Explain the concept of 'focused practice'.",
        "Write a React component with a toggle.",
        "What are the best ways to review your day?",
        "Explain how to use the built-in min with a lambda in Python.",
        "Write a SQL query to find users with no orders.",
        "How do I create a sprint board?",
        "Explain the term 'definition of ready'.",
        "Write a Python script that checks if a file path is a directory.",
        "What is the difference between a process document and a policy?",
        "Explain how to write a daily goals list.",
        "Write a Django model verbose name example.",
        "How do I improve my note structure?",
        "Explain the concept of 'parkinson's law' in tasks.",
        "Write a JavaScript function that converts string to number safely.",
        "What are the best ways to avoid multitasking?",
        "Explain how to use the built-in input in a loop.",
        "Write a FastAPI streaming response example.",
        "How do I create a weekly review template?",
        "Explain the term 'story points'.",
        "Write a Python function that finds all permutations of a string.",
        "What is the difference between a milestone and a checkpoint?",
        "Explain how to write a project kickoff email.",
        "Write a bash script that checks if a port is open.",
        "How do I build a habit scorecard?",
        "Explain the concept of 'execution'.",
        "Write a React component with a theme provider.",
        "What are the best ways to prepare notes for a test?",
        "Explain how to use the built-in filter with a function.",
        "Write a SQL query to find the mode of a column.",
        "How do I create a stakeholder map?",
        "Explain the term 'release plan'.",
        "Write a Python script that checks if a number is Fibonacci.",
        "What is the difference between a task and a sub-task?",
        "Explain how to write a progress tracking doc.",
        "Write a Django queryset filter example.",
        "How do I improve my meeting participation?",
        "Explain the concept of 'one thing at a time'.",
        "Write a JavaScript function that sorts an array of objects.",
        "What are the best ways to build a portfolio?",
        "Explain how to use the built-in glob in a script.",
        "Write a FastAPI healthcheck with timeout example.",
        "How do I create a learning roadmap?",
        "Explain the term 'key result'.",
        "Write a Python function that checks if a string starts with a vowel.",
        "What is the difference between a plan review and a progress review?",
        "Explain how to write a sprint review.",
        "Write a bash script that greps a file and extracts emails.",
        "How do I build a personal system?",
        "Explain the concept of 'experimentation'.",
        "Write a React component that uses a custom hook.",
        "What are the best ways to track progress?",
        "Explain how to use the built-in print with separators in Python.",
        "Write a SQL query to find the second most common value.",
        "How do I create a lesson outline?",
        "Explain the term 'iteration' in product.",
        "Write a Python script that finds the GCD of two numbers.",
        "What is the difference between a mock and a simulation?",
        "Explain how to write a daily reflection.",
        "Write a Django template include example.",
        "How do I improve my body language?",
        "Explain the concept of 'continuous feedback'.",
        "Write a JavaScript function that converts an array to CSV.",
        "What are the best ways to end a work day?",
        "Explain how to use the built-in max with a generator.",
        "Write a FastAPI router organization example.",
        "How do I create a meeting summary template?",
        "Explain the term 'blocker'.",
        "Write a Python function that checks if two lists overlap.",
        "What is the difference between a plan and a proposal?",
        "Explain how to write a sprint plan.",
        "Write a bash script that monitors memory usage.",
        "How do I build a personal OKR cycle?",
        "Explain the concept of 'small wins'.",
        "Write a React component that uses portals for a modal.",
        "What are the best ways to improve handwriting legibility?",
        "Explain how to use the built-in round in Python.",
        "Write a SQL query to find customers with repeat purchases.",
        "How do I create a knowledge map?",
        "Explain the term 'blocking issue'.",
        "Write a Python script that checks if a list has consecutive numbers.",
        "What is the difference between a summary and an abstract?",
        "Explain how to write a kickoff document.",
        "Write a Django many-to-many example.",
        "How do I improve my daily energy?",
        "Explain the concept of 'opportunity cost' in time.",
        "Write a JavaScript function that finds common values.",
        "What are the best ways to prepare a report?",
        "Explain how to use the built-in hasattr in Python.",
        "Write a FastAPI settings management example.",
        "How do I create a project retrospective?",
        "Explain the term 'data point'.",
        "Write a Python function that returns the most frequent character.",
        "What is the difference between a guideline and a rule?",
        "Explain how to write a daily progress note.",
        "Write a bash script that checks system uptime.",
        "How do I build a review routine?",
        "Explain the concept of 'iterative improvement'.",
        "Write a React component that uses useEffect cleanup.",
        "What are the best ways to plan a week?",
        "Explain how to use the built-in vars in Python.",
        "Write a SQL query to find rows with the same name.",
        "How do I create a daily gratitude list?",
        "Explain the term 'stakeholder management'.",
        "Write a Python script that checks if a string contains numbers.",
        "What is the difference between a walkthrough and a review?",
        "Explain how to write a handoff checklist.",
        "Write a Django aggregate example.",
        "How do I improve my phone etiquette?",
        "Explain the concept of 'active engagement'.",
        "Write a JavaScript function that checks for palindromes in a list.",
        "What are the best ways to plan a study session?",
        "Explain how to use the built-in format in Python.",
        "Write a FastAPI CORS example.",
        "How do I create a personal time log?",
        "Explain the term 'burn down'.",
        "Write a Python function that checks if a word is a pangram.",
        "What is the difference between a plan and a calendar?",
        "Explain how to write a meeting request email.",
        "Write a bash script that prints system date in a format.",
        "How do I build a daily stretching routine?",
        "Explain the concept of 'quality time'.",
        "Write a React component that uses useLayoutEffect.",
        "What are the best ways to handle a long day?",
        "Explain how to use the built-in pow in Python.",
        "Write a SQL query to find pairs of users who both bought an item.",
        "How do I create a personal learning plan?",
        "Explain the term 'schedule buffer'.",
        "Write a Python script that finds the longest increasing subsequence.",
        "What is the difference between a milestone and a deliverable date?",
        "Explain how to write a status meeting agenda.",
        "Write a Django admin list filter example.",
        "How do I improve my typing accuracy?",
        "Explain the concept of 'interleaving' in study.",
        "Write a JavaScript function that checks if a date is in the past.",
        "What are the best ways to stay hydrated?",
        "Explain how to use the built-in slice in Python.",
        "Write a FastAPI logging example.",
        "How do I create a weekly menu?",
        "Explain the term 'capacity planning'.",
        "Write a Python function that checks if a number is Armstrong.",
        "What is the difference between a lesson plan and a syllabus?",
        "Explain how to write a project summary.",
        "Write a bash script that finds the newest file.",
        "How do I build a personal calendar?",
        "Explain the concept of 'deep breathing'.",
        "Write a React component that uses a portal for tooltips.",
        "What are the best ways to review notes?",
        "Explain how to use the built-in sum in Python.",
        "Write a SQL query to find trending products.",
        "How do I create a team calendar?",
        "Explain the term 'time budget'.",
        "Write a Python script that checks if a string is a valid time.",
        "What is the difference between a draft and a proposal?",
        "Explain how to write a weekly update email.",
        "Write a Django formset example.",
        "How do I improve my reaction time?",
        "Explain the concept of 'stepwise refinement'.",
        "Write a JavaScript function that converts hex to rgb.",
        "What are the best ways to prepare for a demo?",
        "Explain how to use the built-in isinstance in a validation.",
        "Write a FastAPI multiple file upload example.",
        "How do I create a project glossary?",
        "Explain the term 'resource plan'.",
        "Write a Python function that checks if a string is a valid hex color.",
        "What is the difference between a milestone and a gate?",
        "Explain how to write a change log.",
        "Write a bash script that greps a word in all files.",
        "How do I build a personal feedback loop?",
        "Explain the concept of 'sequential work'.",
        "Write a React component that uses a context provider.",
        "What are the best ways to avoid email chains?",
        "Explain how to use the built-in abs in Python.",
        "Write a SQL query to find the most recent order per customer.",
        "How do I create a reading schedule?",
        "Explain the term 'deliverable'.",
        "Write a Python script that checks if a string is a valid date.",
        "What is the difference between a report and a recommendation?",
        "Explain how to write a demo script.",
        "Write a Django generic view example.",
        "How do I improve my presentation skills?",
        "Explain the concept of 'task batching'.",
        "Write a JavaScript function that calculates age from a date.",
        "What are the best ways to make lists?",
        "Explain how to use the built-in oct and hex in Python.",
        "Write a FastAPI test example.",
        "How do I create a personal budget?",
        "Explain the term 'runbook'.",
        "Write a Python function that checks if a list is symmetric.",
        "What is the difference between a goal review and a progress check?",
        "Explain how to write a handover summary.",
        "Write a bash script that shows a menu.",
        "How do I build a daily log routine?",
        "Explain the concept of 'priority matrix'.",
        "Write a React component that uses a chart library.",
        "What are the best ways to manage a long project?",
        "Explain how to use the built-in bin in Python.",
        "Write a SQL query to find the busiest day.",
        "How do I create a meeting note template?",
        "Explain the term 'handoff'.",
        "Write a Python script that checks if two strings are rotations.",
        "What is the difference between a milestone and a phase?",
        "Explain how to write a project plan outline.",
        "Write a Django class-based view example.",
        "How do I improve my listening in meetings?",
        "Explain the concept of 'active follow-up'.",
        "Write a JavaScript function that validates a URL.",
        "What are the best ways to plan meals?",
        "Explain how to use the built-in any with generators in Python.",
        "Write a FastAPI docs customization example.",
        "How do I create a personal dashboard template?",
        "Explain the term 'critical path method'.",
        "Write a Python function that checks if a number is a happy number.",
        "What is the difference between a plan and an agenda?",
        "Explain how to write a status slide.",
        "Write a bash script that finds files with a pattern and counts them.",
        "How do I build a weekly reflection?",
        "Explain the concept of 'review cadence'.",
        "Write a React component that uses a reducer for a form.",
        "What are the best ways to handle a heavy workload?",
        "Explain how to use the built-in len in a check.",
        "Write a SQL query to find top 3 per group.",
        "How do I create a personal website?",
        "Explain the term 'baseline'.",
        "Write a Python script that checks if a string is a palindrome sentence.",
        "What is the difference between a milestone and a version?",
        "Explain how to write a demo day summary.",
        "Write a Django custom management command example.",
        "How do I improve my planning skills?",
        "Explain the concept of 'estimation'.",
        "Write a JavaScript function that checks if a string is a valid phone number.",
        "What are the best ways to track reading?",
        "Explain how to use the built-in max with default in Python.",
        "Write a FastAPI redirect example.",
        "How do I create a learning diary?",
        "Explain the term 'transition'.",
        "Write a Python function that finds the smallest missing positive integer.",
        "What is the difference between a draft plan and a contingency?",
        "Explain how to write a test case.",
        "Write a bash script that finds the oldest file.",
        "How do I build a project update habit?",
        "Explain the concept of 'review timing'.",
        "Write a React component that uses an external library.",
        "What are the best ways to prepare for a workshop?",
        "Explain how to use the built-in callable in Python.",
        "Write a SQL query to find duplicates with counts.",
        "How do I create a checklist template?",
        "Explain the term 'deliverable checklist'.",
        "Write a Python script that checks if a list contains a substring.",
        "What is the difference between a milestone and a target?",
        "Explain how to write a pre-meeting memo.",
        "Write a Django permission check example.",
        "How do I improve my writing speed?",
        "Explain the concept of 'time audit'.",
        "Write a JavaScript function that checks if a year is a century.",
        "What are the best ways to end a week?",
        "Explain how to use the built-in divmod in Python.",
        "Write a FastAPI form data example.",
        "How do I create a personal note system?",
        "Explain the term 'review board'.",
        "Write a Python function that returns a list of divisors.",
        "What is the difference between a progress report and a status update?",
        "Explain how to write a one-page plan.",
        "Write a bash script that counts files by type.",
        "How do I build a study group?",
        "Explain the concept of 'chunking'.",
        "Write a React component that uses a portal for popovers.",
        "What are the best ways to improve spelling?",
        "Explain how to use the built-in vars in debugging.",
        "Write a SQL query to find gaps in IDs.",
        "How do I create a team norms doc?",
        "Explain the term 'decision log'.",
        "Write a Python script that checks if a string has balanced brackets.",
        "What is the difference between a plan and a roadmap?",
        "Explain how to write a quick summary.",
        "Write a Django signal receiver example.",
        "How do I improve my timeboxing?",
        "Explain the concept of 'retro'.",
        "Write a JavaScript function that converts degrees to radians.",
        "What are the best ways to build stamina?",
        "Explain how to use the built-in round with digits in Python.",
        "Write a FastAPI background jobs example.",
        "How do I create a personal trackers sheet?",
        "Explain the term 'milestone review'.",
        "Write a Python function that checks if a string is a valid CSS color.",
        "What is the difference between a checklist and a rubric?",
        "Explain how to write a handover plan.",
        "Write a bash script that moves files by pattern.",
        "How do I build a daily review routine?",
        "Explain the concept of 'quality checks'.",
        "Write a React component that uses IntersectionObserver.",
        "What are the best ways to learn typing?",
        "Explain how to use the built-in chr and ord in Python.",
        "Write a SQL query to find the second order date per customer.",
        "How do I create a lesson recap?",
        "Explain the term 'deliverable ownership'.",
        "Write a Python script that checks if a file is a symlink.",
        "What is the difference between a draft and a template?",
        "Explain how to write a review checklist.",
        "Write a Django pagination class example.",
        "How do I improve my decision speed under pressure?",
        "Explain the concept of 'timeboxing'.",
        "Write a JavaScript function that returns the current month.",
        "What are the best ways to keep a planner?",
        "Explain how to use the built-in sorted with a custom key in Python.",
        "Write a FastAPI database session example.",
        "How do I create a personal metrics dashboard?",
        "Explain the term 'quality gate'.",
        "Write a Python function that checks if a string is a valid IP.",
        "What is the difference between a milestone and a checkpoint review?",
        "Explain how to write a retrospective summary.",
        "Write a bash script that finds recently changed files.",
        "How do I build a learning schedule?",
        "Explain the concept of 'review loops'.",
        "Write a React component that uses useTransition.",
        "What are the best ways to prepare a summary?",
        "Explain how to use the built-in id in Python.",
        "Write a SQL query to find consecutive duplicates.",
        "How do I create a weekly highlight doc?",
        "Explain the term 'process owner'.",
        "Write a Python script that checks if a string is a valid username.",
        "What is the difference between a plan and a schedule?",
        "Explain how to write a project update.",
        "Write a Django URL include example.",
        "How do I improve my daily standup?",
        "Explain the concept of 'status transparency'.",
        "Write a JavaScript function that gets the week number.",
        "What are the best ways to plan a learning week?",
        "Explain how to use the built-in range in Python.",
        "Write a FastAPI response compression example.",
        "How do I create a personal skill log?",
        "Explain the term 'action items'.",
        "Write a Python function that checks if a number is a perfect square.",
        "What is the difference between a milestone and a deadline?",
        "Explain how to write a post-mortem.",
        "Write a bash script that counts lines per file.",
        "How do I build a daily wins log?",
        "Explain the concept of 'check-in'.",
        "Write a React component that uses a debounce hook.",
        "What are the best ways to manage a calendar?",
        "Explain how to use the built-in tuple in Python.",
        "Write a SQL query to find customers with a single order.",
        "How do I create a project log?",
        "Explain the term 'owner'.",
        "Write a Python script that checks if a string is a valid slug.",
        "What is the difference between a milestone and a target date?",
        "Explain how to write a review meeting notes.",
        "Write a Django cache page example.",
        "How do I improve my weekly planning?",
        "Explain the concept of 'continuous review'.",
        "Write a JavaScript function that returns the last day of a month.",
        "What are the best ways to track wins?",
        "Explain how to use the built-in format spec in Python.",
        "Write a FastAPI security example.",
        "How do I create a personal OKR review?",
        "Explain the term 'task ownership'.",
        "Write a Python function that checks if a string is a valid variable name.",
        "What is the difference between a draft and a final?",
        "Explain how to write a daily update.",
        "Write a bash script that shows file sizes.",
        "How do I build a personal system review?",
        "Explain the concept of 'pair review'.",
        "Write a React component that uses a chart.",
        "What are the best ways to plan a sprint?",
        "Explain how to use the built-in help in Python.",
        "Write a SQL query to find the longest interval.",
        "How do I create a task tracker?",
        "Explain the term 'inventory'.",
        "Write a Python script that checks if a string is a valid number.",
        "What is the difference between a plan and a target?",
        "Explain how to write a handover checklist.",
        "Write a Django queryset select_related example.",
        "How do I improve my daily routine?",
        "Explain the concept of 'weekly review'.",
        "Write a JavaScript function that checks if a number is prime.",
        "What are the best ways to track time?",
        "Explain how to use the built-in type in Python.",
        "Write a FastAPI status code example.",
        "How do I create a personal project plan?",
        "Explain the term 'backlog grooming'.",
        "Write a Python function that checks if a string is a valid email.",
        "What is the difference between a milestone and a review?",
        "Explain how to write a team update.",
        "Write a bash script that finds files containing a word.",
        "How do I build a personal review habit?",
        "Explain the concept of 'planned review'.",
        "Write a React component that uses a progress hook.",
        "What are the best ways to plan a day off?",
        "Explain how to use the built-in bytes in Python.",
        "Write a SQL query to find top N per category.",
        "How do I create a personal dashboard?",
        "Explain the term 'delivery'.",
        "Write a Python script that checks if a string is a palindrome.",
        "What is the difference between a summary and a recap?",
        "Explain how to write a project status.",
        "Write a Django filter and ordering example.",
        "How do I improve my daily focus?",
        "Explain the concept of 'daily review'.",
        "Write a JavaScript function that checks if a string is a valid email.",
        "What are the best ways to prepare a plan?",
        "Explain how to use the built-in map in Python.",
        "Write a FastAPI env configuration example.",
        "How do I create a weekly log?",
        "Explain the term 'deliverable date'.",
        "Write a Python function that checks if a string is a valid integer.",
        "What is the difference between a plan and a version?",
        "Explain how to write a project milestone list.",
        "Write a bash script that greps and shows context.",
        "How do I build a personal routine?",
        "Explain the concept of 'regular review'.",
        "Write a React component that uses a memoized selector.",
        "What are the best ways to plan a project?",
        "Explain how to use the built-in open in Python.",
        "Write a SQL query to find users who haven't logged in.",
        "How do I create a personal tracker?",
        "Explain the term 'tracking'.",
        "Write a Python script that checks if a string is a valid password.",
        "What is the difference between a plan and a forecast?",
        "Explain how to write a daily summary.",
        "Write a Django prefetch_related example.",
        "How do I improve my meeting notes?",
        "Explain the concept of 'summary review'.",
        "Write a JavaScript function that checks if a string is a palindrome.",
        "What are the best ways to plan a year?",
        "Explain how to use the built-in float in Python.",
        "Write a FastAPI JSON example.",
        "How do I create a personal weekly plan?",
        "Explain the term 'deliverable status'.",
        "Write a Python function that checks if a string is a valid file name.",
        "What is the difference between a plan and a tracker?",
        "Explain how to write a progress report.",
        "Write a bash script that finds empty files.",
        "How do I build a personal review process?",
        "Explain the concept of 'weekly planning'.",
        "Write a React component that uses a form hook.",
        "What are the best ways to track goals?",
        "Explain how to use the built-in str in Python.",
        "Write a SQL query to find the longest transaction.",
        "How do I create a personal plan?",
        "Explain the term 'status'.",
    ]
    # de-dup while preserving order
    seen = set(); out = []
    for p in P + cur:
        key = p.strip().lower()
        if key not in seen:
            seen.add(key); out.append(p.strip())
    return out[:N_PROMPTS]

prompts = build_prompts()
log("Prompt list:", len(prompts))
if N_CHUNKS > 1:
    cs = -(-len(prompts) // N_CHUNKS)
    start = (CHUNK - 1) * cs
    end = min(start + cs, len(prompts))
    prompts = prompts[start:end]
    log("Generating chunk %d/%d: %d prompts" % (CHUNK, N_CHUNKS, len(prompts)))
json.dump(prompts, open("/kaggle/working/prompts.json", "w"), ensure_ascii=False)

# --------------------------------------------------
# 4) Teacher generation (incremental save)
# --------------------------------------------------
done = set()
if os.path.exists(CKPT):
    try:
        done = set(json.load(open(CKPT)))
    except Exception:
        done = set()
rows = []
if os.path.exists(OUT_JSONL):
    for ln in open(OUT_JSONL, encoding="utf-8"):
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass

TEACHER_SYS = ("You are a helpful, accurate AI assistant. Answer the user's question clearly and "
               "correctly. Use simple, direct language. If the question asks for code, give complete, "
               "runnable code with a short explanation. If it asks for advice, give practical steps. "
               "Keep answers concise (aim for 2-6 sentences unless the task needs more).")

def upload_dataset(folder, ds_id, title, meta_extra=None, retries=3):
    if not os.path.isdir(folder) or not any(os.scandir(folder)):
        log("upload_dataset: nothing to upload in", folder)
        return False
    meta = {"id": ds_id, "title": title, "isPrivate": False,
            "licenses": [{"name": "other"}], "updateFrequency": "never"}
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(folder, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f)
    for attempt in range(1, retries + 1):
        log("Uploading", ds_id, "(attempt %d/%d)..." % (attempt, retries))
        r = sh("kaggle datasets create -p %s -r zip --quiet" % folder)
        if r is not None and r.returncode == 0:
            log("Upload OK:", ds_id, "(created)")
            return True
        if r is not None and r.returncode != 0 and ("already exists" in (r.stdout + r.stderr).lower()):
            r = sh("kaggle datasets version -p %s -m update --quiet" % folder)
            if r is not None and r.returncode == 0:
                log("Upload OK:", ds_id, "(updated)")
                return True
            log("Version upload failed:", (r.stdout or "")[-200:], (r.stderr or "")[-200:])
        elif r is not None:
            log("Create failed:", (r.stdout or "")[-200:], (r.stderr or "")[-200:])
        time.sleep(20)
    log("Upload FAILED after", retries, "tries:", ds_id)
    return False

def gen_with(model):
    global rows
    todo = [p for p in prompts if (model + "||" + p) not in done]
    log(model, "- pending:", len(todo))
    for i, p in enumerate(todo):
        try:
            ans = ollama_chat([{"role": "system", "content": TEACHER_SYS},
                               {"role": "user", "content": p}], model, max_tokens=240)
        except Exception as e:
            log(model, "error on", i, str(e)[:120]); time.sleep(2); continue
        if not ans:
            continue
        if len(ans) > 1500:
            ans = ans[:1500].rsplit(" ", 1)[0]
        rows.append({"prompt": p, "answer": ans, "teacher": model})
        done.add(model + "||" + p)
        if len(rows) % 25 == 0:
            with open(OUT_JSONL, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            json.dump(list(done), open(CKPT, "w"))
            log("saved", len(rows), "rows so far")
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(list(done), open(CKPT, "w"))

for m in TEACHERS:
    gen_with(m)

log("Dataset rows:", len(rows))
os.makedirs("/kaggle/working/ds_data", exist_ok=True)
shutil.copy(OUT_JSONL, "/kaggle/working/ds_data/train.jsonl")
shutil.copy("/kaggle/working/prompts.json", "/kaggle/working/ds_data/prompts.json")
ok = upload_dataset("/kaggle/working/ds_data", DS_DATA_ID, "NextGen Distill Data")
log("Data upload:", "OK" if ok else "FAILED")

# Free disk: ollama + its model blobs are only needed for data generation.
sh("rm -rf /root/.ollama /usr/local/share/ollama /usr/share/ollama", silent=True)
sh("rm -f /tmp/ollama.tar /tmp/ollama.tar.zst", silent=True)

# --------------------------------------------------
# 4b) Merge chunks generated on the Colab accounts (if any)
# --------------------------------------------------
if MERGE_PARTS:
    part_root = "/kaggle/working/parts"
    os.makedirs(part_root, exist_ok=True)
    got = {}    # part dataset -> found
    deadline = time.time() + MERGE_WAIT_MIN * 60
    while time.time() < deadline and len(got) < len(MERGE_PARTS):
        for ds in MERGE_PARTS:
            if ds in got:
                continue
            d = os.path.join(part_root, ds.split("/")[-1])
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
            r = sh("kaggle datasets download -d %s -p %s --unzip --quiet" % (ds, d), timeout=900)
            if r is not None and r.returncode == 0 and os.path.exists(os.path.join(d, "train.jsonl")):
                got[ds] = d
                log("Found part:", ds)
        if len(got) < len(MERGE_PARTS):
            log("Parts so far:", len(got), "/", len(MERGE_PARTS), "- retrying in 90s...")
            time.sleep(90)
    if got:
        seen_keys = set((x["teacher"], x["prompt"].strip().lower()) for x in rows)
        extra = 0
        for ds, d in got.items():
            for ln in open(os.path.join(d, "train.jsonl"), encoding="utf-8"):
                try:
                    x = json.loads(ln)
                except Exception:
                    continue
                if (x.get("teacher"), x.get("prompt", "").strip().lower()) in seen_keys:
                    continue
                rows.append(x)
                seen_keys.add((x.get("teacher"), x.get("prompt", "").strip().lower()))
                extra += 1
        log("Merged", extra, "rows from Colab parts -> total", len(rows))
        with open(OUT_JSONL, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        log("No Colab parts arrived; proceeding with own chunk only:", len(rows), "rows")

# --------------------------------------------------
# 5) Fine-tune Mistral-7B
# --------------------------------------------------
def detect_gpu():
    try:
        out = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                             shell=True, capture_output=True, text=True, timeout=30)
        name = (out.stdout or out.stderr or "").strip().split("\n")[0]
    except Exception:
        name = ""
    low = name.lower()
    caps = {"p100": 60, "p40": 61, "k80": 37, "m60": 52,
            "v100": 70, "t4": 75, "l4": 89, "a100": 80, "h100": 90,
            "a6000": 86, "a5000": 86, "rtx": 86, "gpu": 75}
    for k, v in caps.items():
        if k in low:
            return v, name
    return 99, name

gpu_cap, gpu_name = detect_gpu()
log("GPU detected:", repr(gpu_name), "capability:", gpu_cap)

MAX_SEQ = 2048
data_rows = []
for ln in open(OUT_JSONL, encoding="utf-8"):
    try:
        data_rows.append(json.loads(ln))
    except Exception:
        pass
log("Loaded", len(data_rows), "training rows")

def build_ds():
    from datasets import Dataset
    ds = Dataset.from_list([{"prompt": r["prompt"], "answer": r["answer"]} for r in data_rows])
    def fmt(ex):
        return {"text": "<s>[INST] " + ex["prompt"] + " [/INST] " + ex["answer"] + "</s>"}
    return ds.map(fmt)

if gpu_cap and gpu_cap < 70:
    # ---- LEGACY PATH: P100 / K80 / P40 (no sm_70 kernels in modern torch) ----
    log("Old GPU (cap<70): installing torch 2.3.1+cu118 + standard LoRA stack (no unsloth)")
    sh("pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118", silent=False, timeout=1800)
    sh("pip install -q transformers==4.44.2 accelerate peft trl==0.9.4 datasets sentencepiece gguf", silent=False, timeout=1800)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    log("Loading", MODEL_ID, "(fp16 LoRA, gradient checkpointing)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    train_ds = build_ds()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        max_seq_length=1024,
        dataset_text_field="text",
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_ratio=0.05,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            max_grad_norm=1.0,
            output_dir="/kaggle/working/out",
            report_to=[],
        ),
    )
    log("Training (legacy path)...")
    trainer.train()
    log("Training done.")
    log("Merging LoRA...")
    merged = model.merge_and_unload()
    merged.save_pretrained("/kaggle/working/merged")
    tokenizer.save_pretrained("/kaggle/working/merged")

    log("Exporting GGUF (q8_0 via llama.cpp converter)...")
    shutil.rmtree("/kaggle/working/gguf", ignore_errors=True)
    os.makedirs("/kaggle/working/gguf", exist_ok=True)
    if not os.path.isdir("/kaggle/working/llama.cpp"):
        sh("git clone --depth 1 https://github.com/ggerganov/llama.cpp /kaggle/working/llama.cpp", silent=False, timeout=1800)
    conv = "/kaggle/working/llama.cpp/convert_hf_to_gguf.py"
    r = sh("cd /kaggle/working && python %s /kaggle/working/merged --outfile /kaggle/working/gguf/nextgen-trained.gguf --outtype q8_0" % conv, silent=False, timeout=1800)
    if r is None or r.returncode != 0 or not os.path.exists("/kaggle/working/gguf/nextgen-trained.gguf"):
        log("q8_0 failed; retrying as f16...")
        sh("cd /kaggle/working && python %s /kaggle/working/merged --outfile /kaggle/working/gguf/nextgen-trained.gguf --outtype f16" % conv, silent=False, timeout=1800)
    log("GGUF files:", os.listdir("/kaggle/working/gguf"))
else:
    # ---- MODERN PATH: T4 / L4 / A100 / H100 (unsloth) ----
    log("Installing unsloth...")
    sh("pip install -q unsloth", silent=False, timeout=1800)
    sh("pip install -q trl datasets", silent=False, timeout=1800)

    import torch

    from unsloth import FastLanguageModel, is_bfloat16_supported

    from transformers import TrainingArguments

    log("Loading Mistral-7B-Instruct-v0.3 (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_ds = build_ds()

    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        max_seq_length=MAX_SEQ,
        dataset_text_field="text",
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.05,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            average_tokens_across_devices=False,
            output_dir="/kaggle/working/out",
            report_to=[],
        ),
    )
    log("Training...")
    trainer.train()
    log("Training done.")

    # --------------------------------------------------
    # 6) Export GGUF (disk-safe: /kaggle/working is ~20GB)
    #    Merge into working; write GGUF intermediates to /root then cleanup.
    # --------------------------------------------------
    log("Exporting GGUF (q4_k_m)...")
    shutil.rmtree("/kaggle/working/gguf", ignore_errors=True)
    os.makedirs("/kaggle/working/gguf", exist_ok=True)
    shutil.rmtree("/kaggle/working/merged_f16", ignore_errors=True)
    log("Merging LoRA to 16-bit...")
    model.save_pretrained_merged("/kaggle/working/merged_f16", tokenizer, save_method="merged_16bit")
    sh("rm -rf ~/.cache/huggingface", silent=True)
    sh("rm -rf ~/.cache/pip", silent=True)
    conv = "/root/.unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py"
    f16 = "/root/nextgen_f16.gguf"
    r = sh("python %s --outfile %s /kaggle/working/merged_f16 --outtype f16" % (conv, f16), silent=False, timeout=3600)
    if r is None or r.returncode != 0 or not os.path.exists(f16):
        log("convert arg-order 1 failed; trying positional outfile...")
        r = sh("python %s /kaggle/working/merged_f16 --outfile %s --outtype f16" % (conv, f16), silent=False, timeout=3600)
    if r is None or r.returncode != 0 or not os.path.exists(f16):
        raise RuntimeError("GGUF f16 conversion failed")
    shutil.rmtree("/kaggle/working/merged_f16", ignore_errors=True)
    qz = None
    for p in ["/root/.unsloth/llama.cpp/build/bin/llama-quantize",
              "/root/.unsloth/llama.cpp/build/bin/Release/llama-quantize",
              "/root/.unsloth/llama.cpp/llama-quantize"]:
        if os.path.exists(p):
            qz = p
            break
    if qz is None:
        raise RuntimeError("llama-quantize not found")
    out_gguf = "/kaggle/working/gguf/nextgen-trained.gguf"
    r = sh("%s %s %s q4_k_m" % (qz, f16, out_gguf), silent=False, timeout=3600)
    if r is None or r.returncode != 0 or not os.path.exists(out_gguf):
        raise RuntimeError("llama-quantize failed")
    sh("rm -f %s" % f16, silent=True)
    log("GGUF exported:", str(os.path.getsize(out_gguf) // (1024**3)) + "GB")

# free VRAM/disk
import gc
del model, trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
sh("rm -rf /root/.ollama", silent=True)

# --------------------------------------------------
# 7) Upload GGUF to Kaggle dataset
# --------------------------------------------------
gguf_dir = "/kaggle/working/gguf"
files = os.listdir(gguf_dir)
log("GGUF files:", files)
src = os.path.join(gguf_dir, "nextgen-trained.gguf")
if not os.path.exists(src):
    for f in files:
        if f.endswith(".gguf"):
            os.rename(os.path.join(gguf_dir, f), src)
            break
if os.path.exists(src):
    os.makedirs("/kaggle/working/ds_model", exist_ok=True)
    shutil.copy(src, "/kaggle/working/ds_model/nextgen-trained.gguf")
    ok = upload_dataset("/kaggle/working/ds_model", DS_MODEL_ID, "NextGen Model")
    log("Model upload:", ok)
else:
    log("NO GGUF FOUND")
log("ALL DONE.")
