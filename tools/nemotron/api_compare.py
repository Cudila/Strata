# Stage 7: the same 25 prompts through llama-swap's OpenAI API to two slots (Strata and the llama.cpp reference on the
# Usage: python tools/nemotron/api_compare.py <model> [<reference model>]   (API_URL overrides the endpoint)
# same GGUF), greedy; per prompt: the checked answer (expected substrings), the common prefix of the two answers.
import json, os, sys, time, urllib.request

U = os.environ.get("API_URL", "http://127.0.0.1:1234/v1/chat/completions")
P = [
    ("Какая столица Австралии? Одно слово.", ["Канберр"]),
    ("Сколько будет 17 * 23? Только число.", ["391"]),
    ("What is the chemical symbol for gold? One word.", ["Au"]),
    ("Кто написал роман «Война и мир»? Только фамилия.", ["Толст"]),
    ("Решите уравнение 3x + 7 = 31. Ответ: x = ?", ["8"]),
    ("Translate to English: «Кошка спит на диване». Only the translation.", ["cat", "sofa|couch"]),
    ("Write a Python function is_prime(n) that returns True for primes. Only code.", ["def is_prime"]),
    ("Сколько секунд в одних сутках? Только число.", ["86400|86 400|86,400"]),
    ("What is the largest planet in the Solar System? One word.", ["Jupiter"]),
    ("Какой газ преобладает в атмосфере Земли? Одно слово.", ["Азот|азот"]),
    ("If a train goes 120 km in 1.5 hours, what is its average speed in km/h? Number only.", ["80"]),
    ("Назови три основных цвета модели RGB через запятую.", ["красн|Красн", "зел|Зел", "син|Син"]),
    ("Sort these numbers ascending: 42, 7, 19, 3, 25. Output only the list.", ["3, 7, 19, 25, 42|3,7,19,25,42"]),
    ("Какое число идёт следующим: 2, 4, 8, 16, ...? Только число.", ["32"]),
    ("In what year did the first human land on the Moon? Number only.", ["1969"]),
    ("Напиши SQL-запрос, который выбирает все строки из таблицы users, где age > 30. Только запрос.", ["SELECT", "users", "30"]),
    ("What is the derivative of x^3 with respect to x?", ["3x^2|3x²|3 x^2|3*x^2|3x\\^2"]),
    ("Переведи на русский: «The weather is nice today». Только перевод.", ["погод"]),
    ("How many sides does a hexagon have? Number only.", ["6"]),
    ("Какая самая длинная река в мире по общепринятым данным: Нил или Амазонка? Одно слово.", ["Нил|Амазонк"]),
    ("Explain in one sentence what a hash table is.", ["key|ключ"]),
    ("Сколько будет 2 в степени 10? Только число.", ["1024"]),
    ("What is the boiling point of water at sea level in Celsius? Number only.", ["100"]),
    ("Напиши на bash однострочник, который считает количество строк в файле data.txt.", ["wc -l"]),
    ("A rectangle is 7 by 9. What is its area? Number only.", ["63"]),
]


def ask(model, q):
    body = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": 1024, "temperature": 0,
            "top_k": 1}
    r = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    j = json.load(urllib.request.urlopen(r, timeout=1800))
    m = j["choices"][0]["message"]
    return (m.get("content") or "").strip(), (m.get("reasoning_content") or ""), time.time() - t


def ok(ans, exp):
    import re
    return all(re.search(e, ans) for e in exp)


models = sys.argv[1:]
res = {m: [ask(m, q) for q, _ in P] for m in models}
json.dump(res, open(sys.argv[0].replace(".py", ".json"), "w"), ensure_ascii=False, indent=1)
for i, (q, exp) in enumerate(P):
    row = [f"{i + 1:2d}"]
    for m in models:
        a = res[m][i][0]
        row.append(("OK " if ok(a, exp) else "BAD") + f" {a[:40]!r}")
    if len(models) == 2:
        a, b = res[models[0]][i][0], res[models[1]][i][0]
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
        row.append(f"same={a == b} prefix={n}/{max(len(a), len(b))}")
    print(" | ".join(row))
for m in models:
    print(m, "correct", sum(ok(res[m][i][0], e) for i, (_, e) in enumerate(P)), "/", len(P),
          f"time {sum(x[2] for x in res[m]):.0f}s")
if len(models) == 2:
    print("identical answers", sum(res[models[0]][i][0] == res[models[1]][i][0] for i in range(len(P))), "/", len(P))
