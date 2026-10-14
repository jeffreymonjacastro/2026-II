Hi Robert – thanks for reaching out!  
Below is a step‑by‑step guide on how to create a Crew, kick it off, and add memory so your agents can remember past interactions, facts, and decisions. I’ve pulled the relevant information straight from the CrewAI docs so you can reference the official sources whenever you need.

---

## 1️⃣  Create a Crew

```python
from crewai import Crew, Agent, Task, Process

# 1. Define your agents
researcher = Agent(
    role="Researcher",
    goal="Collect up‑to‑date information about the topic",
    backstory="You are a seasoned data scientist."
)

writer = Agent(
    role="Writer",
    goal="Turn research into a polished report",
    backstory="You are a professional technical writer."
)

# 2. Define the tasks
task1 = Task(
    description="Gather all relevant data about the topic",
    expected_output="A list of key facts and sources."
)

task2 = Task(
    description="Write a concise report using the gathered facts",
    expected_output="A final report document."
)

# 3. Assemble the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential,   # optional – defaults to sequential
    verbose=True                  # optional – turn on logging
)
```

> **Reference**: Crew attributes and creation details – <https://docs.crewai.com/en/concepts/crews>

---

## 2️⃣  Add Memory to the Crew

CrewAI’s unified `Memory` class lets agents share a common knowledge base.  
You can either use the default memory (auto‑configured) or supply a custom `Memory` instance.

### 2.1  Using the Default Memory

```python
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    memory=True          # <‑‑ enables the default memory
)
```

> **Reference**: “With Crews – Pass memory=True for default settings” – <https://docs.crewai.com/en/concepts/memory>

### 2.2  Using a Custom Memory Instance

```python
from crewai import Memory

# Create a memory with custom settings (e.g., faster recall)
custom_memory = Memory(
    recency_weight=0.5,
    recency_half_life_days=7,
    embedder={"provider": "openai"}   # optional – use a different embedder
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    memory=custom_memory
)
```

> **Reference**: “With Crews – Pass a configured Memory instance” – <https://docs.crewai.com/en/concepts/memory>

---

## 3️⃣  Kick Off the Crew

```python
# Kick off synchronously
result = crew.kickoff(
    inputs={"topic": "Artificial Intelligence"}   # any inputs your tasks need
)

print("Final result:", result)
```

> **Reference**: “Kickoff Crew” – <https://docs.crewai.com/en/concepts/crews>

If you want real‑time streaming of the crew’s output:

```python
stream = crew.kickoff(
    inputs={"topic": "Artificial Intelligence"},
    stream=True
)

for chunk in stream:
    print(chunk.content, end="", flush=True)

final_result = stream.result
```

> **Reference**: “Streaming Crew Execution” – <https://docs.crewai.com/en/concepts/crews>

---

## 4️⃣  Working with Memory Inside the Crew

### 4.1  Remembering Facts

Agents can call `memory.remember()` to store new facts.  
If you want to add a fact manually before kickoff:

```python
crew.memory.remember("The project deadline is 30th September.")
```

### 4.2  Recalling Information

During a task, an agent can ask the crew’s memory:

```python
matches = crew.memory.recall("What is the project deadline?")
for m in matches:
    print(f"[{m.score:.2f}] {m.record.content}")
```

### 4.3  Resetting Memory

If you need a fresh start (e.g., for a new project):

```python
crew.memory.reset()          # clears all scopes
# or reset a specific subtree
crew.memory.reset(scope="/project/old")
```

> **Reference**: “Reset unified memory” – <https://docs.crewai.com/en/concepts/memory>

---

## 5️⃣  Quick Checklist

| Step | What to Do | Where to Find It |
|------|------------|------------------|
| 1 | Define agents & tasks | <https://docs.crewai.com/en/concepts/crews> |
| 2 | Create a Crew object | Same page |
| 3 | Enable memory (`memory=True` or custom `Memory`) | <https://docs.crewai.com/en/concepts/memory> |
| 4 | Kick off (`crew.kickoff(...)`) | Same page |
| 5 | (Optional) Stream output | Same page |
| 6 | Use `memory.remember()` / `memory.recall()` inside tasks | Same page |
| 7 | Reset memory if needed | Same page |

---

### 🎉 Final Thoughts

- **Memory is optional** – if you don’t need shared context, you can skip it.  
- **Custom memory** gives you control over recall speed, embedding provider, and scoring weights.  
- **Agents can also have their own memory** if you want isolated knowledge bases per agent (not covered here but supported).

Feel free to copy the snippets above into your project, tweak the agent roles or task descriptions, and let me know if you hit any snags. Happy building! 🚀

---

**Sources**

- Crew concepts: <https://docs.crewai.com/en/concepts/crews>  
- Memory concepts: <https://docs.crewai.com/en/concepts/memory>