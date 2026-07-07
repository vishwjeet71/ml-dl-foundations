# Agent

An AI agent is an intelligent system that receives a high-level goal from a user and autonomously plans, decides, and executes a sequence of actions by using external tools, APIs, or knowledge sources while maintaining context, reasoning over multiple steps, adapting to new information, and optimizing for the intended outcome.


## Characteristics of an AI Agent

### 1. Goal-driven

The user only tells the agent **what** they want to achieve, not **how** to achieve it.

The agent decides the required steps on its own.

**Example**

```text
Goal: Find the cheapest flight from Delhi to Mumbai.

Agent:
- Search flight websites
- Compare prices
- Choose the cheapest flight
- Return the result
```

### 2. Autonomous Planning

The agent automatically breaks a complex problem into smaller tasks and decides the order in which they should be executed.

Instead of following predefined instructions, it creates its own execution plan.

**Example**

```text
Goal: Create a travel itinerary.

Plan:
1. Search tourist places
2. Check weather
3. Find hotels
4. Estimate budget
5. Generate itinerary
```

### 3. Tool Using

AI agents can use external tools whenever additional information or computation is needed.

Common tools include:

- Web Search
- APIs
- Databases
- Python
- Calculators
- File Readers

Without tools, an LLM only answers using its trained knowledge.

With tools, it can access real-time information and perform actions.

## 4. Context-aware

The agent remembers previous conversations, intermediate results, and completed steps.

This memory helps it make better decisions throughout the task.

**Example**

```text
Step 1:
Find restaurants near my hotel.

Step 2:
Book the highest-rated restaurant.

The agent remembers which hotel was selected in Step 1.
```

## 5. Adaptive

The agent can change its plan when conditions change.

If a tool fails or new information becomes available, it replans instead of stopping.

**Example**

```text
Original Plan:
Use Weather API A

API A fails

Agent:
Switches to Weather API B
```

---

# ReAct (Reasoning + Acting)

ReAct is a design pattern used in AI agents.

It stands for **Reasoning + Acting**, where the LLM alternates between thinking and performing actions.

Instead of generating the final answer immediately, the agent repeatedly:

1. Thinks about the problem.
2. Decides the next action.
3. Uses a tool if needed.
4. Observes the result.
5. Continues until the final answer is ready.

## ReAct Workflow

```text
Question
    │
    ▼
Thought
    │
    ▼
Action (Tool)
    │
    ▼
Observation
    │
    ▼
Thought
    │
    ▼
Action
    │
    ▼
Final Answer
```

---

# Agent vs Agent Executor

## Agent

The **Agent** is the brain of the system.

Its responsibilities are:

- Understand the user's request.
- Decide what to do next.
- Select the appropriate tool.
- Generate the final answer.

> The Agent only makes decisions. It does **not** execute tools.

## Agent Executor

The **Agent Executor** controls the complete execution loop.

It acts as the bridge between the **Agent** and the available tools.

### Responsibilities

1. Sends the user's input and chat history to the agent.
2. Receives the next action from the agent.
3. Executes the selected tool.
4. Stores the tool's output **(Observation)**.
5. Sends the updated history back to the agent.
6. Repeats the loop until the agent returns the **Final Answer**.