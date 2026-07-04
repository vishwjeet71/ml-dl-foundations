# Chains
Chains are used to build pipelines for LLM applications, where the output of one component becomes the input of the next component. This simplifies the code, reduces development complexity, and makes applications more efficient and easier to maintain.

```
chain = prompt | model | output_parser -> RunnableSequence 
```

There are three main types of chains:

1. **Sequential Chain (Simple):** Components are executed one after another, where each step uses the output of the previous step.

2. **Parallel Chain:** Multiple components run simultaneously, and their outputs can be combined later.

3. **Conditional Chain:** The next component is selected based on a condition or the result of a previous step.


# Runnables

Runnables are objects that can **take an input, do some work, and return an output**. They provide a standard way to build AI workflows by connecting different steps together.

### 1. Task-Specific Runnables

These are runnables **built for a specific AI task**.

Examples:

- **Prompt Templates** – Format the input into a prompt.
- **Chat Models (LLMs)** – Generate responses.
- **Output Parsers** – Convert the model's output into a desired format.
- **Retrievers** – Fetch relevant documents from a knowledge base.

### 2. Runnable Primitives

These are basic building blocks **used to create workflows** by combining runnables.

Common examples:

- **RunnableSequence** – Runs steps one after another.
- **RunnableParallel** – Runs multiple steps at the same time.
- **RunnableBranch** – Chooses which runnable to execute based on a condition.
- **RunnableLambda** – Wraps a Python function.
- **RunnablePassthrough** – Passes the input through unchanged (often while adding extra data).