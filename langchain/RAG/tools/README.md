# [Tools](https://docs.langchain.com/oss/python/integrations/tools)


A **Tool** is a Python function (or API) that is wrapped in a format that an **LLM can understand and call whenever it needs to perform a specific task**.

Without tools, an LLM can only generate text.

With tools, it can perform real actions like:
- Search the web
- Query databases
- Run Python code
- Call APIs
- Send emails
- Execute shell commands

# Types of Tools

LangChain provides two categories of tools:

```text
Tools
├── Built-in Tools
└── Custom Tools
```

# 1. Built-in Tools

Built-in tools are **already implemented by LangChain**.

You only need to **import and use them**—no need to write the underlying logic.

## Examples

| Tool | Purpose |
|------|---------|
| `DuckDuckGoSearchRun` | Search the web |
| `WikipediaQueryRun` | Get Wikipedia summaries |
| `PythonREPLTool` | Execute Python code |
| `ShellTool` | Run shell commands |
| `RequestsGetTool` | Send HTTP GET requests |
| `GmailSendMessageTool` | Send Gmail emails |
| `SlackSendMessageTool` | Send Slack messages |
| `SQLDatabaseQueryTool` | Execute SQL queries |

### When to Use

- Common tasks
- Standard integrations
- Quick development

# 2. Custom Tools

Custom tools are **tools you create yourself**.

Use them when the built-in tools cannot solve your problem.

### Common Use Cases

- Call your own APIs
- Implement business logic
- Access your database
- Interact with your application
- Perform custom calculations

### Ways to Create Custom Tools

There are multiple ways to create custom tools in LangChain, but the **three most commonly used** are:

```text
Custom Tools
├── @tool Decorator
├── StructuredTool + Pydantic
└── BaseTool Class
```

## 1. Using `@tool` Decorator

The **simplest and most common** way to create a tool.

Just decorate a Python function with `@tool`, and LangChain automatically converts it into a tool.

### Best For

- Simple functions
- Single or few inputs
- Quick development

### Example Use Cases

- Add two numbers
- Convert temperature
- Get current time

## 2. Using `StructuredTool` + `Pydantic`

A **Structured Tool** accepts inputs in a **well-defined schema**.

The input schema is usually created using a **Pydantic model**.

Instead of passing plain text, the LLM sends structured fields.

### Why Use It?

- Multiple input parameters
- Input validation
- Clear descriptions for every field
- More reliable tool calling

### Best For

- Multiple inputs
- Complex tools
- APIs requiring structured data


## 3. Using `BaseTool`

`BaseTool` is the **base (abstract) class** from which all LangChain tools are built.

It defines the standard interface every tool must follow.

> `@tool` and `StructuredTool` are built on top of `BaseTool`.

### Why Use It?

Use `BaseTool` when you need full control over how the tool behaves.

### Examples

- Custom execution logic
- State management
- Async support
- Advanced validation
- Highly customized tools

### Best For

- Large applications
- Complex production systems
- Maximum customization

# Tool (Binding, Calling, Execution)

## Tool Binding

Tool Binding is the step where you register tools with a Language Model (LLM) so that:
1. The LLM knows what tools are available
2. It knows what each tool does (via description)
3. It knows what input format to use (via schema)

## Tool Calling

Tool Calling is the **process** where the LLM (language model) decides, during a conversation or task, that it needs to use a specific tool (function) and generates a structured output with:
- the name of the tool
- and the arguments to call it with

> The LLM does not actually run the tool it just suggests the tool and the input arguments. The actual execution is handled by LangChain or you.

## Tool Execution 

Tool Execution is the step where the actual Python function (tool) is run using the input arguments that the LLM suggested during tool calling.