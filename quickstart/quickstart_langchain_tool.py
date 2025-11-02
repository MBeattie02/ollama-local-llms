from typing import Dict

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)


# Define two simple tools the model can call
@tool
def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Return the product of a and b."""
    return a * b


def main() -> None:
    llm = ChatOllama(
        # model="gpt-oss:20b",  # swap to a model you have locally (e.g., "gpt-oss:20b")
        model="llama3.2:3b",
        temperature=0.2,
    )

    # Bind tools to the model
    tools = [add, multiply]
    tools_by_name: Dict[str, object] = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    # Conversation: the model should decide to call tools to compute the result
    messages = [
        SystemMessage(content="You are a helpful math assistant. Use tools when needed."),
        HumanMessage(content="Compute (3 + 5) * 2 and explain briefly."),
    ]

    print("Invoking model (step 1)...")
    # First call: model may return tool calls
    try:
        ai_msg = llm_with_tools.invoke(messages)
    except Exception as e:
        print(f"Error during first invoke: {e}")
        return

    # If tools were requested, execute them and send results back
    if getattr(ai_msg, "tool_calls", None):
        tool_results = []
        for call in ai_msg.tool_calls:
            tool_name = call.get("name")
            args = call.get("args", {})
            tool_fn = tools_by_name[tool_name]
            observation = tool_fn.invoke(args)
            tool_results.append(
                ToolMessage(content=str(observation), tool_call_id=call["id"])
            )

        # Second call: pass observations to get the final, user-facing answer
        print("Invoking model (step 2 with tool results)...")
        try:
            final = llm_with_tools.invoke(messages + [ai_msg, *tool_results])
            print(final.content)
        except Exception as e:
            print(f"Error during second invoke: {e}")
    else:
        # Fallback: if no tools requested, just print the model reply
        print(ai_msg.content)


if __name__ == "__main__":
    main()


