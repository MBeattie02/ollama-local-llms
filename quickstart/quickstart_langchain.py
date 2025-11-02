from langchain_ollama import ChatOllama


def main() -> None:

    llm = ChatOllama(
        model="gpt-oss:20b",  # Use any local Ollama model name here
        temperature=0.2,
    )

    # Simple one-shot call
    result = llm.invoke("Why is the sky blue? Keep it concise.")
    print(result.content)


if __name__ == "__main__":
    main()


