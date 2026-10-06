from dotenv import load_dotenv

# langchain reads ANTHROPIC_API_KEY from the environment, so load .env once here.
# every agent file imports from this module, so it always runs first.
load_dotenv()

MODEL = "anthropic:claude-haiku-4-5"  # "provider:model", Haiku for testing


def message_text(message) -> str:
    # an AI message's content is either a plain string or a list of blocks
    # like [{"type": "text", "text": "..."}]. we always want just the text.
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if block.get("type") == "text")


def run_subagent(agent, request: str) -> str:
    # a specialist agent works like any agent: invoke with one input dict.
    # each call is a fresh conversation, a specialist does NOT remember earlier
    # requests, so the supervisor must put everything it needs inside `request`.
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    # the last message is the specialist's final answer, give the supervisor only that
    return message_text(result["messages"][-1])
