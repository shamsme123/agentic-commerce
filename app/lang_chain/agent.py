from dotenv import load_dotenv
from langchain.agents import create_agent
from app.lang_chain.tools import LANGCHAIN_TOOLS
from app.llms.answer_schema import AgentAnswer
from pprint import pprint

load_dotenv()  # so ANTHROPIC_API_KEY is available

agent = create_agent(
    model="anthropic:claude-haiku-4-5",  # "provider:model"
    tools=LANGCHAIN_TOOLS,  # your 5 @tool functions
    system_prompt="You are a support agent. Use tools, never guess data.",
    response_format=AgentAnswer,  # same idea as output_format earlier
)

questions = [
    "Hello, how are you?",
    "name and email of customer 6?",
    "show me customer 6's cart and the cheapest phone under $400",
    "name of customer 9999",
]

for question in questions:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"{question}"}]},
    )

for message in result["messages"]:
    message.pretty_print()
