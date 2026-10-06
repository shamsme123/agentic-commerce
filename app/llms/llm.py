import json
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from app.llms.tools_schema import TOOLS
from app.llms.answer_schema import AgentAnswer
from pprint import pprint
from anthropic.types import Message, TextBlock, ToolUseBlock
from app.tools.customers import get_customers, get_customer_cart
from app.tools.products import get_product, get_products_by_category, search_products

load_dotenv()
client = Anthropic()
MODEL = "claude-haiku-4-5"


MAX_ROUNDS = 5  # safety limit so a bug can never loop forever and burn credits

# tool name (the string Claude sends in block.name) -> the real python function
TOOLS_FUNCTIONS = {
    "get_customer": get_customers,
    "get_customer_cart": get_customer_cart,
    "get_product": get_product,
    "search_products": search_products,
    "get_products_by_category": get_products_by_category,
}


def identify_block(block, response: Message):
    if isinstance(block, ToolUseBlock):
        return block.type
    if isinstance(block, TextBlock):
        return block.type


def is_tool_use_called(response: Message):
    if response.stop_reason == "tool_use":
        return True


def add_user_message(prompt: str, messages: list[dict]):
    user_message = {"role": "user", "content": prompt}
    messages.append(user_message)
    return messages


def add_assistant_message(prompt: str, messages: list[dict]):
    assistant_message = {"role": "assistant", "content": prompt}
    messages.append(assistant_message)
    return messages


def call_anthropic(messages: list[dict]) -> Message:
    # 1000 (not 300) because answers about carts and product lists are longer
    params = {"model": MODEL, "max_tokens": 1000, "messages": messages}
    # parse() instead of create(): output_format makes the final reply match AgentAnswer
    response = client.messages.parse(**params, tools=TOOLS, output_format=AgentAnswer)
    return response


def tool_result_to_string(result) -> str:
    # our tools return a model, a list of models, or None.
    # a tool_result "content" must be a string, so convert each case.
    if result is None:
        return "No data found"
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, list):
        return json.dumps([item.model_dump() for item in result])
    return str(result)


def run_tool(block: ToolUseBlock) -> str:
    # block.name is a string like "get_customer", use it to find the real function
    function = TOOLS_FUNCTIONS[block.name]
    # block.input is a dict like {"cust_id": 6}, ** unpacks it into cust_id=6
    result = function(**block.input)
    return tool_result_to_string(result)


def build_tool_result(block: ToolUseBlock) -> dict:
    # tool_use_id must match the id of the ToolUseBlock Claude sent,
    # that is how Claude knows which request this result belongs to
    try:
        content = run_tool(block)
        is_error = False
    except Exception as error:
        # a bad customer id raises HTTPError, tell Claude instead of crashing
        content = f"Error: {error}"
        is_error = True
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": content,
        "is_error": is_error,
    }


def print_final_answer(response):
    # parsed_output is already a validated AgentAnswer object, not a string.
    # model_dump() gives a dict, exclude_none hides the parts that were not used
    answer = response.parsed_output
    pprint(answer.model_dump(exclude_none=True), sort_dicts=False)


def run_agent(question: str):
    # one fresh conversation per question, then the agent loop from Phase 3
    messages = []
    add_user_message(question, messages)
    response = call_anthropic(messages)

    # keep going while Claude asks for tools
    rounds = 0
    while rounds < MAX_ROUNDS:
        # Claude gave a final answer (end_turn), nothing more to run
        if not is_tool_use_called(response):
            break
        rounds += 1

        # 1. save Claude's tool request exactly as received
        add_assistant_message(response.content, messages)

        # 2. run every tool Claude asked for (can be more than one),
        # fresh list each round so old results are not sent twice
        tools_results = []
        for block in response.content:
            if identify_block(block, response) == "tool_use":
                print("786 running tool ===>", block.name, block.input)
                tools_results.append(build_tool_result(block))

        # 3. send all results back in ONE user message
        add_user_message(tools_results, messages)

        # 4. ask Claude again, it now sees the tool results
        response = call_anthropic(messages)

    return response


if __name__ == "__main__":
    questions = [
        "Hello! how are you?",
        "whats the name and email of customer id 6?",
        "whats in the cart of customer id 6?",
        "show me 3 smartphones",
    ]
    for question in questions:
        print("\n786 question ===>", question)
        response = run_agent(question)
        print_final_answer(response)
