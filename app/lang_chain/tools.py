import json
import requests
from langchain.tools import tool
from pydantic import BaseModel
from app.tools.customers import get_customers, get_customer_cart
from app.tools.products import get_product, search_products, get_products_by_category


def to_text(result) -> str:
    # a tool must hand Claude text. our plain functions return one of 3 shapes,
    # so convert each one (same idea as tool_result_to_string in llm.py)
    if result is None:
        return "No data found"
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, list):
        return json.dumps([item.model_dump() for item in result])
    return str(result)


def call_api(function, *args) -> str:
    # run one of our plain functions and always hand back text.
    # a bad id (404), a timeout or no internet raises a requests error, and
    # LangChain would crash the whole run on it. so catch it and return the
    # error as text, then Claude can read it and answer "I could not find that".
    try:
        return to_text(function(*args))
    except requests.RequestException as error:
        return f"Error: {error}"


# @tool reads: name from the function name, description from the docstring,
# argument types from the type hints. the docstring is what Claude reads to
# decide WHEN to use the tool, so write it for Claude.
@tool
def get_customers_tool(customer_id: int) -> str:
    """Get one customer (name, email) by its numeric customer id."""
    return call_api(get_customers, customer_id)


@tool
def get_customer_cart_tool(customer_id: int) -> str:
    """Get a customer's shopping cart (products, quantities, prices, totals) by customer id.
    Use this to see what a customer has in their cart."""
    # get_customer_cart returns None when there is no cart, to_text handles that
    return call_api(get_customer_cart, customer_id)


@tool
def get_product_tool(product_id: int) -> str:
    """Get one product's details (title, category, price, rating, stock) by its numeric product id.
    Use this when you already know the exact product id."""
    return call_api(get_product, product_id)


@tool
def search_products_tool(query: str) -> str:
    """Search the product catalog by a free text keyword such as 'phone' or 'rolex'.
    Use this when the customer describes a product but you do not know its id or category."""
    # returns a list of Product, to_text turns the list into one JSON string
    return call_api(search_products, query)


@tool
def get_products_by_category_tool(category: str) -> str:
    """List all products in one exact category slug such as 'smartphones' or 'laptops'.
    Use this to browse a whole category. For keyword search use search_products_tool instead."""
    return call_api(get_products_by_category, category)


# the list we will hand to the agent in agent.py
LANGCHAIN_TOOLS = [
    get_customers_tool,
    get_customer_cart_tool,
    get_product_tool,
    search_products_tool,
    get_products_by_category_tool,
]


if __name__ == "__main__":
    # what @tool built for each tool from the function
    for t in LANGCHAIN_TOOLS:
        print("name:", t.name)  # from the function name
        print("description:", t.description)  # from the docstring
        print("args:", t.args)  # from the type hints
        print()

    # run the tools directly: input is a dict, like block.input in your manual loop
    print(get_customers_tool.invoke({"customer_id": 6}))
    print(get_customer_cart_tool.invoke({"customer_id": 6})[:200], "...")
    print(get_product_tool.invoke({"product_id": 1}))
    print(search_products_tool.invoke({"query": "phone"})[:200], "...")
    print(get_products_by_category_tool.invoke({"category": "smartphones"})[:200], "...")
