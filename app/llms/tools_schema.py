from pydantic import BaseModel, Field


# Each tool's INPUT is a Pydantic model. Claude only sees the JSON Schema we generate
# from it, so the Field(description=...) text is what tells Claude what to pass in.
class GetCustomerInput(BaseModel):
    cust_id: int = Field(description="The numeric customer id, e.g. 6")


class GetCustomerCartInput(BaseModel):
    cust_id: int = Field(description="The numeric customer id whose cart to fetch")


class GetProductInput(BaseModel):
    product_id: int = Field(description="The numeric product id, e.g. 1")


class SearchProductsInput(BaseModel):
    query: str = Field(description="Search text, e.g. 'phone' or 'rolex'")


class GetProductsByCategoryInput(BaseModel):
    category: str = Field(description="Product category slug, e.g. 'smartphones'")


def to_anthropic_tool(name: str, description: str, input_model: type[BaseModel]) -> dict:
    """Turn a Pydantic input model into the dict shape the Anthropic API expects.

    Anthropic wants {"name", "description", "input_schema"}, where input_schema is
    plain JSON Schema. model_json_schema() builds that from the model for us.
    """
    return {
        "name": name,
        "description": description,
        "input_schema": input_model.model_json_schema(),
    }


# The list we will pass to client.messages.create(tools=TOOLS)
TOOLS = [
    to_anthropic_tool(
        "get_customer",
        "Get a customer's name and email by customer id.",
        GetCustomerInput,
    ),
    to_anthropic_tool(
        "get_customer_cart",
        "Get a customer's cart (products, quantities, prices, totals) by customer id.",
        GetCustomerCartInput,
    ),
    to_anthropic_tool(
        "get_product",
        "Get one product's details by product id.",
        GetProductInput,
    ),
    to_anthropic_tool(
        "search_products",
        "Search products by a text query.",
        SearchProductsInput,
    ),
    to_anthropic_tool(
        "get_products_by_category",
        "List all products in a category.",
        GetProductsByCategoryInput,
    ),
]


if __name__ == "__main__":
    import json

    # quick look at exactly what Claude will receive
    print(json.dumps(TOOLS[0], indent=2))
