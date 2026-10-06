from pydantic import BaseModel, Field


# The shape of Claude's FINAL answer. Structured outputs forces every reply to match
# AgentAnswer, so one envelope covers every kind of question.
# Rule: fill only the part that matches the question, leave the rest as None.


class CustomerInfo(BaseModel):
    name: str
    email: str


class CartItem(BaseModel):
    title: str
    quantity: int
    price: float


class CartInfo(BaseModel):
    items: list[CartItem]
    total: float = Field(description="Cart total in dollars, before discount")


class ProductInfo(BaseModel):
    title: str
    price: float
    category: str


class AgentAnswer(BaseModel):
    # always present, so greetings and unclear questions still have a place to go
    answer: str = Field(description="One short sentence answering the question")
    # filled only when the question is about a customer
    customer: CustomerInfo | None = None
    # filled only when the question is about a cart
    cart: CartInfo | None = None
    # filled only when the question is about products
    products: list[ProductInfo] | None = None
