import requests
from pydantic import BaseModel

BASE_URL = "https://dummyjson.com"


class User(BaseModel):
    id: int
    firstName: str
    lastName: str
    email: str


class CartProduct(BaseModel):
    id: int
    title: str
    price: float
    quantity: int
    total: float
    discountPercentage: float
    discountedTotal: float
    thumbnail: str


class Cart(BaseModel):
    id: int
    products: list[CartProduct]
    total: float
    discountedTotal: float
    userId: int
    totalProducts: int
    totalQuantity: int


def get_customers(cust_id: int) -> User:
    """Fetch one customer by id, e.g. get_customers(6) -> Olivia Wilson."""
    # timeout stops the call from hanging forever if the API never answers
    response = requests.get(f"{BASE_URL}/users/{cust_id}", timeout=10)
    # a bad id returns HTTP 404, so raise an error instead of parsing the error message
    response.raise_for_status()
    # build the model here so the return type (-> User) is true; extra API keys are ignored
    return User.model_validate(response.json())


def get_customer_cart(cust_id: int) -> Cart | None:
    """Fetch the customer's first cart, or None if they have no cart."""
    response = requests.get(f"{BASE_URL}/users/{cust_id}/carts", timeout=10)
    response.raise_for_status()
    # the response is {"carts": [...], "total": ..., "skip": ..., "limit": ...}
    carts = response.json()["carts"]
    if not carts:
        # empty list: indexing [0] would raise IndexError, so return None instead
        return None
    return Cart.model_validate(carts[0])


if __name__ == "__main__":
    print(get_customers(6))
    print(get_customer_cart(6))

    # edge cases
    try:
        get_customers(9999)
    except requests.HTTPError as e:
        print("bad customer id ->", e)

    # an unknown user gives a 404 here too (DummyJSON never returns an empty carts list
    # for real users, so the "return None" branch above is a safety net for other APIs)
    try:
        get_customer_cart(9999)
    except requests.HTTPError as e:
        print("bad customer id for cart ->", e)
