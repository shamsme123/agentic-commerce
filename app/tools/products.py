import requests
from pydantic import BaseModel

BASE_URL = "https://dummyjson.com"


# Only the fields the agent needs. Pydantic ignores every other key in the API response.
class Product(BaseModel):
    id: int
    title: str
    category: str
    price: float
    rating: float
    stock: int


# The search endpoint wraps the products in a dict: {"products": [...], "total": 23, ...}.
# Note: this "total" is the NUMBER of matches, not money.
class ProductSearchResponse(BaseModel):
    products: list[Product]
    total: int
    skip: int
    limit: int


def get_product(product_id: int) -> Product:
    """Fetch one product by id, e.g. get_product(130) -> Realme XT."""
    response = requests.get(f"{BASE_URL}/products/{product_id}", timeout=10)
    # A bad id returns HTTP 404, so raise an error instead of parsing an error message
    response.raise_for_status()
    return Product.model_validate(response.json())


def search_products(query: str) -> list[Product]:
    """Search the catalog, e.g. search_products("phone"). Returns a list of Product."""
    response = requests.get(
        f"{BASE_URL}/products/search", params={"q": query}, timeout=10
    )
    response.raise_for_status()
    # Validate the whole response, then return only the list of products
    validated = ProductSearchResponse.model_validate(response.json())
    return validated.products


def get_products_by_category(category: str) -> list[Product]:
    """List products in one category, e.g. get_products_by_category("smartphones").
    Unlike search, this returns only real phones, not chargers or cases."""
    response = requests.get(f"{BASE_URL}/products/category/{category}", timeout=10)
    response.raise_for_status()
    # Same wrapper shape as the search endpoint, so we reuse the same model
    validated = ProductSearchResponse.model_validate(response.json())
    return validated.products


if __name__ == "__main__":
    # Test 1: one product
    print(get_product(130))

    # Test 2: search, then show products cheaper than the Realme XT ($349.99)
    results = search_products("phone")
    print(f"{len(results)} products found")
    for p in results:
        if p.price < 349.99:
            print(f"cheaper: {p.title} ${p.price}")

    # Test 3: only smartphones cheaper than the Realme XT, cheapest first
    phones = get_products_by_category("smartphones")
    cheaper_phones = sorted(
        (p for p in phones if p.price < 349.99), key=lambda p: p.price
    )
    print(f"{len(cheaper_phones)} smartphones cheaper than the Realme XT:")
    for p in cheaper_phones:
        print(f"  {p.title} ${p.price}")
