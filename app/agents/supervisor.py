from langchain.agents import create_agent
from app.agents.common import MODEL
from app.agents.customer import ask_customer_agent
from app.agents.product import ask_product_agent
from app.agents.policy import ask_policy_agent
from app.llms.answer_schema import AgentAnswer

# the supervisor never touches the data itself. it only decides WHO to ask,
# asks one or more specialists, and combines their answers.
SUPERVISOR_PROMPT = """You are the supervisor of an online store support team.
You do not look up data yourself. Delegate to your specialists:
- ask_customer_agent: customer details and carts
- ask_product_agent: finding, comparing and recommending products
- ask_policy_agent: company policies (returns, refunds, warranty, shipping)

If a question needs more than one specialist, ask each of them, then combine
their answers into one reply. Specialists do not remember earlier requests, so
put everything they need (customer id, price limit, product) inside each request.
Never invent facts. If a specialist says it has no data, tell the customer that."""

supervisor = create_agent(
    model=MODEL,
    tools=[ask_customer_agent, ask_product_agent, ask_policy_agent],
    system_prompt=SUPERVISOR_PROMPT,
    # the final answer keeps the same structured shape as the single agent
    response_format=AgentAnswer,
)


if __name__ == "__main__":
    # the guard means importing `supervisor` elsewhere (FastAPI later) runs nothing
    questions = [
        "Can I return a smartphone I already opened if it works fine?",
        "Do refunds of $700 need approval?",
        "What is your shipping time?",
        "Customer 6 wants to return the Realme XT. Is that allowed, and what is it worth?",
    ]
    for question in questions:
        print("\n786 question ===>", question)
        result = supervisor.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        # every message in order: question, delegations, specialist answers, final
        for message in result["messages"]:
            message.pretty_print()

        print("786 final answer ===>")
        print(result["structured_response"].model_dump(exclude_none=True))
