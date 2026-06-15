
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable
import os

MAX_ITERATION = 10

MODEL = "gpt-5.5"

@tool
def get_prodcut_price(product_name: str) -> str:
    """A tool that gets the price of a product from an e-commerce website."""
    # In a real implementation, this function would make an API call to an e-commerce website to get the price of the product.
    # For this example, we'll just return a hardcoded price.
    print(f"Getting the price of '{product_name}' in catalog..")
    prices = {
        "iPhone 14": 799,
        "MacBook Pro": 1299,
        "Keyboard": 99,
        "Laptop": 999,
    }
    return prices.get(product_name, "0")

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """A function that applies a discount to a dictionary of prices."""
    print(f"Applying a '{discount_tier}' discount to the price {price}...")
    discounted_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discounted_price = discounted_percentage.get(discount_tier, 0) * price / 100
    return discounted_price


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools =[get_prodcut_price, apply_discount]

    tools_dic = {tool.name: tool for tool in tools}

    llm =init_chat_model(
        model=MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a helpful shopping assistant."
                      "You have access to a product catalog tool and a discount tool.\n\n"
                      "STRICT RULES - you must follow these exactly:\n"
                      "1. NEVER guess or assume any product price."
                      "You MUST call get_pruduct_price first to get the real price.\n"
                      "2. Only call apply_discount AFTER you have received a price from get_product_price."
                      "Pass the exact price returned by get_product_price - do NOT pass a made-up number.\n"
                      "3. NEVER calculate the discount by yourself using math."
                      "Always use the apply_discount tool.\n"
                      "4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one."),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATION +1):
        print(f"----Iteration {iteration}----")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        #if no tool calls, we have our final answer
        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")


        print(f"Tool selected: {tool_name} with args {tool_args}")

        tool_to_use = tools_dic.get(tool_name)

        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max iterations reached without a final answer.")
    return None
    

if __name__ == "__main__":
    print("Hello from langchain-course - chapter tools!")
    result = run_agent("What is the price of the Laptop after applying a gold discount?")
    