from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
load_dotenv()
import os



llm = ChatOpenAI(
        model="gpt-5.5",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,)
tools= [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():    
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    print(result)
if __name__ == "__main__":    
    main()