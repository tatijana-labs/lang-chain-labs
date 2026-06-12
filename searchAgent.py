from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str=Field(description="The URL of the source")
    
class AgentResponse(BaseModel):
    """Schema for the agent's response"""

    answer:str=Field(description="The agent's answer to the user's question")
    sources:List[Source]=Field(default_factory=list, description="A list of sources used by the agent to answer the question")    


llm = ChatOpenAI(

        model="gpt-5.5",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    response = result["structured_response"]
    print(response.answer)
    print("Sources:", [source.url for source in response.sources])


if __name__ == "__main__":
    main()