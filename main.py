import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-5.2")
vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.getenv("index_name"))

retriver = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt_template = ChatPromptTemplate.from_template(
    """"Answer the question base only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""

)

def format_docs(docs):
    """Format retrived documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)



def retrieval_chain_without_lcel(query: str):
    """ A simple retrieval chain without LCEL
        Manually retrives documents, formats them, and generates a response.

        Limitations:
        - Manual step-by-step execution
        - No build-in streaming support
        - No sync support without additional code
        - Harder to compose with other chaings
        - More verbose and error-prone
    """
     # Step 1: Retrieve relevant documents
    docs = retriver.invoke(query)

    # Step 2: Format the retrived documents into a single string
    context = format_docs(docs)

    # Step 3: Create the prompt with the formatted context and the question
    messages = prompt_template.format(context=context, question=query)

    response = llm.invoke(messages)

    return response.content

def create_retrival_chain_with_lcel():
    """ A retrieval chain using LCEL
        Returns a chain that can be invoked with ("question": "...")

        Benefits:
        - Encapsulates the entire retrieval and generation process
        - Built-in support for streaming and sync execution
        - Easier to compose with other chains
        - More concise and less error-prone
    """
    retrival_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriver | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrival_chain

if __name__ == "__main__":
    print("Retriving....")

    question = "What is pinecone in machine learning?"

    #====================================================
    # OPTION 1
    #====================================================

    print("\n"+"=" * 70)
    print("Option 1: Raw LLM invocation")
    print("\n"+"=" * 70)

    result_rew = llm.invoke([HumanMessage(content=question)])
    print("\nAnswer:")
    print(result_rew.content)

    #====================================================
    # OPTION 2 - Without LangChain Execution Language (LCEL)
    #====================================================

    print("\n"+"=" * 70)
    print("Option 1: Without LCEL")
    print("\n"+"=" * 70)

    result_without_lcel = retrieval_chain_without_lcel(question)
    print("\nAnswer:")  
    print(result_without_lcel)

    #====================================================
    # OPTION 3 - With LangChain Execution Language (LCEL)
    #====================================================

    print("\n"+"=" * 70)
    print("Option 1: With LCEL")
    print("\n"+"=" * 70)

    result_with_lcel = create_retrival_chain_with_lcel().invoke({"question": question})
    print("\nAnswer:")  
    print(result_with_lcel)


