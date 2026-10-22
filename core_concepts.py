from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def demo_basic_chain():
    """Demonstrate a basic chat template"""

    # Component 1 : Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant .Answer in one sentence {question}")    

    model = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0.7
        )
    parser = StrOutputParser()

    # Compose with pipe operator

    chain = prompt | model | parser

    # execute the chain with input

    result = chain.invoke({"question": "What is LangChain?"})

    print(f"Response : {result}")

    return chain


def demo_batch_execution():
    prompt = ChatPromptTemplate.from_template("Translate to french: {text}")
    model = ChatOpenAI(model="gpt-5-nano",temperature=0.1)
    parser = StrOutputParser()

    chain = prompt | model | parser

    inputs = [
        {"text": "Hello , how are you ?"},
        {"text": "What is your name ?"},
        {"text": "Where is the nearest restaurant ?"}
    ]
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0] ['text']} => Output: {text[1]}")

    return chain

def demo_streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    # model = ChatAnthropic(
    #     model_name="claude-haiku-5-5")
    model = ChatOpenAI(
        model="gpt-5-nano",
        temperature="0.1"
    )
    parser = StrOutputParser()

    chain = prompt | model | parser

    print("Streaming Output: ")
    for chunk in chain.stream({"topic":"nature"}):
        print(chunk,end="",flush=True)
    print()  # for newline after streaming  

    return chain


def demo_schema_inspection():
    """Demonstrate input/output schema inspection"""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(
            model="gpt-5-nano",
            temperature="0.1"
        )
    parser = StrOutputParser()
    
    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input schema: {input_schema}")
    print(f"Output schema: {output_schema}")


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_execution()
    # demo_streaming()
    demo_schema_inspection()

