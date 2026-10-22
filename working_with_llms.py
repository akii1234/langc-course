import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv()

def demo_init_chat_model():
    chat_model = init_chat_model(
        model="gpt-5-nano",
        temperature=0.1,
        streaming=True,
        max_retries=3
    )

    response = chat_model.invoke(
        "What is the capital of France ? Answer in one word"
        )

    print(f"Response : {response.content}")
    return 
    
def demo_model_comparison():
    prompt = "Explain Docker to me in 1 sentence?"
    models = {"gpt" : init_chat_model(
        model="gpt-5-nano",
        model_provider="openai",
        temperature=0.1,
        streaming=False
    ),
    "gpt-4o-mini" : init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",
        temperature=0.7,
        streaming=False
    )
    }
    # add anthropic if available
    if ("ANTHROPIC_API_KEY"):
        models["claude"] = init_chat_model(
            model="claude-haiku-5-5",
            model_provider="anthropic",
            streaming=False
        )

    print(f"Prompt : {prompt}\n")
    for model_name,model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")    


def demo_message():
    model = ChatOpenAI(model="gpt-5-nano",temperature=0)

    #using message objects (more control over roles)
    # system_msg = SystemMessage(
    #     content="You are an helpful assistant , specifically created to answer " \
    #              "domain questions related to banking and financial services "
    #              "if someone asks " \
    #              "any other questions instead of answering some lengthy explanation " \
    #              "Answer that I am domin specialist chatbot"
    # )
    system_message = SystemMessage(
        content="You are a pirate . Always answer like a pirate"
    )
    human_message = HumanMessage(content="What's the weather today?")

    messages = [
        system_message,human_message
    ]

    # print("Using messages objects:")
    # print(f"Messages: {messages}")

    response = model.invoke(messages)
    print(f"Response from the assistant: {response.content}\n")

    

    # Multi-turn conversation using message objects
    messages.append(response)
    print("continuing the conversation\n")
    messages.append(HumanMessage(content="What about tomorrow ? City is Mumbai"))

    print("\nMulti turn conversations")
    response = model.invoke(messages)
    print(f"follow up response from the pirate : {response.content}")



demo_message()