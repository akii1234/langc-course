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

def exercise_multi_model():
    '''
    EXERCISE : Create a function that:
    1. Takes a question and a list of model names
    2. Gets a responses from all models
    3. Returns a dict of {model_name: response}

    Test with : question="What is AI?",models=["gpt-5-nano","gpt-4o-mini"]

    '''

    models = {
        "gpt-5-nano" : init_chat_model(
            model = "gpt-5-nano",
            model_provider="openai",
            temperature=0.1,
            streaming=False
        ),
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai",
            temperature=0.1,
            streaming=False
        ),
        "claude":init_chat_model(
            model="claude-haiku-5-5",
            model_provider="anthropic",
            streaming=False
        )
    }

    question = input("Enter your questions : ")
    user_models = input("Enter you selection , in case of multi-models use comma separated values : ")

    input_models = [m.strip() for m in user_models.split(",")]
    print(input_models)

    output = {}

    for model_name,model in models.items():
        if model_name in input_models:
            response = model.invoke(question)
            output[model_name] = response.content

    return output

exercise_multi_model()    