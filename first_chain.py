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

def exercise_first_chain():
    """
        EXCERCISE : Create a chain that:
        1. Takes a product name and target audience
        2. Generates a marketing tagline
        3. Returns just the tagline as a string

        Test with : product="AI Course", audience="developers"
    """
    prompt = ChatPromptTemplate.from_template(
        "You are an helpful assistant. Generates a single marketing tagline " \
        "based on '{product}' targeting the audience '{audience}' "
        )
    model = ChatOpenAI(
                model="gpt-5-nano",
                temperature=0.7
            )
    parser = StrOutputParser()
        
    chain = prompt | model | parser

    # input = [
    #     {"product":"AI Course","audience":"developers"}
    # ]
    single_input = {"product":"AI Course","audience":"developers"}

    result = chain.invoke(single_input)

    print(f"Response : {result}")
    return chain


def new_first_chain():
    chatmodel = init_chat_model()
    pass

exercise_first_chain()