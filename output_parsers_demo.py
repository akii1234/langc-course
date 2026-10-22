from langchain_core.prompts import(
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

#PydanticOutputParser example
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

def working_with_parsers():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates '{input_language}' to '{output_language}' .",
            ),
            (
            "human",
            "Translate the following text: '{text}' "
            )
        ]
    )

    parser = StrOutputParser()

    llm = init_chat_model(model="gpt-5-nano",temperature=0)

    chain = prompt | llm | parser

    response = chain.invoke({"input_language":"English","output_language":"French",'text':'My Name is Khan'})

    print(type(response))




# working_with_parsers()

class Person(BaseModel):
    name: str = Field(description="The Person's name")
    age: int = Field(description="The Preson's age")
    occupation: str = Field(description="The person's occupation")

# llm = init_chat_model(model="ollama:llama3.2",temperature=0)
llm = init_chat_model(model="openai:gpt-5-nano",temperature=0)

def working_with_pydantic_parser():
    parser = PydanticOutputParser(pydantic_object=Person)

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name','age', and 'occupation' for: {description}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    response = chain.invoke({"description":"A 30 year old artist Maria"})
    print(response)


# structured Output
class MovieReview(BaseModel):
    title:str = Field(description="Title of the movie")
    review:str = Field(description="Review of the movie")
    rating:str = Field(description="Rating of the movie")

def movie_review():
    structured_model = llm.with_structured_output(MovieReview)    
    result = structured_model.invoke("Review: Inception is a mind-boggling thriller.9/10")
    print(result)
    return result

if __name__ == '__main__':
    user_input = input("Enter the number either 1 or 2 : ")
    if int(user_input) == 1:
        working_with_parsers()

    elif int(user_input) == 2:        
        working_with_pydantic_parser()
    movie_review()    