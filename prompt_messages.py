from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,ChatMessage,ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

# prompt template single - message

# prompt = ChatPromptTemplate.from_template(
#     "Tell me a {adjective} joke about {topic} "
# )


# messages = prompt.format_messages(adjective="funny",topic='chickens')


# multi-message templates

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}.",
#         ),
#         (
#          "human",
#          "Translate the following text: {text}"
#          )
#     ]
# )


# messages = prompt.format_messages(
#     input_language = "English", output_language = "French", text="I love programming"
# )

# model = init_chat_model(model='gpt-4o-mini',temperature=0)
# model = init_chat_model(model='ollama:llama3.2',temperature=0)
# response = model.invoke(messages)
# print(response.content)

# outpt parser : Message Types : 
# messages = [
#     HumanMessage(content="Hello !"),
#     AIMessage(content="Hi There! How can I assist you today?"),
#     SystemMessage(content="This is a system message."),
#     ToolMessage(content="Tool executed successfully."),
#     ChatMessage(content="This is a general chat message.")
# ]

examples = [
    {'input':'happy','output':'sad'},
    {'input':'tall','output':'short'}
]

example_prompt = ChatPromptTemplate.from_messages([
    ('human','{input}'),
    ('ai','{output}')
])

fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples   
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Give the opposite of each word."),
        fewshot_prompt,
        ("human",'{input}'),
     
    ]
)

model = init_chat_model(model="gpt-5-nano",temperature=0)
response = model.invoke(final_prompt.format_messages(input="happy"))

print(response.content)