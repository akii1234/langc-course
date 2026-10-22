from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ChatMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model

load_dotenv()

# 1. Correctly instantiate your message history array
messages_history = [
    HumanMessage(content="Hello !"),
    AIMessage(content="Hi There! How can I assist you today?"),
    SystemMessage(content="This is a system message."),
    ToolMessage(content="Tool executed successfully.", tool_call_id="call_123abc"), # Fixed
    ChatMessage(content="This is a general chat message.", role="assistant")      # Fixed
]

# 2. Set up few-shot examples
examples = [
    {'input': 'happy', 'output': 'sad'},
    {'input': 'tall', 'output': 'short'}
]

example_prompt = ChatPromptTemplate.from_messages([
    ('human', '{input}'),
    ('ai', '{output}')
])

fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples   
)

# 3. Use MessagesPlaceholder to inject the history array
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Give the opposite of each word."),
        # This acts as a slot where your message array will be inserted safely
        MessagesPlaceholder(variable_name="chat_history"), 
        fewshot_prompt,
        ("human", '{input}'),
    ]
)

# 4. Initialize a valid model (changed from gpt-5-nano to gpt-4o-mini)
model = init_chat_model(model="ollama:llama3.2", temperature=0)

# 5. Format the prompt by passing BOTH the input word and the history array
formatted_messages = final_prompt.format_messages(
    chat_history=messages_history, 
    input="hot"
)

response = model.invoke(formatted_messages)
print(response.content)