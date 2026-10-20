from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

class DemoChain():
    def __init__(self):
        pass

    def demo_basic_chain():
        """Demonstrate a basic chat template"""    
