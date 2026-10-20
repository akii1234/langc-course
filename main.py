from dotenv import load_dotenv
from langchain_core import __version__ as core_version
from langchain import __version__ as lg_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from importlib.metadata import version

load_dotenv()

# core_version = version("lanchain_core")
# lg_version = version("langgraph")

print(f"lanchain core :: {core_version}")
print(f"langgraph version :: {lg_version}")

def main():
    # llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0)
    # response = llm.invoke("Say,Set up complete! IN one word")
    # print(f"Response from OpenAI : {response}")

    # Test - Anthropic
    llm_anthropic = ChatAnthropic(model_name="claude-sonnet-4-5-20250929",temperature=0)
    response_anthropic = llm_anthropic.invoke("Say set up complete in one word")
    print(f"Response from Anthropic : {response_anthropic}")


if __name__ == "__main__":
    main()
