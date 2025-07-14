import os

os.environ["ANTHROPIC_API_KEY"] = ""

from biomni.agent import A1
from biomni.tool.systems_biology import chatnt_call

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path="./data", llm="claude-sonnet-4-20250514")

# Execute biomedical tasks using natural language
agent.go("Please analyze the function of AACCTTGG based on ChatNT.")

from biomni.llm import get_llm
from biomni.utils import function_to_api_schema

# llm = get_llm('claude-sonnet-4-20250514')
# desc = function_to_api_schema(chatnt_call, llm)
# print(desc)

# print(chatnt_call("Please tell me the details of this sequence","AATTCC"))
