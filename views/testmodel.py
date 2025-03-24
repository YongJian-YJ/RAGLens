from langchain_community.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOllama

# Load environment variables
load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Initialize AzureChatOpenAI
# model = AzureChatOpenAI(
#     openai_api_base=OPENAI_API_BASE,
#     openai_api_key=OPENAI_API_KEY,
#     openai_api_version=OPENAI_API_VERSION,
#     model_name=DEPLOYMENT_NAME,
#     temperature=0.0,
# )
llm = ChatOllama(
    model="command-r-plus:latest", temperature=0.0, base_url="http://localhost:8888"
)
# Test the model
response = llm.invoke("Hello, world!")
print(response.content)
