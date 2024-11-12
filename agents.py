import os
import time
import logging
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set API keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Check if GOOGLE_API_KEY is set
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

# Initialize the Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=GOOGLE_API_KEY)

# Initialize the search tool
search = SerpAPIWrapper(serpapi_api_key=SERPER_API_KEY)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Use this tool to search the web and get company history"
    )
]

# Initialize the conversation memory
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

# Define the prompt template
prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Human: {input}
AI: """

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history"]
)

# Initialize the agent
llm_chain = ZeroShotAgent.from_llm_and_tools(
    llm=gemini_model,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=llm_chain,
    tools=tools,
    memory=memory,
    verbose=True
)

# Implement token bucket for rate limiting
class TokenBucket:
    def __init__(self, tokens, fill_rate):
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.timestamp = time.time()

    def consume(self, tokens):
        now = time.time()
        tokens_to_add = (now - self.timestamp) * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.timestamp = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# Initialize the token bucket (e.g., 10 tokens, refill 1 token per second)
bucket = TokenBucket(10, 1)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_agent_with_retry(input_query: str):
    while not bucket.consume(1):
        time.sleep(0.1)
    return agent_executor.run(input_query)

def run_agent(input_query: str):
    try:
        logger.info(f"Running agent with input: {input_query}")
        response = run_agent_with_retry(input_query)
        logger.info(f"Agent response received")
        return response
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return f"Error: {str(e)}"

# Define agent roles
fraud_classifier = agent_executor # Run the agent
def main():
    input_query = "Your input query here"
    response = run_agent(input_query)
    print(response)

if __name__ == "__main__":
    main()