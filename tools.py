# TO check
# 1. llm model
# 2. internet to get the company history

from dotenv import load_dotenv
load_dotenv()

import os

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

from crewai_tools import SerperDevTool

# Initialize the tool for internet search capabilities
tool = SerperDevTool()

