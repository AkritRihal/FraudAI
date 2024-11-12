# 1. fraud check
# 2. scorer
# 3. summarize


from crewai import Agent, Task

# Create custom agents
fraud_classifier = Agent(
    role='Fraud Classifier',
    goal='Classify messages as fraud or not fraud',
    backstory='You are an expert in detecting fraudulent messages',
    allow_delegation=False
)

reliability_score = Agent(
    role='Reliability Scorer',
    goal='Score the reliability of messages from 1 to 10',
    backstory='You are an expert in assessing the reliability of information',
    allow_delegation=False
)

summarizer = Agent(
    role='Summarizer',
    goal='Summarize messages in simple language',
    backstory='You are an expert in condensing information into clear summaries',
    allow_delegation=False
)

# Fraud check task
fraud_task = Task(
    description=(
        "Analyse the {topic} and classify it into fraud message or not fraud. Focus on identifying whether the message is useful or is just a promotional or spam. "
        "Your final answer must take into consideration the context, sarcasm and overall meaning of the message, to ensure high accuracy in classification process."
    ),
    expected_output="An output describing if it is a fraud message or not",
    agent=fraud_classifier
)

# Scorer task
score_task = Task(
    description=(
        "Analyse the {topic} and see how much reliable is the message. Focus on identifying whether the message is useful and can be trusted or is just a promotional or spam. "
        "Your final answer should be a digit from 1 to 10 where 1 signifies lowest reliability and 10 signifies full reliability. "
        "Do not hallucinate or reuse same input."
    ),
    expected_output="A number signifying the reliability of the message",
    agent=reliability_score
)

# Summarizer task
summary_task = Task(
    description=(
        "Analyse the {topic} and summarize it in a simple and easy to understand language. Focus on identifying whether the message is useful or is just a promotional or spam. "
        "Do not halluncinate or reuse same input. "
        "Your final answer should be a summary of the message in 2-3 sentences."
    ),
    expected_output="A summary of the message in 2-3 sentences",
    agent=summarizer
)