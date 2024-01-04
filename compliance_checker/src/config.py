"""
This file contains all the config related information
"""
# Reference site for compliance
COMPLIANCE_SITE = "https://stripe.com/docs/treasury/marketing-treasury"
# Path to store text embeddings
STORAGE_PATH = "./cachce/"
# Add openai_api key
OPENAI_API_KEY = None
# Try modifying the prompt for better response
PROMPT_TEMPLATE = """ 
{context}
\n\n
Given above context on compliance, return a list of the sentences and words with  reasoning from the below
{question} that violates the above compliance
"""
