from pydantic import BaseModel, ValidationError
from typing import Optional, Literal, Union
from langchain_openai import AzureChatOpenAI
import os
from src.consts.project_prompts import Prompts
from src.consts.project_constants import OPENAI_CONFIG,SQL_CONFIG

OpenAIConfigObj = OPENAI_CONFIG()

llm = AzureChatOpenAI(
        model = OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME,
        azure_deployment = OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME,
        azure_endpoint = OpenAIConfigObj.OPENAI_LLM_AZURE_ENDPOINT,
        api_key=OpenAIConfigObj.OPENAI_LLM_API_KEY,
        api_version=OpenAIConfigObj.OPENAI_LLM_API_VERSION
    )

prompts = Prompts()

class TaskResponse(BaseModel):
    task: Literal["aadi_nlp_to_sql_query",                   
                  "prohibited_nlp_to_sql",
                  "greeting", 
                  "generic_question"]
    reason: str
    rephrased_query: Union[str, bool]
    original_query: str

def intent_llm(query: str) -> TaskResponse:
    """
    Determines the intent of a given user query using an LLM-based classification system.

    Args:
        query (str): The user-provided query to analyze.

    Returns:
        TaskResponse: A structured response object containing the detected task type.

    Raises:
        ValueError: If there is a validation error in the TaskResponse.
        RuntimeError: If any other error occurs during task determination.
    """
    try:
        prompt = prompts.prompt_intent_llm_agent(query=query)
        structured_llm = llm.with_structured_output(TaskResponse, method="json_mode")
        task_response = structured_llm.invoke(prompt)
        return task_response
    except ValidationError as ve:
        raise ValueError(f"Validation error in TaskResponse: {ve}")
    except Exception as e:
        raise RuntimeError(f"Error during task determination: {str(e)}")
    
def intent_reply(intent: TaskResponse) -> str:
    """
    Generates a response message based on the classified intent of a user query.

    Args:
        intent (TaskResponse): The structured task response obtained from intent_llm().

    Returns:
        str: A predefined response message corresponding to the detected intent.

    Possible Responses:
        - "aadi_nlp_to_sql_question": Returns an empty response.
        - "greeting": Returns a friendly greeting message.
        - "prohibited_nlp_to_sql": Explains restrictions on generating modifying or deleting SQL queries.
        - "generic_question": Provides a general response about Mars Lab data insights.
    """
    if intent.task=="aadi_nlp_to_sql_question":
        reply=""
        return reply    
    elif intent.task=="greeting":
        reply=f"""Hello! I hope you're having a great day. I'm here to provide insights based on Mars Lab data. Let me know how I can assist you!"""
        return reply 
    elif intent.task=="prohibited_nlp_to_sql":
        reply=f"""I'm sorry, but I am unable to generate SQL queries that modify or delete data due to security, ethical, and technical considerations. Ensuring data integrity and compliance is essential. However, if you need help understanding your data structure or forming safe queries, I'd be happy to assist you in a responsible manner!"""
        return reply
    elif intent.task=="generic_question":
        reply=f"""Thank you for your question! I specialize in providing insights based on Mars Lab data, including OTD, LVP, lab re-testing, and regional lab activity. Let me know how I can assist you within these areas!"""
        return reply
    
