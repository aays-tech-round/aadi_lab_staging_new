# import sqlite3
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from openai import AzureOpenAI
from src.openai.openai import OpenAI_Chat
from src.chroma.chroma import ChromaDB_VectorStore 
import pandas as pd
from src.consts.project_constants import OPENAI_CONFIG,SQL_CONFIG


class MyAadi(ChromaDB_VectorStore, OpenAI_Chat):
    """
        A subclass of OpenAIChat that initializes an Azure OpenAI client 
        with predefined configurations for LabGenAI usage.
    """
    def __init__(self, config=None):
        """
            Initializes the LabGenAI class with an Azure OpenAI client.

            Args:
                config (dict, optional): Configuration dictionary for the model.
                    Expected keys:
                    - 'model': Name of the deployment model.
                    - 'temperature': The temperature setting for the model.
            
            Example:
                lga = LabGenAI(config={'model': 'gpt-4', 'temperature': 0.0})
        """
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=AzureOpenAI(
                              azure_endpoint = OPENAI_CONFIG.OPENAI_LLM_AZURE_ENDPOINT,
                              api_key=OPENAI_CONFIG.OPENAI_LLM_API_KEY,
                              api_version=OPENAI_CONFIG.OPENAI_LLM_API_VERSION), config=config) # Make sure to put your AzureOpenAI client here
path = r'dbs'
ab = MyAadi(config={'model': OPENAI_CONFIG.OPENAI_LLM_DEPLOYMENT_NAME,'temperature':0.0,'path':path})


ab.connect_to_mssql(SQL_CONFIG.CONNECTION_STRING)

def aadi_genai():
    """
        Returns the initialized LabGenAI instance.

        Returns:
            LabGenAI: An instance of the LabGenAI class.
        
        Example:
            genai_instance = lab_genai()
    """
    return ab