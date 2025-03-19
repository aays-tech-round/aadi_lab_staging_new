# import sqlite3
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from openai import AzureOpenAI
from src.openai.openai import OpenAI_Chat
from src.chroma.chroma import ChromaDB_VectorStore 
import pandas as pd
from src.consts.project_constants import OPENAI_CONFIG,SQL_CONFIG
# from training_material import ddls, aggregated_doc, hierarchy_doc, relation_doc, new_query, naming_conventions, yearperiod_doc


class MyAadi(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=AzureOpenAI(
                              azure_endpoint = OPENAI_CONFIG.OPENAI_LLM_AZURE_ENDPOINT,
                              api_key=OPENAI_CONFIG.OPENAI_LLM_API_KEY,
                              api_version=OPENAI_CONFIG.OPENAI_LLM_API_VERSION), config=config) # Make sure to put your AzureOpenAI client here
path = r'dbs'
vn = MyAadi(config={'model': OPENAI_CONFIG.OPENAI_LLM_DEPLOYMENT_NAME,'temperature':0.0,'path':path})


vn.connect_to_mssql(SQL_CONFIG.CONNECTION_STRING)
# Training
# df_information_schema = vn.run_sql('SELECT * FROM INFORMATION_SCHEMA.COLUMNS')
# plan = vn.get_training_plan_generic(df_information_schema)
#
# ddls = ddls
# doc_list = [aggregated_doc, hierarchy_doc, relation_doc, naming_conventions, yearperiod_doc]
# training_df = new_query

# def train_vn(ddls, query_df, doc_list):
#     for ddl in ddls:
#         vn.train(ddl=ddl)
#     for i in doc_list:
#         vn.train(documentation=i)
#     for i, j in query_df.iterrows():
#         vn.train(question = j[0], sql = j[1])

# train_vn(ddls,new_query,doc_list)

def vn_aays():
    return vn

