from dotenv import load_dotenv
import os

load_dotenv()


class SQL_CONFIG:
    DRIVER = 'ODBC Driver 17 for SQL Server'
    SERVER = os.getenv('SQL_SERVER')
    DATABASE = os.getenv('SQL_DATABASE')
    USER_ID = os.getenv('SQL_USER_ID')
    PASSWORD = os.getenv('SQL_PASSWORD')
    CONNECTION_STRING=f'Driver={DRIVER};Server={SERVER};Database={DATABASE};Uid={USER_ID};Pwd={PASSWORD};Encrypt=yes;TrustServerCertificate=no;'
 
class OPENAI_CONFIG:  # confidential need to move them to either pythontoml or dot env? 
    OPENAI_LLM_AZURE_ENDPOINT = os.getenv('OPENAI_LLM_AZURE_ENDPOINT')
    OPENAI_LLM_DEPLOYMENT_NAME = os.getenv('OPENAI_LLM_DEPLOYMENT_NAME')
    OPENAI_LLM_API_KEY = os.getenv('OPENAI_LLM_API_KEY')
    OPENAI_LLM_API_VERSION = '2024-02-01'
    OPENAI_EMBED_AZURE_ENDPOINT = os.getenv('OPENAI_EMBED_AZURE_ENDPOINT')
    OPENAI_EMBED_API_KEY = os.getenv('OPENAI_EMBED_API_KEY')
    OPENAI_EMBED_MODEL_API_VERSION = '2023-09-15-preview'

class GEN_SQL:
    def __init__(self, sql, formatted_code, df,summary,fig, status, reply, exception, fig_python, header, con_score, query, plot_suggestion, viz_extracted_scores, sql_scores):
        self.sql = sql
        self.formatted_code = formatted_code
        self.df = df
        self.summary = summary
        self.fig = fig
        self.status = status
        self.reply = reply
        self.exception = exception
        self.fig_python = fig_python
        self.header = header
        self.con_score = con_score
        self.plot_suggestion = plot_suggestion
        self.query = query
        self.viz_extracted_scores = viz_extracted_scores
        self.sql_scores = sql_scores  


           
