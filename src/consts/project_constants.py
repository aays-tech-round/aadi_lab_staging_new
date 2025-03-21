from dotenv import load_dotenv
import os

load_dotenv()


class SQL_CONFIG:
    """
    Configuration class for SQL database connection settings.

    Attributes:
        DRIVER (str): Database driver.
        SERVER (str): SQL server address.
        DATABASE (str): Name of the database.
        CLIENT_ID (str): Client ID for authentication.
        CLIENT_SECRET (str): Client secret for authentication.
        TENANT_ID (str): Tenant ID for authentication.
        connection_string (str): Formatted database connection string.
    """
    
    DRIVER = 'ODBC Driver 17 for SQL Server'
    SERVER = os.getenv('SQL_SERVER')
    DATABASE = os.getenv('SQL_DATABASE')
    USER_ID = os.getenv('SQL_USER_ID')
    PASSWORD = os.getenv('SQL_PASSWORD')
    CONNECTION_STRING=f'Driver={DRIVER};Server={SERVER};Database={DATABASE};Uid={USER_ID};Pwd={PASSWORD};Encrypt=yes;TrustServerCertificate=no;'
 
class OPENAI_CONFIG:  # confidential need to move them to either pythontoml or dot env? 
    """
    Configuration class for OpenAI API settings.

    Attributes:
        OPENAI_LLM_AZURE_ENDPOINT (str): Azure OpenAI endpoint.
        OPENAI_LLM_DEPLOYMENT_NAME (str): OpenAI model deployment name.
        OPENAI_LLM_API_KEY (str): API key for OpenAI.
        OPENAI_LLM_API_VERSION (str): API version for OpenAI.
        OPENAI_EMBED_AZURE_ENDPOINT (str): Azure OpenAI embedding endpoint.
        OPENAI_EMBED_API_KEY (str): API key for embedding model.
        OPENAI_EMBED_MODEL_API_VERSION (str): API version for embedding model.
    """ 
    OPENAI_LLM_AZURE_ENDPOINT = os.getenv('OPENAI_LLM_AZURE_ENDPOINT')
    OPENAI_LLM_DEPLOYMENT_NAME = os.getenv('OPENAI_LLM_DEPLOYMENT_NAME')
    OPENAI_LLM_API_KEY = os.getenv('OPENAI_LLM_API_KEY')
    OPENAI_LLM_API_VERSION = '2024-02-01'
    OPENAI_EMBED_AZURE_ENDPOINT = os.getenv('OPENAI_EMBED_AZURE_ENDPOINT')
    OPENAI_EMBED_API_KEY = os.getenv('OPENAI_EMBED_API_KEY')
    OPENAI_EMBED_MODEL_API_VERSION = '2023-09-15-preview'

class GEN_SQL:
    """
    A class representing the structured response of an AI-generated SQL query 
    along with its analysis and visualization.

    Attributes:
        sql (str): The generated SQL query.
        formatted_code (str): Formatted version of the SQL code.
        df (DataFrame): DataFrame containing the query results.
        summary (str): Summary of the query result.
        fig (object): Visualization figure (e.g., a Plotly or Matplotlib figure).
        status (str): Execution status of the query.
        reply (str): AI-generated reply explaining the query.
        exception (str or None): Exception message if an error occurs.
        fig_python (str): Python code used to generate the visualization.
        header (str): Header for the result set.
        con_score (float): Confidence score of the generated query.
        query (str): Original user-provided query.
        plot_suggestion (str): Suggested plot type for visualization.
        viz_extracted_scores (dict): Scores extracted for visualization elements.
        sql_scores (dict): Scores related to SQL query correctness.
    """
    def __init__(self, sql, formatted_code, df,summary,fig, status, reply, exception, 
                 fig_python, header, con_score, query, 
                 plot_suggestion, viz_extracted_scores, sql_scores):
        
        """
        Initializes the GenSql object.

        Args:
            sql (str): The generated SQL query.
            formatted_code (str): Formatted SQL query.
            df (DataFrame): Query result as a DataFrame.
            summary (str): Summary of the query result.
            fig (object): Visualization figure.
            status (str): Execution status.
            reply (str): AI-generated reply explaining the query.
            exception (str or None): Exception message if any error occurs.
            fig_python (str): Python code for visualization.
            header (str): Header for the result.
            con_score (float): Confidence score of the query.
            query (str): User's original query.
            plot_suggestion (str): Suggested visualization type.
            viz_extracted_scores (dict): Scores for visualization aspects.
            sql_scores (dict): Scores for SQL correctness.
        """
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


           
