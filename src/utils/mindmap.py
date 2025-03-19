from sqlalchemy import create_engine, engine, inspect, MetaData, Table, select, distinct, insert, and_, true
import pandas as pd
import re
from src.utils.datastore import driver, server, database, uid, pwd
import datetime

# mindmap_df = pd.read_csv('datas/mind_map_2.csv')

class DB_ENGINE:
    """
    This class will handle all the Database Related Connection objects using SQLAlchemy
    """
    def __init__(self) -> object:
        self.connection_url = engine.URL.create(
                                "mssql+pyodbc",
                                username=uid,
                                password=pwd,
                                host=server.split(":")[1].split(",")[0],
                                port=1433,
                                database=database,
                                query={
                                    "driver": driver
                                },
                            )
        print("Obtaining the Connection Object for Database")
        self.engine_obj = create_engine(self.connection_url)
        print("Successfully obtained the Connection Object for Database")
        self.meta_obj = MetaData()

    def get_table(self, table_name):
        print(f"Getting the Table Object for table: '{table_name}'")
        table_obj = Table(table_name, self.meta_obj, autoload_with=self.engine_obj)
        print(f"Successfully obtained the Table Object for table: '{table_name}'")
        return table_obj

def initialize_connection() -> tuple:
    """
    This function will intialize the DB Connection and metadata table object connection

    Returns:
    engine_obj: Contains the engine SQLAlchemy Object
    mind_map_table: Contains a table object connection
    """
    db = DB_ENGINE()
    engine_obj = db.engine_obj
    mind_map_table = db.get_table('mind_map')
    return engine_obj, mind_map_table


def fetch_the_rows(engine_obj, mind_map_table, sql_stmt) -> list:
    """
    This function will help to fetch all the rows based on the SQL Statement Provided

    Args:
        engine_obj: SQLAlchemy SQL Engine Object which is connected to the Database
        mind_map_table: Metadata Table object
        sql_stmt: SQL Statement object which is helpful to fetch the rows

    Returns:
        rows(list): This is the list of all the fetched rows
    """
    try: 
        with engine_obj.connect() as conn:
            rows = conn.execute(sql_stmt).fetchall()
        print(f'\nFetched {len(rows)} row(s)')
        return rows

    except Exception as err:
        print(f"An error occurred: {err}")

def map_the_variables_in_sub_question(ques,mapping_dict) -> str:
    """
    This function will help to map the <VAR_variable_VAR> keywords which are in the SQL fetched rows with their corresponding values

    Args:
        ques (str): String of sub_question text which is fetched from SQL Fetch and contains the <VAR_variable_VAR> keywords in the text
        mapping_dict (dict): This dictionary contains the <VAR_variable_VAR> used in SQL and their corresponnding values

    Returns:
        ques(str): Final replaced version of the sub_question
    """
    print(f'For Question: \n {ques}')
    for key in mapping_dict.keys():
        print(f'Checking for Variable {key}')
        if key in ques:
            ques = ques.replace(key, mapping_dict[key])
            print(f'After replacing: \n {ques}')
    return ques

def extract_analysis(user_input: str) -> dict:
    """
    Extracts analysis-related entities from the user's input string.

    This function uses predefined mappings to identify and extract the type of analysis
    from the provided user input. It returns a dictionary containing the identified
    analysis type.

    Args:
        user_input (str): The input string from the user.

    Returns:
        dict: A dictionary containing the extracted analysis type with the key 'Analysis Type'.
    """
    analysis_type_mapping = {
        "trend analysis": "trend analysis",
        "decomposition analysis": "decomposition analysis",
        "variance analysis": "variance analysis",
        "trend variance analysis": "Trend_Variance_Analysis",
        "decomposition variance analysis": "Decomposition_Variance Analysis",
        "higher": "root_cause_analysis",
        "summarize": "Summary"
    }


    entities = {}
    # Extract Analysis Type using mapping
    for key in analysis_type_mapping:
        if key in user_input.lower():
            entities["Analysis Type"] = analysis_type_mapping[key]
    return entities

def component_extract(input_string: str) -> tuple:
    """
    Extracts year and period information from the input string, identifies relevant analysis types based on keywords,
    and calculates current, previous, and last-to-last periods.

    Args:
        input_string (str): The input string containing information about the year, period, and analysis components.

    Returns:
        tuple: A tuple containing the current period, previous period, last-to-last period, and identified analysis component.
    """

    # Convert input string to lowercase for easier processing
    input_string = input_string.lower()

    # Extract year using a regex pattern
    year_match = re.search(r'\b(20\d{2})\b', input_string)
    year = year_match.group(1) if year_match else '2023'
 
    # Extract period using a regex pattern for various period formats
    period_match = re.search(r'\b(?:period\s*|p)(\d{1,3})\b', input_string)
    period = period_match.group(1).zfill(3) if period_match else '013'  # Default to period 13 if none found

    # Analysis component detection using keyword matching with regex
    keyword_to_analysis = {
        'GSV': r'\bgross sales value|revenue|sales|gsv\b',
        'Prime Costs': r'\bprime costs?|prime\b',
        'Trade Costs': r'\btrade costs?|trade\b',
        'General And Admin Overhead': r'\bgeneral and admin overhead\b'
    }

    # Identify the analysis component
    head = ''
    for key, pattern in keyword_to_analysis.items():
        if re.search(pattern, input_string):
            head = key
            break

    # Calculate previous and last-to-last periods
    if period == '001':
        year_n = int(year) - 1
        previous_yp = f"{year_n}013"
    else:
        p_period = str(int(period) - 1).zfill(3)
        previous_yp = f"{year}{p_period}"

    previous_year = str(int(year) - 1)
    last_to_last = str(int(year) - 2)

    current_period = f"{year}{period}"
    previous_year_same_period = f"{previous_year}{period}"
    last_to_last_period = f"{last_to_last}{period}"

    return current_period, previous_yp, previous_year_same_period, last_to_last_period, head

def sub_queries(question: str) -> list:
    """
    Generates sub-queries based on the analysis type extracted from the given question.

    This function first extracts the current, previous, and last-to-last periods from the input question using the `year_extract` function.
    It then identifies the analysis type from the question using the `extract_analysis` function and generates specific sub-queries
    tailored to each analysis type. The sub-queries are related to financial analysis metrics such as percentage contribution,
    dollar value contribution, deviations, rolling averages, and percentage changes.

    Args:
        question (str): The input question from the user.

    Returns:
        list: A list containing sub-queries relevant to the identified analysis type.
    """
    engine_obj, mind_map_table = initialize_connection()
    current_period, previous_yp, previous_year_same_period, last_to_last_period, head = component_extract(question)
    VAR_REPLACE = {
        '<VAR_current_period_VAR>': current_period,
        '<VAR_previous_yp_VAR>': previous_yp,
        '<VAR_previous_year_same_period_VAR>': previous_year_same_period,
        '<VAR_last_to_last_period_VAR>': last_to_last_period,
        '<VAR_head_VAR>': head
    }

    analysis_type = extract_analysis(question)['Analysis Type']
    sub_queries = []
    if analysis_type == 'decomposition analysis':
        sql_stmt = select(mind_map_table.c['sub_questions','sub_question_header']).where(mind_map_table.c['analysis_type']==analysis_type)
        rows = fetch_the_rows(engine_obj, mind_map_table, sql_stmt)
        for sub_query in rows:
            query = map_the_variables_in_sub_question(sub_query[0],VAR_REPLACE)
            sub_queries.append((query, sub_query[1]))
        return sub_queries

    elif analysis_type == 'trend analysis':
        sql_stmt = select(mind_map_table.c['sub_questions','sub_question_header']).where(mind_map_table.c['analysis_type']==analysis_type)
        rows = fetch_the_rows(engine_obj, mind_map_table, sql_stmt)
        for sub_query in rows:
            query = map_the_variables_in_sub_question(sub_query[0],VAR_REPLACE)
            sub_queries.append((query, sub_query[1]))
        return sub_queries

    elif analysis_type == 'variance analysis':
        sql_stmt = select(mind_map_table.c['sub_questions', 'sub_question_header']).where(mind_map_table.c['analysis_type']==analysis_type)
        rows = fetch_the_rows(engine_obj, mind_map_table, sql_stmt)
        for sub_query in rows:
            query = map_the_variables_in_sub_question(sub_query[0],VAR_REPLACE)
            sub_queries.append((query, sub_query[1]))
        return sub_queries
    elif analysis_type=="higher":
        sql_stmt = select(mind_map_table.c['sub_questions', 'sub_question_header']).where(mind_map_table.c['analysis_type']==analysis_type)
        rows = fetch_the_rows(engine_obj, mind_map_table, sql_stmt)
        for sub_query in rows:
            query = map_the_variables_in_sub_question(sub_query[0],VAR_REPLACE)
            sub_queries.append((query, sub_query[1]))
        return sub_queries