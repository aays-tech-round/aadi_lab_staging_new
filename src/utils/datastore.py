import pyodbc
from datetime import datetime,timezone,timedelta
import json
from flask import Flask, request, jsonify
from src.consts.project_constants import SQL_CONFIG

# Instantiationg SQL Constant Object
SQLConstObj = SQL_CONFIG()

# Intializing Database Credentials
driver = SQLConstObj.DRIVER
server = SQLConstObj.SERVER
database = SQLConstObj.DATABASE
uid = SQLConstObj.USER_ID
pwd = SQLConstObj.PASSWORD


def get_connection():
    """This function creates a connection object to MS SQL Database 

    Returns:
        Database connection object
    """
    return pyodbc.connect(f'Driver={driver};Server={server};Database={database};Uid={uid};Pwd={pwd};')


def store_response(SQLQueryObj, response_id, Section_id, summary, fig, formatted_code,df, exception, python_code_figure,prompt, query,header, score,all_score, status, reply, summ_of_summ = None, version = 1, system_user_id=None, cached_from_response_id = None):
    """This function will help to create a new record in a database table 

    Args:
        SQLQueryObj (_type_): This contains an instantiated Object of SQL_QUERY which contains various methods of different sql queries
        response_id (_type_): This contains a response id also known as message id
        Section_id (_type_): This contains a section id 
        summary (_type_): This contains an LLM generated summary
        fig (_type_): This contains the JSON code which helps the UI side to create the components
        formatted_code (_type_): This contains a programming code which is properly formatted.
        df (_type_): This contains a pandas dataframe relevant to user query
        exception (_type_): This contains the exception text if existed any
        python_code_figure (_type_): This contains the python code which is written using relevant visualization library
        prompt (_type_): This contains an engineered prompt which is used to guide an LLM response
        query (_type_): This contains the user query
        version (int, optional): This contains the version. Defaults to 1.
        system_user_id (_type_, optional): This contains the system user id, i.e the user which queries something about the data. Defaults to None.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        #generating user_id
        system_user_id = 'DIP ROY'

        gmt_offset_hours = 0  # Set this to your desired GMT offset, e.g., +5 for GMT+5
        gmt_offset = timezone(timedelta(hours=gmt_offset_hours))

        creation_timestamp = datetime.now(gmt_offset)
        update_timestamp = creation_timestamp

        
        # df = json.dumps(df)
        # fig = json.dumps(fig)

        cursor.execute(SQLQueryObj.sql_insert_response(), 
                       (response_id, Section_id, summary, fig, formatted_code, df, exception, version, creation_timestamp, update_timestamp, system_user_id, str(python_code_figure), prompt, query, header, score, all_score, summ_of_summ, status, reply, cached_from_response_id))
        conn.commit()

    except Exception as e:
        pass
    
    finally:
        cursor.close()
        conn.close()

def re_draw_data_store(SQLQueryObj, Response_id, Section_id, python_code_figure, figure, version_history):
    """This function helps to get the context of the data and previous code in case of redrawing a particular chart/visual

    Args:
        SQLQueryObj (_type_): This contains an instantiated Object of SQL_QUERY which contains various methods of different sql queries
        Response_id (_type_): This contains a response id also known as message id
        Section_id (_type_): This contains a section id 
        python_code_figure (_type_): This contains the python code which is written using relevant visualization library
        figure (_type_): This contains the JSON code which helps the UI side to create the components
        version_history (_type_): This contains the version history

    Returns:
        _type_: This returns a json object which contains the code related the chart which needs to be redrawn
    """

    if not Response_id:
        return jsonify({'error': 'response_id is required'}), 400

    try: 
        conn = get_connection()
        cursor = conn.cursor()
        
        query = SQLQueryObj.sql_query_get_re_draw_max_version()
        cursor.execute(query, (Response_id,Section_id,Response_id,Section_id))
        row = cursor.fetchone()

        gmt_offset_hours = 0  # Set this to your desired GMT offset, e.g., +5 for GMT+5
        gmt_offset = timezone(timedelta(hours=gmt_offset_hours))

        # Get the current time in GMT+0 (UTC)
        update_timestamp = datetime.now(gmt_offset)

        

        if row:

            response_id = row[0]
            Section_id = row[1]
            summary = row[2]
            fig = json.dumps(figure)
            formatted_code = row[4]
            df = row[5]
            exception = row[6]
            version = row[7]
            creation_timestamp = row[8].strftime('%Y-%m-%d %H:%M:%S')
            update_timestamp = update_timestamp
            system_user_id = row[10]
            python_code_figure = python_code_figure
            User_query = row[12]
            Child_query = row[13]
            header = row[14]
            score = row[15]
            all_score = row[16]
            summ_of_summ = row[17]
            status = row[18]
            reply = row[19]
            cached_from_response_id = row[20]

            new_version = int(version) + 1

        
            cursor.execute(SQLQueryObj.sql_insert_re_draw_new_version(), 
                        (response_id, Section_id, summary, fig, formatted_code, df, exception, new_version, creation_timestamp, update_timestamp, system_user_id, python_code_figure, User_query, Child_query, header, score, all_score, summ_of_summ, status, reply, cached_from_response_id))
            
            cursor.execute(SQLQueryObj.sql_update_aadi_history(), 
                        (fig, python_code_figure,update_timestamp, version_history, str(Section_id), response_id)
                        )
            
            conn.commit()
            
        cursor.close()
        conn.close()
            
    except Exception as e:
            return jsonify({'error':e})
    

def reset(SQLQueryObj, response_id,child_id):
    """This function helps to reset the session context and go back to the all the codes and values which were there on the first execution of the chat

    Args:
        SQLQueryObj (_type_): This contains an instantiated Object of SQL_QUERY which contains various methods of different sql queries
        response_id (_type_): This contains a response id also known as message id
        child_id (_type_): This contains the child id.

    Returns:
        _type_: Returns a JSON object which contains all the relevant values which takes back to the initial state.
    """
    try:
        connection = get_connection()
        with connection:
            cursor = connection.cursor()
            cursor.execute(SQLQueryObj.sql_query_reset_to_start(),
                           (response_id,child_id)
            )
            row = cursor.fetchone()
            if row:
                fig = json.loads(row[3])
                python_code_figure = row[11]
                return jsonify({'figure': fig, 'python_code_figure':python_code_figure})
            else:
                return jsonify({'error': 'No data found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def undo(SQLQueryObj, response_id,child_id,version):
    """This function help to go bac to the previous state in the current chat/conversation context

    Args:
        SQLQueryObj (_type_): This contains an instantiated Object of SQL_QUERY which contains various methods of different sql queries
        response_id (_type_): This contains a response id also known as message id
        child_id (_type_): This contains the child id.
        version (_type_): This contains the current version of the conversation state.

    Returns:
        _type_: Returns a JSON object which contains all the relevant values which takes back to the previous state.
    """
    try:
        if int(version) > 1:
            undo_version = int(version) - 1
            connection = get_connection()
            with connection:
                cursor = connection.cursor()
                cursor.execute(SQLQueryObj.sql_query_undo(), (response_id,child_id,str(undo_version))
                )
                row = cursor.fetchone()
                if row:
                    fig = json.loads(row[3])
                    python_code_figure = row[11]
                    return jsonify({'figure': fig, 'python_code_figure':python_code_figure})
                else:
                    return jsonify({'error': 'No data found'}), 404
        else:
            pass
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        

def get_data_followup(SQLQueryObj, response_id, section_id):
    """This function helps to get the relevant context of the data so that the follow up question can be asked within the data context

    Args:
        SQLQueryObj (_type_): This contains an instantiated Object of SQL_QUERY which contains various methods of different sql queries
        response_id (_type_): This contains a response id also known as message id
        section_id (_type_): This contains a section id

    Returns:
        _type_: This returns all the relevant values which are needed to understand the data context.
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = SQLQueryObj.sql_query_for_followup()

    cursor.execute(query, (response_id, section_id,response_id, section_id))
    row = cursor.fetchone()
    
    summary = row[2]
    sql = row[4]
    df = row[5]
    main_question = row[12] 
    child_question = row[13] 
    header = row[14]

    return summary, sql, df, main_question, child_question, header

def update_table(SQLQueryObj,old_response_id: str, new_response_id: str):
    
    try:
        connection = get_connection()
        cursor = connection.cursor()

        query=SQLQueryObj.sql_query_for_update_table()
        
        cursor.execute(query, (new_response_id, old_response_id))

        connection.commit()
        
    except Exception as e:
        pass
    
    finally:
        cursor.close()
        connection.close()


def get_the_cached_response(SQLQueryObj,prompt):
    
    try:
        connection = get_connection()
        cursor = connection.cursor()

        query=SQLQueryObj.sql_query_get_the_cached_response()
        cursor.execute(query,(prompt.lower()))
        row = cursor.fetchall()
        return row
        
    except Exception as e:
        pass
    
    finally:
        cursor.close()
        connection.close()