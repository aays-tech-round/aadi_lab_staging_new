from src.consts.project_sql import SQLQUERY
from dotenv import load_dotenv
import os
load_dotenv()

def test_sql_query_response_user_data():
    response_user_table = os.environ['RESPONSE_USER_DATA_TABLE']
    expected_output = f"SELECT * FROM {response_user_table}"
    output = SQLQUERY().sql_query_response_user_data()
    assert expected_output == output


def test_sql_insert_response():
    response_user_table = os.environ['RESPONSE_USER_DATA_TABLE']
    expected_output = f"INSERT INTO {response_user_table} (Response_id, Section_id, Summary, Visual, SQL, DataFrame, Response_Code, Version, Creation_Timestamp, Update_Timestamp, userId, python_code_figure, User_query, Child_query) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    output = SQLQUERY().sql_insert_response()
    assert expected_output == output

def test_sql_query_get_re_draw_max_version():
    response_user_table = os.environ['RESPONSE_USER_DATA_TABLE']
    expected_output = f"""
            SELECT *
                FROM [dbo].[{response_user_table}]
                WHERE Response_id = ?
                AND Section_id = ?
                AND Version = (
                    SELECT MAX(Version)
                    FROM [dbo].[{response_user_table}]
                    WHERE Response_id = ?
                    AND Section_id = ?
                );

        """
    output = SQLQUERY().sql_query_get_re_draw_max_version()
    assert expected_output == output

def test_sql_insert_re_draw_new_version():
    response_user_table = os.environ['RESPONSE_USER_DATA_TABLE']
    expected_output = f"INSERT INTO {response_user_table} (Response_id, Section_id, Summary, Visual, SQL, DataFrame, Response_Code, Version, Creation_Timestamp, Update_Timestamp, userId, python_code_figure, User_query, Child_query) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    output = SQLQUERY().sql_insert_re_draw_new_version()
    assert expected_output == output

def test_sql_update_aadi_history():
    expected_output ="""UPDATE aadi_history
                  SET figure = ?, python_figure = ?, createdDate = ?, Version = ?
                  WHERE child_response_id = ? AND messageID = ?"""
    output = SQLQUERY().sql_update_aadi_history()
    assert expected_output == output

def test_sql_query_reset_to_start():
    response_user_table = os.environ['RESPONSE_USER_DATA_TABLE']
    expected_output = f"""
                SELECT * FROM {response_user_table} WHERE Response_id = ? AND Section_id = ? AND Version = '1'
                """
    output = SQLQUERY().sql_query_reset_to_start()
    assert expected_output == output

def test_sql_query_undo():
    response_user_table = os.environ['RESPONSE_USER_DATA_TABLE']
    expected_output = f"""
                SELECT * FROM {response_user_table} WHERE Response_id = ? AND Section_id = ? AND Version = ?
                """
    output = SQLQUERY().sql_query_undo()
    assert expected_output == output

def test_sql_query_for_followup():
        response_user_table = os.environ['RESPONSE_USER_DATA_TABLE']
        expected_output = f"""
                        SELECT *
                            FROM [dbo].[{response_user_table}]
                            WHERE Response_id = ?
                            AND Section_id = ?
                            AND Version = (
                                SELECT MAX(Version)
                                FROM [dbo].[{response_user_table}]
                                WHERE Response_id = ?
                                AND Section_id = ?
                            );
                    """
        
        output = SQLQUERY().sql_query_for_followup()
        assert expected_output == output

def test_sql_query_get_the_cached_response():
        response_user_table = os.environ['RESPONSE_USER_DATA_TABLE']
        expected_output = f"""
            SELECT TOP 1 *
            FROM [dbo].[{response_user_table}]
            WHERE LOWER(Child_query) = ?;
        """
        
        output = SQLQUERY().sql_query_get_the_cached_response()
        assert expected_output == output