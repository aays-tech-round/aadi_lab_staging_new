
import os


class SQLQUERY:
    def __init__(self) -> None:
         self.response_user_table=os.getenv('RESPONSE_USER_DATA_TABLE')
    def sql_query_response_user_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        query = f"SELECT * FROM {self.response_user_table}"
        return query
    
    def sql_insert_response(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        query = f"INSERT INTO {self.response_user_table} (Response_id, Section_id, Summary, Visual, SQL, DataFrame, Response_Code, Version, Creation_Timestamp, Update_Timestamp, userId, python_code_figure, User_query, Child_query, header, score, all_score, sum_of_sum, status_msg, reply, cached_from) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

        return query
    
    def sql_query_get_re_draw_max_version(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        query = f"""
            SELECT *
                FROM [dbo].[{self.response_user_table}]
                WHERE Response_id = ?
                AND Section_id = ?
                AND Version = (
                    SELECT MAX(Version)
                    FROM [dbo].[{self.response_user_table}]
                    WHERE Response_id = ?
                    AND Section_id = ?
                );

        """
        return query
    
    def sql_insert_re_draw_new_version(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        query = f"INSERT INTO {self.response_user_table} (Response_id, Section_id, Summary, Visual, SQL, DataFrame, Response_Code, Version, Creation_Timestamp, Update_Timestamp, userId, python_code_figure, User_query, Child_query, header, score, all_score, sum_of_sum, status_msg, reply, cached_from) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        return query
    
    def sql_update_aadi_history(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        query =f"""UPDATE aadi_history
                  SET figure = ?, python_figure = ?, createdDate = ?, Version = ?
                  WHERE child_response_id = ? AND messageID = ?"""
    
        return query
    
    def sql_query_reset_to_start(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        query = f"""
                SELECT * FROM {self.response_user_table} WHERE Response_id = ? AND Section_id = ? AND Version = '1'
                """
        return query
    
    def sql_query_undo(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        query = f"""
                SELECT * FROM {self.response_user_table} WHERE Response_id = ? AND Section_id = ? AND Version = ?
                """
        return query
    
    def sql_query_for_followup(self):
            """_summary_

        Returns:
            _type_: _description_
        """
            query = f"""
                        SELECT *
                            FROM [dbo].[{self.response_user_table}]
                            WHERE Response_id = ?
                            AND Section_id = ?
                            AND Version = (
                                SELECT MAX(Version)
                                FROM [dbo].[{self.response_user_table}]
                                WHERE Response_id = ?
                                AND Section_id = ?
                            );
                    """
            
            return query
    
    def sql_query_for_update_table(self):
          """_summary_

        Returns:
            _type_: _description_
        """
          query = f"""UPDATE {self.response_user_table}
                    SET Response_id = ?
                     WHERE Response_id = ?;"""
          return query
    
    def sql_query_get_the_cached_response(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        query = f"""
               SELECT *
            FROM [dbo].[{self.response_user_table}]
            WHERE Response_id = 
            (
                SELECT TOP 1 Response_id from 
                [dbo].[{self.response_user_table}]
                WHERE LOWER(User_query) = ? and cached_from IS NULL and Response_id NOT LIKE '%llmeval'
                GROUP BY Response_id
                ORDER BY MAX(Creation_Timestamp) DESC
            ) AND Version = 1 ;
        """
        return query

     
    


    
