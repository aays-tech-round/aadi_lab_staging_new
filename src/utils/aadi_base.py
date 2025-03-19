import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from openai import AzureOpenAI
import re
import sqlparse
from src.utils.datastore import *

class aadibase:
    def __init__(self, PromptsObj, SQLQueryObj) -> None:
        self.PromptsObj = PromptsObj
        self.SQLQueryObj = SQLQueryObj

    def is_sql_valid(self, sql: str) -> bool:
        """
        Example:
        ```python
        vn.is_sql_valid("SELECT * FROM customers")
        ```
        Checks if the SQL query is valid. This is usually used to check if we should run the SQL query or not.
        By default it checks if the SQL query is a SELECT statement. You can override this method to enable running other types of SQL queries.

        Args:
            sql (str): The SQL query to check.

        Returns:
            bool: True if the SQL query is valid, False otherwise.
        """

        parsed = sqlparse.parse(sql)

        for statement in parsed:
            if statement.get_type() == 'SELECT':
                return True

        return False
    
    def create_summary_prompt(self,query,df):
        """
        This function is used to create summary prompt

        Args:
            query: User input from front end of AADI app
            df : Dataframe generated via SQL

        Returns:
            summary prompt
        """
        prompt = self.PromptsObj.prompt_create_summary(df = df,
                                                       query = query)
        return prompt


    def generate_summary(self, LLMClient, LLMModelName, query, df):
        """
        This function is used to generate summary 

        Args:
            LLMClient : LLM object used for setting up connection
            LLMModelName : LLM model used for summary generation
            query : User input from front end of AADI app
            df : Dataframe generated via SQL

        Returns:
            A summary for the query given by user prompt 
        """
        prompt= self.create_summary_prompt(query,df)
        chat_completion = LLMClient.chat.completions.create(
        model=LLMModelName,
        temperature=0,
        messages = self.PromptsObj.prompt_generate_summary(prompt = prompt)
        )
        return chat_completion.choices[0].message.content

    def summary_of_summaries(self, LLMClient, LLMModelName, summary_list: list, query: str) -> str:  
        """ This function is used to generate summary for summary of all questions including follow up questions

        Args:
            LLMClient : LLM object used for setting up connection
            LLMModelName : LLM model used for summary generation
            query : User input from front end of AADI app
            df : Dataframe generated via SQL

        Returns:
            A summarised summary for the query given by user prompt 
        """
        messages = self.PromptsObj.prompt_summary_of_summaries(summary_list = summary_list,
                                                             query = query) 

        chat_completion = LLMClient.chat.completions.create(
            model=LLMModelName,
            temperature=0,
            messages = messages
        )
        return chat_completion.choices[0].message.content

    
        
    def should_generate_chart(self, df: pd.DataFrame) -> bool:
        """
        Example:
        ```python
        ab.should_generate_chart(df)
        ```

        Checks if a chart should be generated for the given DataFrame. By default, it checks if the DataFrame has more than one row and has numerical columns.
        You can override this method to customize the logic for generating charts.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Returns:
            bool: True if a chart should be generated, False otherwise.
        """

        if len(df) >= 1 and df.select_dtypes(include=['number']).shape[1] > 0:
            return True

        return False

    # ____________________________data visualization code_____________________________________
    def convert_yearperiod_to_string(self,df):
        """ Converts year period to string

        Args:
            df (_type_): SQL generated Dataframe

        Returns:
            a Dataframe where all YearPeriod columns are converted to string
        """
        # Identify columns related to yearperiod
        yearperiod_columns = [col for col in df.columns if 'Period' in col or 'Year' in col]
        
        # Convert identified columns to string if any are found
        if yearperiod_columns:
            for col in yearperiod_columns:
                df[col] = df[col].astype(str)
        
        if 'FY' in df.columns:
            df['FY'] = df['FY'].astype(str)
        
        columns_to_convert = ['Profit_Center', 'Functional_Area', 'Cost_Center', 'GL_account']
        for col in df.columns:
            for keyword in columns_to_convert:
                if keyword in col:
                    df[col] = df[col].astype(str)
        
        return df
    
    def _extract_python_code(self, markdown_string: str) -> str:

        """
            Extracts the first Python code block from a given markdown string.

            This function searches the provided markdown string for code blocks
            specified with triple backticks (```) and labeled as Python. It
            extracts and returns the content of the first Python code block found.
            If no Python code blocks are found, it returns the original markdown string.

            Args:
                markdown_string (str): A string containing markdown content.

            Returns:
                str: The content of the first Python code block if found, otherwise
                     the original markdown string.

            Examples:
                >>> markdown = "Some text\n```python\nprint('Hello, world!')\n```\nMore text"
                >>> _extract_python_code(markdown)
                "print('Hello, world!')"

                >>> markdown = "Some text without code block"
                >>> _extract_python_code(markdown)
                "Some text without code block"
            """
        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        """
            Removes the 'fig.show()' statement from the Plotly code.

            Args:
                raw_plotly_code (str): The Plotly code as a string.

            Returns:
                str: The sanitized Plotly code without the 'fig.show()' statement.
        """
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code

    def plot_suggesion(self, LLMClient, LLMModelName, question: str = None, df: pd.DataFrame = None, **kwargs
                       ) -> str:

        """
            Suggests the most suitable Plotly chart type for visualizing the results of a query.

            Args:
                question (str, optional): User's question regarding the data visualization.
                df (pd.DataFrame, optional): The pandas DataFrame containing the data to visualize.

            Returns:
                str: The name of the suggested Plotly chart type.

            The function uses the provided question and DataFrame metadata to create a prompt for
            the Azure OpenAI service, which is expected to return the name of the most appropriate
            Plotly chart type based on the data and the guidelines for various visualization types.
        """

        df_metadata = {c: dt for c, dt in zip(df.columns, df.dtypes)}

        prompt = self.PromptsObj.prompt_plot_sugguestion(question = question,
                                                         df = df,
                                                         df_metadata = df_metadata)

        response = LLMClient.chat.completions.create(
            model=LLMModelName,
            messages=prompt,
            stop=None,
            temperature=0.0,
        )

        suggested_chart = response.choices[0].message.content

        return suggested_chart

    def redraw_chart(self, LLMClient, LLMModelName, message_id:str, child_question_id:int, Updated_prompt:str):
        """ Function to redraw chart

        Args:
            LLMClient : LLM object used for setting up connection
            LLMModelName : LLM model used for summary generation
            message_id : Id of message
            child_question_id (int): Id of Child question
            Updated_prompt (str):  Prompt that gets updated 

        Returns:
            A re-drawn chart
        """

        conn = get_connection()
        query = self.SQLQueryObj.sql_query_response_user_data()
    
        # Execute the query and fetch the results into a DataFrame
        df = pd.read_sql(query, conn)
    
        # Close the connection
        conn.close()
        plotly_code = df[(df.Response_id==message_id) & (df.Section_id==child_question_id)].sort_values(by='Version').iloc[-1]['python_code_figure']
        dataframe = df[(df.Response_id==message_id) & (df.Section_id==child_question_id)]['DataFrame'].reset_index(drop=True)[0]

        dataframe = pd.read_json(dataframe)
        dataframe = self.convert_yearperiod_to_string(dataframe)

        df_metadata = {c: dt for c, dt in zip(dataframe.columns, dataframe.dtypes)}
        
        messages = self.PromptsObj.prompt_redraw_chart(plotly_code = plotly_code,
                                                       dataframe = dataframe,
                                                       Updated_prompt = Updated_prompt,
                                                       df_metadata = df_metadata
                                                       )
        
        
        chat_completion = LLMClient.chat.completions.create(
            model=LLMModelName,
            messages=messages
        )

        plotly_code_new = chat_completion.choices[0].message.content
        
        return self._sanitize_plotly_code(self._extract_python_code(plotly_code_new)), dataframe

    def generate_plotly_code(self, LLMClient, 
                             LLMModelName, 
                             question: str = None, 
                             sql: str = None, 
                             df: pd.DataFrame = None, 
                             **kwargs
                             ) -> str:

        """
            Generates Plotly code for a chart based on a user question, SQL query, and DataFrame.

            Args:
                question (str, optional): The user's question.
                sql (str, optional): The SQL query used to obtain the data.
                df (str, optional): The DataFrame containing the data.
                **kwargs: Additional keyword arguments.

            Returns:
                str: The generated Plotly Python code.
            """
        plot_suggestion = self.plot_suggesion(LLMClient, LLMModelName, question, df)
        df_metadata = {c: dt for c, dt in zip(df.columns, df.dtypes)}

        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df}. Metadata: {df_metadata}"

        if 'YearPeriod' in df.columns:
            df['YearPeriod'] = df['YearPeriod'].astype(str)

        prompt = self.PromptsObj.prompt_generate_plotly_code(system_msg = system_msg,
                                                             plot_suggestion = plot_suggestion)


        response = LLMClient.chat.completions.create(
            model=LLMModelName,
            messages=prompt,
            stop=None,
            temperature=0,
        )

        plotly_code = response.choices[0].message.content

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code)), plot_suggestion

    def get_plotly_figure(
            self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        **Example:**
        ```python
        fig = vn.get_plotly_figure(
            plotly_code="fig = px.bar(df, x='name', y='salary')",
            df=df
        )
        fig.show()
        ```
        Get a Plotly figure from a dataframe and Plotly code.

        Args:
            df (pd.DataFrame): The dataframe to use.
            plotly_code (str): The Plotly code to use.

        Returns:
            plotly.graph_objs.Figure: The Plotly figure.
        """
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)

            fig = ldict.get("fig", None)
        except Exception as e:
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Decision-making for plot type
            if len(numeric_cols) >= 2:
                # Use the first two numeric columns for a scatter plot
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                # Use a bar plot if there's one numeric and one categorical column
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
                # Use a pie chart for categorical data with fewer unique values
                fig = px.pie(df, names=categorical_cols[0])
            else:
                # Default to a simple line plot if above conditions are not met
                fig = px.line(df)

        if fig is None:
            return None

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig


    #----------------------------------------FollowUp---------------------------------------------#
    def follow_up_question_generation(self, LLMClient, LLMModelName, previous_asked_question:str, current_question:str, sql:str = None, summary:str=None) -> str:
        """_summary_

        Args:
            LLMClient : LLM object used for setting up connection
            LLMModelName : LLM model used for summary generation
            query : User input from front end of AADI app
            df : Dataframe generated via SQL
        Returns:
            The follow up questions
        """
        system_prompt = f"""You are a helpful assistant highly skilled at rephrasing a question based on a previously asked question, AI-generated SQL, and the summary.
                            Previously Asked Question:
                            {previous_asked_question}
                            AI Reply: """
        if sql is not None:
            system_prompt +=f"""SQL:
                                {sql}\n"""
        if summary is not None:
            system_prompt +=f"""
                            Summary:
                            {summary}\n\n"""

        system_prompt +=f"""Question to rephrase: {current_question}"""

        messages = self.PromptsObj.prompt_follow_up_question_generation(system_prompt = system_prompt)
                
        response = LLMClient.chat.completions.create(
                            model=LLMModelName,
                            messages=messages,
                            stop=None,
                            temperature=0.2,
                        )
        
        question = response.choices[0].message.content

        return question
