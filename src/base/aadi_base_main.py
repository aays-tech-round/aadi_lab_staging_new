from abc import ABC, abstractmethod
import re
from ..exception import DependencyError, ImproperlyConfigured, ValidationError
import pandas as pd
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlparse
import pyodbc

class AadiBase(ABC):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 14000)
        self.connection_string = config.get('connection_string',None)

    def log(self, message: str, title: str = "Info"):
        print(f"{title}: {message}")

    def _response_language(self) -> str:
        if self.language is None:
            return ""

        return f"Respond in the {self.language} language."
    

    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """
        Example:
        ```python
        ab.generate_sql("What are the top 10 customers by sales?")
        ```
        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list+[f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"


        return self.extract_sql(llm_response)
    
    def extract_sql(self, llm_response: str) -> str:
        """
        Example:
        ```python
        ab.extract_sql("Here's the SQL query in a code block: ```sql\nSELECT * FROM customers\n```")
        ```

        Extracts the SQL query from the LLM response. This is useful in case the LLM response contains other information besides the SQL query.
        Override this function if your LLM responses need custom extraction logic.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted SQL query.
        """

        # If the llm_response contains a CTE (with clause), extract the last sql between WITH and ;
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response is not markdown formatted, extract last sql by finding select and ; in the response
        sqls = re.findall(r"SELECT.*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
        sqls = re.findall(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        sqls = re.findall(r"```(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        return llm_response
    # ----------------- Use Any Embeddings API ----------------- #
    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> list[float]:
        pass

    # ----------------- Use Any Database to Store and Retrieve Context ----------------- #
    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        pass

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    

    # ----------------- Use Any Language Model API ----------------- #

    @abstractmethod
    def system_message(self, message: str) -> any:
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> any:
        pass

    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4

    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += "\n===Tables \n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt

    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: list[str],
        max_tokens: int = 14000,
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += "\n===Additional Context \n\n"

            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def add_sql_to_prompt(
        self, initial_prompt: str, sql_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(sql_list) > 0:
            initial_prompt += "\n===Question-SQL Pairs\n\n"

            for question in sql_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(question["sql"])
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"

        return initial_prompt

    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Example:
        ```python
        ab.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. " + \
            "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, please explain why it can't be generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log
    

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        Example:
        ```python
        ab.submit_prompt(
            [
                ab.system_message("The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."),
                ab.user_message("What are the top 10 customers by sales?"),
            ]
        )
        ```

        This method is used to submit a prompt to the LLM.

        Args:
            prompt (any): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response
    #-------------------------Summary-Generation--------------------------------#
    def generate_summary(self, query: str, df: pd.DataFrame, **kwargs):

        user_msg = f"""
            You are an expert Finance Analyst. Summarize the following data for the question: "{query}". Provide the summary in markdown format. Adhere strictly to the given instructions:

            1. Understand the asked question and the provided dataset carefully before summarizing.
            2. Do not merely list all the items in the dataframe as a summary.
            3. Format 'YearPeriod' as "Year Period". Example: 'YearPeriod' 2023001 should be shown as "Year 2023 Period 001".
            4. Display all numbers greater than 6 digits as millions.
            5. Display all numbers greater than 9 digits as billions.
            6. Limit all fractional numbers to a maximum of 2 decimal places.

            Dataset:
            {df.to_markdown()}
            """

        system_msg = """write a summary of dataframe given, do not give any table, summarize all information in paragraph."""

        message_log = [
            self.system_message(system_msg),
            self.user_message(user_msg),
        ]

        summary = self.submit_prompt(message_log, kwargs=kwargs)
        
        return summary
    
    def summary_of_summaries(self, summary_list: list, query: str, **kwargs) -> str:
        user_msg =f"""
                You are an expert Finance Analyst. Summarize the following list of summaries for the question provided.

                - Summaries: ```{summary_list}```
                - Question: ```{query}```

                Please include only the information provided in the summary list. Do not add any other information or external context.
                
                follow this instructions:
                1. If the summary is full of numbers, focus on material details and explain key insights.
                2. Mention all major points. 
                3. Do not miss any information provided into the list of summaries.
         
                """
        system_msg = """write a concise summary of given list of summaries, do not give any table, summarize all information in paragraph."""
        message_log = [
            self.system_message(system_msg),
            self.user_message(user_msg),
        ]
        overall_summary = self.submit_prompt(message_log, kwargs=kwargs)
        
        return overall_summary
    #------------------------db-connection------------------------------#
    # @abstractmethod
    # def get_connection(self, connection_string):
    #     pass


    #------------------------Visualization------------------------------#
    
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

        if len(df) > 1 and df.select_dtypes(include=['number']).shape[1] > 0:
            return True

        return False
    
    def convert_yearperiod_to_string(self,df):
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
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code
    
    def plot_suggesion(self, question: str = None, df: pd.DataFrame = None, **kwargs
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

        system_message = f"""You are an expert in data visualization using Plotly. Your task is to suggest the most suitable Plotly chart for visualizing the results of a query, based on the user's question and the given metadata.

                                    User Question: '{question}'

                                    The resulting pandas DataFrame 'df' contains the following data: \n{df.to_markdown()}
                                    Metadata: {df_metadata}

                                    Based on the following guidelines for data visualization, suggest the most appropriate Plotly chart type:

                                    1. **Looking for impact of a dimensions on another dimension:**
                                        - **Pie Chart/Donut Charts:** For single dimention
                                        - **Stacked Bar Charts:** For two dimensions.
                                        - **Sunbrust Chart:** For more than two dimensions.

                                    2. **Showing change over time or difference between periods/months :**
                                        - **Water Fall chart:** show the difference between periods and contribution of variation by different dimensions.
                                        - **Bar Chart:** Encodes value by the heights of bars from a baseline.
                                        - **Line Chart:** Encodes value by the vertical positions of points connected by line segments. Useful when a baseline is not meaningful or if the number of bars would be overwhelming.
                                        - **Box Plot:** Useful for showing the distribution of values for each time period.
                                        - **Specialized Charts:** Financial domain charts like the candlestick chart or Kagi chart.

                                    3. **Showing part-to-whole composition:**
                                        - **Pie Chart/Donut Chart:** Represents the whole with a circle, divided into parts.
                                        - **Stacked Bar Chart:** Divides each bar into sub-bars to show part-to-whole composition.
                                        - **Stacked Area Chart:** Uses shading under the line to divide the total into sub-group values.
                                        - **Hierarchical Charts:** Marimekko plot, treemap for showing hierarchical relationships.

                                    4. **Looking at data distribution:**
                                        - **Bar Chart:** Used for qualitative variables with discrete values.
                                        - **Histogram:** Used for quantitative variables with numeric values.
                                        - **Density Curve:** Smoothed estimate of the underlying distribution.
                                        - **Violin Plot:** Compares numeric value distributions between groups using a density curve.
                                        - **Box Plot:** Summarizes statistics for comparing distributions between groups.

                                    5. **Comparing values between groups:**
                                        - **Bar Chart:** Compares values by assigning a bar to each group.
                                        - **Dot Plot:** Uses point positions to indicate value, useful without a vertical baseline.
                                        - **Line Chart:** Compares values across time with one line per group.
                                        - **Grouped Bar Chart:** Compares data across two grouping variables with multiple bars at each location.
                                        - **Violin/Box Plot:** Compares data distributions between groups.
                                        - **Funnel Chart:** Shows how quantities move through a process.
                                        - **Bullet Chart:** Compares a true value to one or more benchmarks.

                                    6. **Observing relationships between variables:**
                                        - **Scatter Plot:** Standard for showing the relationship between two variables.
                                        - **Bubble Chart:** Adds color, shape, or size to each point to indicate additional variables.
                                        - **Connected Scatter Plot:** Connects points with line segments when a third variable represents time.
                                        - **Dual-Axis Plot:** Combines a line chart and bar chart with a shared horizontal axis for a temporal third variable.
                                        - **Heatmap:** Shows the relationship between groups for non-numeric variables or purely numeric data.

                                    7. **Looking at geographical data:**
                                        - **Choropleth:** Colors in geopolitical regions.
                                        - **Cartogram:** Uses the size of each region to encode value, with some distortion in shapes and topology.

                                    Analyze the provided data and metadata, and suggest the most appropriate Plotly chart type to effectively visualize the data.
                                    """

        user_message = "Can you suggest a plotly chart as following the query and meta-data. Do not answer with any explanations -- only chart name."
        message_log = [
            self.system_message(system_message),
            self.user_message(user_message),
        ]



        suggested_chart = self.submit_prompt(message_log, kwargs=kwargs)

        return suggested_chart
    
    def generate_plotly_code(self, question: str = None, sql: str = None, df: pd.DataFrame = None, **kwargs
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
        plot_suggestion = self.plot_suggesion(question, df)
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

        # df = df[(df.select_dtypes(include=[np.number]) >= 0).all(axis=1)]

        user_msg = f"""Can you generate {plot_suggestion} Python plotly code to chart the results of the dataframe? 
                        Assume the data is in a pandas dataframe called 'df'. 
                        Instruction:
                        1. If there is only one value in the dataframe, use an Indicator. 
                        2. For PIE CHARTS, DONUT CHARTS, and WATERFALL CHARTS, if there are multiple values below 1, combine them into a single category named 'Others' before plotting.                                
                        3. Do not use append function in dataframe instead use concat. for an example:
                            Instead of "df = df.append('Profit_Center_Desc': 'Others', 'PercentageContribution': 'others_sum', ignore_index=True)"
                            Use "pd.concat([df, pd.DataFrame('Profit_Center_Desc': ['Others'], 'PercentageContribution': ['others_sum'])], ignore_index=True)
                        4. Include a title that summarizes the main insight from the data, make it bold, and left-justify it.
                        5. If the chart has axes, make the axis titles bold. (For pie chart do not do.)
                        6. Respond with only Python code. Do not answer with any explanations -- just the code."""
            
        message_log = [
            self.system_message(system_msg),
            self.user_message(user_msg),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code)), plot_suggestion
    
    def get_plotly_figure(
        self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        **Example:**
        ```python
        fig = ab.get_plotly_figure(
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
    
    def redraw_chart(self, message_id:str, child_question_id:int, Updated_prompt:str, **kwargs):

        conn = pyodbc.connect(self.connection_string)
        query = "SELECT * FROM ResponseUserData"

        # Execute the query and fetch the results into a DataFrame
        df = pd.read_sql(query, conn)

        # Close the connection
        conn.close()
        
        
        plotly_code = df[(df.Response_id==message_id) & (df.Section_id==child_question_id)].sort_values(by='Version').iloc[-1]['python_code_figure']
        dataframe = df[(df.Response_id==message_id) & (df.Section_id==child_question_id)]['DataFrame'].reset_index(drop=True)[0]

        

        dataframe = pd.read_json(dataframe)
        dataframe = self.convert_yearperiod_to_string(dataframe)
        df_metadata = {c: dt for c, dt in zip(dataframe.columns, dataframe.dtypes)}

        system_msg = """You update plotly code basis on the instructions, old plotly code and dataframe"""
        user_msg =f"""Update this plotly code ```{plotly_code}``` for the data: {dataframe} modified to new dataframe as per {Updated_prompt}, Metadata: {df_metadata}, based on the following instructions:

                            1. {Updated_prompt} 
                            2. Our dataframe “df” only contains the following columns: {dataframe.columns}. For any additional field or value required in plot, compute it accurately after understanding the requiremnet thoroughly.
                            3. For PIE CHARTS, DONUT CHARTS, and 100% STACKED BAR CHARTS, if there are multiple small values, show top 10 and combine rest of the categories into a single category named 'Others' before plotting. 
                            4. Do not use append function in dataframe instead use concat. for an example:
                                Instead of "df = df.append('Profit_Center_Desc': 'Others', 'PercentageContribution': 'others_sum', ignore_index=True)"
                                Use "pd.concat([df, pd.DataFrame('Profit_Center_Desc': ['Others'], 'PercentageContribution': ['others_sum'])], ignore_index=True)
                            5. Include a title that summarizes the main insight from the data make it in bold.
                            6. Axis titles should be in bold.
                            7. Ensure all values and fields are shown in the graph with same labels as in dataframe. 
                            8. Ensure the code runs without any errors.
                            9. Ensure each and every variable is defined in the code you return
                            10. Respond with only Python code. Do not answer with any explanations -- just the code."""

        message_log = [
            self.system_message(system_msg),
            self.user_message(user_msg),
        ]
        new_plotly_code = self.submit_prompt(message_log, kwargs=kwargs)
        
        return self._sanitize_plotly_code(self._extract_python_code(new_plotly_code)), dataframe
        
    #------------------------------------------------------db-connection----------------------------------------------------#
    
    def connect_to_mssql(self, odbc_conn_str: str):
        """
        Connect to a Microsoft SQL Server database. This is just a helper function to set [`ab.run_sql`]

        Args:
            odbc_conn_str (str): The ODBC connection string.

        Returns:
            None
        """
        try:
            import pyodbc
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install pyodbc"
            )

        try:
            import sqlalchemy as sa
            from sqlalchemy.engine import URL
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install sqlalchemy"
            )

        connection_url = URL.create(
            "mssql+pyodbc", query={"odbc_connect": odbc_conn_str}
        )

        from sqlalchemy import create_engine

        engine = create_engine(connection_url)

        def run_sql_mssql(sql: str):
            # Execute the SQL statement and return the result as a pandas DataFrame
            with engine.begin() as conn:
                df = pd.read_sql_query(sa.text(sql), conn)
                conn.close()
                return df

            raise Exception("Couldn't run sql")
        self.dialect = "T-SQL / Microsoft SQL Server"
        self.run_sql = run_sql_mssql
        self.run_sql_is_set = True
        
        
    def train(
            self,
            question: str = None,
            sql: str = None,
            ddl: str = None,
            documentation: str = None
        ) -> str:
            """
            **Example:**
            ```python
            ab.train()
            ```

            Train aadi on a question and its corresponding SQL query.

            Args:
                question (str): The question to train on.
                sql (str): The SQL query to train on.
                ddl (str):  The DDL statement.
                documentation (str): The documentation to train on.
                plan (TrainingPlan): The training plan to train on.
            """

            if question and not sql:
                raise ValidationError("Please also provide a SQL query")

            if documentation:
                print("Adding documentation....")
                return self.add_documentation(documentation)

            if sql:
                # if question is None:
                #     question = self.generate_question(sql)
                #     print("Question generated with sql:", question, "\nAdding SQL...")
                return self.add_question_sql(question=question, sql=sql)

            if ddl:
                print("Adding ddl:", ddl)
                return self.add_ddl(ddl)
            
    def run_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        """
        Example:
        ```python
        ab.run_sql("SELECT * FROM my_table")
        ```

        Run a SQL query on the connected database.

        Args:
            sql (str): The SQL query to run.

        Returns:
            pd.DataFrame: The results of the SQL query.
        """
        raise Exception(
            "You need to connect to a database first by running ab.connect_to_snowflake(), ab.connect_to_postgres(), similar function, or manually set ab.run_sql"
        )
    
    def is_sql_valid(self, sql: str) -> bool:
        """
        Example:
        ```python
        ab.is_sql_valid("SELECT * FROM customers")
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
    
    # @abstractmethod
    # def store_response(self, response_id, Section_id, summary, fig, formatted_code,df, exception, python_code_figure, version = 1, system_user_id=None):
    #     pass

    #-----------------------------------Follow Up----------------------------------------------------------#

    #----------------------------------------FollowUp---------------------------------------------#
    def follow_up_question_generation(self,previous_asked_question:str, current_question:str, sql:str = None, summary:str=None, **kwargs) -> str:
        system_msg = f"""You are a helpful assistant highly skilled at rephrasing a question based on a previously asked question, AI-generated SQL, and the summary.
                            Previously Asked Question:
                            {previous_asked_question}
                            AI Reply: """
        if sql is not None:
            system_msg +=f"""SQL:
                                {sql}\n"""
        if summary is not None:
            system_msg +=f"""
                            Summary:
                            {summary}\n\n"""

        system_msg +=f"""Question to rephrase: {current_question}"""

        user_msg = "Give me a rephrased question based on the previous question and answers so that a TEXT TO SQL ENGINE CAN EXTRACT it. JUST GIVE ME THE QUESTION."

        message_log = [
            self.system_message(system_msg),
            self.user_message(user_msg),
        ]
        rephased_question = self.submit_prompt(message_log, kwargs=kwargs)
        
        return rephased_question
    

    #------------------------------------FORECAST--------------------------------------------------------------------------------#
    def forecast_extract_query(self,query) -> bool:
        query = query.lower()

        keywords = ['forecast', 'predict']

        for key in keywords:
            if key in query:
                return True
            else:
                continue
            
        return False

    def extract_python_code(self, text):
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches


    def forecast_func(self, df,user_query, **kwargs):

        system_message = """
                        You are a helpful agent that takes a dataframe as df. You will be provided with two columns, YearPeriod and one continuous dtype column other than YearPeriod like GSV, NSV, PercentageofGSV, PercentageofNSV, PrimeCost, etc.
                        1- BUILD A PYTHON CODE TO TAKE df and make a forecasting model using ARIMA and statsmodels library.
                        2- If you do not have YearPeriod column, then DO NOT MAKE ANY CODE, just return null.
                        3- Forecast for next periods based on user query, by default PREDICT for NEXT three periods. If the last period is 2023013, the next periods are 2024001, 2024002, 2024003.
                        4- YearPeriod is in the format 2021001, 2021003, 2021013. The format of YearPeriod is YEAR + 0 + Period (i.e., 20XX013). If the period is 10, 11, 12, 13, there is one zero preceding. If less than 10, then two zeros preceding. There are 13 periods in a year. DO NOT CONVERT IT INTO DATETIME; just take the series and do the prediction.
                        5- The resulting dataframe should include both actual and predicted columns with two columns: YearPeriod and Value.
                        6- Ensure the code runs without any errors.
                        7- Return example data like the following:

                        ```python
                        
                        sample_data = {
                            'YearPeriod': ['2023001', '2023002', '2023003', '2023004', '2023005', '2023006', '2023007', '2023008', '2023009', '2023010',
                                        '2023011', '2023012', '2023013', '2024001', '2024002', '2024003', '2024004', '2024005'],
                            'TotalGSV': [668351700.0, 682920700.0, 684651300.0, 616379700.0, 660259200.0, 609385600.0, 675362100.0, 814772900.0, 797045300.0,
                                        781543500.0, 728425800.0, 632715100.0, 618148200.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            'Predicted Value': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 625673400.0, 636437000.0, 641012200.0, 647703600.0, 648008700.0]
                        }

                        df = pd.DataFrame(sample_data)

                        # Filter out future periods for actual values
                        df_actual = df[df['TotalGSV'] > 0]
                        # Filter out past periods for predicted values
                        df_predicted = df[df['Predicted Value'] > 0]

                        # Combine the filtered data
                        df_filtered = pd.concat([df_actual[['YearPeriod', 'TotalGSV']].rename(columns={'TotalGSV': 'Value'}), 
                                                df_predicted[['YearPeriod', 'Predicted Value']].rename(columns={'Predicted Value': 'Value'})])

                        # Add a column to differentiate between actual and predicted values
                        df_filtered['Type'] = ['Actual'] * len(df_actual) + ['Predicted'] * len(df_predicted)

                        return df_filtered"""
        
        user_message = f"Generate a python code and build a model using arima on dataframe {df}. The user query is \n\n {user_query}. \n\n consider the data already in df as dataframe, dont create dataframe"

        message_log = [
                self.system_message(system_message),
                self.user_message(user_message),
            ]

        code = self.submit_prompt(message_log, kwargs=kwargs)
        code = self.extract_python_code(code)[0]
        print(code)
        if code:
            try:
                local_vars = {'df': df}
                exec(code, globals(), local_vars)
                result_df = local_vars.get('df_filtered', None)
                if result_df is not None:
                    print(result_df)
                    return result_df, code
                else:
                    return "Error: 'df_filtered' not found in the generated code", 400
            except Exception as e:
                return f"Error: {e}", 400
        else:
            return "Not able to run the code", 400
