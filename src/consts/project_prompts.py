import pandas as pd
import json
class Prompts:

    def prompt_intent_llm_agent(self, **kwargs):
        query=kwargs['query']
        json_template = {
        "task": "<aadi_nlp_to_sql_query|prohibited_nlp_to_sql|greeting|generic_question>",
        "reason": "<Provide a brief explanation of why this task was selected>",
        "rephrased_query":"<Rephrased query mapped to the correct quarter format as per the database>",
        "original_query": query
        }
        json_string = json.dumps(json_template, indent=4)
        return f"""
            A. You are Query Classifier with 100% F1 score.
            - Your task is to classify a user query into one of 4 categories <aadi_nlp_to_sql_query|prohibited_nlp_to_sql|greeting|generic_question>:
                1. aadi_nlp_to_sql_query: 
                    Examples:
                    - **Query**: "Recommend 3 or 5 focus actions to increase GSV?"
                        **Output**: `task:aadi_nlp_to_sql_query` 
                    - **Query**: "What Is My  depreciation As Percentage Of GSV For Period 13 Year 2023?
What Is My  depreciation As Percentage Of GSV For Period 13 Year 2023?"               
                        **Output**: `task:aadi_nlp_to_sql_query`
                    - **Query**: "What Is My Total NSV YTD for 2023 period 13?"               
                        **Output**: `task:aadi_nlp_to_sql_query`
                    - **Query**: "Compare Net Income For 2023 Period 4 With 2023 Period 2?"               
                        **Output**: `task:aadi_nlp_to_sql_query`
                    - **Query**: "Give Me Last 13 Period Prime Cost Since 2023 Period 2?"               
                        **Output**: `task:aadi_nlp_to_sql_query`
                    - **Query**: "Compare Value Of Prime Cost For P7 2023, With Past Trend?"               
                        **Output**: `task:aadi_nlp_to_sql_query`
                    - **Query**: "Give Me Decomposition Analysis For GSV For 2023 Period 7?"               
                        **Output**: `task:aadi_nlp_to_sql_query`
                    - **Query**: "Give Me Variance Analysis For GSV For 2023 Period 7?"               
                        **Output**: `task:aadi_nlp_to_sql_query`
                    - **Query**: "Give Me Treand Analysis For GSV For 2023 Period 7?"               
                        **Output**: `task:aadi_nlp_to_sql_query`
                2. **prohibited_nlp_to_sql**: Queries that attempt to insert, modify or delete data in a database, which should be restricted due to security, ethical, or technical reasons.  
                    Examples:
                    - **Query**: "Delete all records."  
                        **Output**: `task:prohibited_nlp_to_sql`  
                    - **Query**: "Insert blank records."  
                        **Output**: `task:prohibited_nlp_to_sql`
                    - **Query**: "Remove all test results from last year."  
                        **Output**: `task:prohibited_nlp_to_sql`  
                    - **Query**: "Update the test result for 98765 to 'negative'."  
                        **Output**: `task:prohibited_nlp_to_sql`  
                    - **Query**: "Change all pending lab reports to 'approved'."  
                        **Output**: `task:prohibited_nlp_to_sql`  
                    - **Query**: "Modify the turnaround time for all tests to 24 hours."  
                        **Output**: `task:prohibited_nlp_to_sql`  

                3. **greeting**: The query is a friendly or polite opening that does not require a detailed response beyond acknowledgment. These queries are commonly used to initiate a conversation in a casual or formal manner (e.g., "Hi, how are you?" or "Good morning!").  
                    Examples:  
                    - **Query**: "Hi, how are you?"  
                        **Output**: `task:greeting`  
                    - **Query**: "Hello!"  
                        **Output**: `task:greeting`  
                    - **Query**: "Good morning. What's up?"  
                        **Output**: `task:greeting`  
                    - **Query**: "Hey there!"  
                        **Output**: `task:greeting`  
                    - **Query**: "Good afternoon!"  
                        **Output**: `task:greeting`  
                    - **Query**: "Yo!"  
                        **Output**: `task:greeting`  
                    - **Query**: "How's it going?"  
                        **Output**: `task:greeting`                      
 
                4. **generic_question**: The query seeks general information, facts, or explanations on a broad range of topics without specifying a particular document or dataset. These questions can be about definitions, concepts, events, time, locations, or general knowledge (e.g., "What is the capital of France?" or "How does machine learning work?").  
                    Examples:  
                    - **Query**: "What is the capital of France?"  
                        **Output**: `task:generic_question`  
                    - **Query**: "How does machine learning work?"  
                        **Output**: `task:generic_question`  
                    - **Query**: "Who is the CEO of Tesla?"  
                        **Output**: `task:generic_question`  
                    - **Query**: "What are the benefits of drinking water?"  
                        **Output**: `task:generic_question`  
                    - **Query**: "Can you explain the theory of relativity?"  
                        **Output**: `task:generic_question`  
                    - **Query**: "What is the time today?"  
                        **Output**: `task:generic_question`  
                    - **Query**: "What day of the week is it?"  
                        **Output**: `task:generic_question`  
                    - **Query**: "How many days are in a leap year?"  
                        **Output**: `task:generic_question`  
                    - **Query**: "What is the population of India?"  
                        **Output**: `task:generic_question`  
                    - **Query**: "What is the weather like today?"  
                        **Output**: `task:generic_question` 

            Output your decision with 100% F1 score in the following JSON format:
            ```json
            {json_string}
            ```
    
            Do not return anything outside this JSON format. 
            
            Here is the query: 
            "{query}"
            """ 


    def prompt_generate_summary_len_df_ge_two(self, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        prompt = [
                        {
                            "role": "system",
                            "content": f"""
                            You are a financial analyst. Your task is to summarize financial data in a style consistent with professional financial analysis reports. Below are examples of summaries written by financial analysts:

                            Example 1:
                            In the current month, the Gross Sales Value (GSV) trend is showing positive indicators within the set confidence interval, as corroborated by the data from the last two years. Delving into the trend variances, we observe a slight deviation with the GSV declining by a marginal 0.50% from the confidence interval. Further analysis presents a promising picture with the rolling average for GSV standing at 538 million, which marks a significant milestone. On a more granular level, the month-to-month analysis indicates a robust growth of 10.30%, equating to an increase of 50 million from the preceding month. However, a year-to-year comparison showcases a slight dip, where the GSV fell by 8 million, representing a decrease of 1.30% compared to the same period last year.

                            Example 2:
                            For the Year-To-Date analysis of Gross Sales Value (GSV), the trend showcases that the value currently stands at 3318 million. Delving into the nuances of the trend variances, GSV has exhibited a commendable growth trajectory. The positive variation noted so far this year is an impressive 318 million, translating to a robust increase of 10.30%.

                            Example 3:
                            The year-on-year difference in Gross Sales Value (GSV) can be chiefly attributed to four major factors as outlined by the accounting profile. Quantity rebates are the most significant contributor, accounting for 60% of the variance. Returns also play a substantial role, making up 20% of the year-on-year change. GSV trading deductions and sales increases each contribute 10%, making them less impactful but still relevant to the overall fluctuation in GSV. Therefore, these four elements collectively explain the year-on-year variation in GSV.

                            The user asked the question: '{kwargs['question']}'

                            Below is a pandas DataFrame with the results of the query:
                            {kwargs['df'].to_markdown()}

                            Your task is to summarize the data in response to the question, incorporating the provided data description: {kwargs['describe']}. Ensure that your summary is clear, concise, and follows the style of a professional financial analyst.Use bullet points for important points.
                            """
                        },
                        {
                            "role": "user",
                            "content": "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary."
                        }
                    ]
        return prompt
    
        
    def prompt_generate_summary_len_df_lt_two(self, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        prompt = [
                {
                    "role": "system",
                    "content": f"""
                            You are a financial analyst. You have to adopt the style of a financial analyst. Few example summaries written by financial analysts are below: \n\n
                            1. In the current month, the Gross Sales Value (GSV) trend is showing positive indicators within the set confidence interval, as corroborated by the data from the last two years. Delving into the trend variances, we observe a slight deviation with the GSV declining by a marginal 0.50% from the confidence interval. Further analysis presents a promising picture with the rolling average for GSV standing at 538 million, which marks a significant milestone. On a more granular level, the month-to-month analysis indicates a robust growth of 10.30%, equating to an increase of 50 million from the preceding month. However, a year-to-year comparison showcases a slight dip, where the GSV fell by 8 million, representing a decrease of 1.30% compared to the same period last year.\n
                            2. For the Year-To-Date analysis of Gross Sales Value (GSV), the trend showcases that the value currently stands at 3318 Millions. Delving into the nuances of the trend variances, GSV has exhibited a commendable growth trajectory. The positive variation noted so far this year is an impressive 318 Millions, translating to a robust increase of 10.30%.\n
                            3. The year-on-year difference in Gross Sales Value (GSV) can be chiefly attributed to four major factors as outlined by the accounting profile. Quantity rebates are the most significant contributor, accounting for 60% of the variance. Returns also play a substantial role, making up 20% of the year-on-year change. GSV trading deductions and sales increases each contribute 10%, making them less impactful but still relevant to the overall fluctuation in GSV. Therefore, these four elements collectively explain the year-on-year variation in GSV.\n\n
                            The user asked the question: '{kwargs["question"]}'\n\nThe following is a pandas DataFrame with the results of the query: \n{kwargs['df'].to_markdown()}\n summarize the data as per the question.
                            """
                },
                {
                    "role": "user",
                    "content": "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary."
                }
            ]
        
        return prompt
    
    def prompt_summary_of_summaries(self,**kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        user_msg =f"""
            You are an expert Finance Analyst. Summarize the following list of summaries for the question provided.

            - Summaries: ```{kwargs['summary_list']}```
            - Question: ```{kwargs['query']}```

            Please include only the information provided in the summary list. Do not add any other information or external context.
            
            follow this instructions:
            1. If the summary is full of numbers, focus on material details and explain key insights.
            2. Mention all major points. 
            3. Do not miss any information provided into the list of summaries.
        
            """
        
        system_msg = "write a concise summary of given list of summaries, do not give any table, summarize all information in paragraph."
        messages = [user_msg, system_msg]

        return messages
    
    def prompt_plot_sugguestion(self,**kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        question = kwargs["question"]
        df_metadata = kwargs['df_metadata']
        df = kwargs['df']

        prompt = [
            {
                "role": "system",
                "content": f"""You are an expert in data visualization using Plotly. Your task is to suggest the most suitable Plotly chart for visualizing the results of a query, based on the user's question and the given metadata.

                                    User Question: '{question}'

                                    The resulting pandas DataFrame 'df' contains the following data: \n{df.to_markdown()}
                                    Metadata: {df_metadata}

                                    Based on the following guidelines for data visualization, suggest the most appropriate Plotly chart type:

                                    1. - when showing comparison of 2 or more numbers, use bar chart

                                    2. **Looking for impact of a dimensions on another dimension:**
                                        - **Pie Chart/Donut Charts:** For single dimention
                                        - **Stacked Bar Charts:** For two dimensions.
                                        - **Sunbrust Chart:** For more than two dimensions.

                                    3. **Showing change over time or difference between periods/months :**
                                        - **Water Fall chart:** show the difference between periods and contribution of variation by different dimensions.
                                        - **Bar Chart:** Encodes value by the heights of bars from a baseline.
                                        - **Line Chart:** Encodes value by the vertical positions of points connected by line segments. Useful when a baseline is not meaningful or if the number of bars would be overwhelming.
                                        - **Box Plot:** Useful for showing the distribution of values for each time period.
                                        - **Specialized Charts:** Financial domain charts like the candlestick chart or Kagi chart.

                                    4. **Showing part-to-whole composition:**
                                        - **Pie Chart/Donut Chart:** Represents the whole with a circle, divided into parts.
                                        - **Stacked Bar Chart:** Divides each bar into sub-bars to show part-to-whole composition.
                                        - **Stacked Area Chart:** Uses shading under the line to divide the total into sub-group values.
                                        - **Hierarchical Charts:** Marimekko plot, treemap for showing hierarchical relationships.

                                    5. **Looking at data distribution:**
                                        - **Bar Chart:** Used for qualitative variables with discrete values.
                                        - **Histogram:** Used for quantitative variables with numeric values.
                                        - **Density Curve:** Smoothed estimate of the underlying distribution.
                                        - **Violin Plot:** Compares numeric value distributions between groups using a density curve.
                                        - **Box Plot:** Summarizes statistics for comparing distributions between groups.

                                    6. **Comparing values between groups:**
                                        - **Bar Chart:** Compares values by assigning a bar to each group.
                                        - **Dot Plot:** Uses point positions to indicate value, useful without a vertical baseline.
                                        - **Line Chart:** Compares values across time with one line per group.
                                        - **Grouped Bar Chart:** Compares data across two grouping variables with multiple bars at each location.
                                        - **Violin/Box Plot:** Compares data distributions between groups.
                                        - **Funnel Chart:** Shows how quantities move through a process.
                                        - **Bullet Chart:** Compares a true value to one or more benchmarks.

                                    7. **Observing relationships between variables:**
                                        - **Scatter Plot:** Standard for showing the relationship between two variables.
                                        - **Bubble Chart:** Adds color, shape, or size to each point to indicate additional variables.
                                        - **Connected Scatter Plot:** Connects points with line segments when a third variable represents time.
                                        - **Dual-Axis Plot:** Combines a line chart and bar chart with a shared horizontal axis for a temporal third variable.
                                        - **Heatmap:** Shows the relationship between groups for non-numeric variables or purely numeric data.

                                    8. **Looking at geographical data:**
                                        - **Choropleth:** Colors in geopolitical regions.
                                        - **Cartogram:** Uses the size of each region to encode value, with some distortion in shapes and topology.


                                    Analyze the provided data and metadata, and suggest the most appropriate Plotly chart type to effectively visualize the data.
                                    """

            },
            {
                "role": "user",
                "content": "Can you suggest a plotly chart as following the query and meta-data. Do not answer with any explanations -- only chart name."
            }
        ]
         
        return prompt
    
    def prompt_redraw_chart(self,**kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        prompt = f"""Update this plotly code ```{kwargs['plotly_code']}``` for the data: {kwargs['dataframe']} modified to new dataframe as per {kwargs['Updated_prompt']}, Metadata: {kwargs['df_metadata']}, based on the following instructions:

                            1. {kwargs['Updated_prompt']} 
                            2. Our dataframe “df” only contains the following columns: {kwargs['dataframe'].columns}. For any additional field or value required in plot, compute it accurately after understanding the requiremnet thoroughly.
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
        
        messages=[
            {
                "role":"system",
                "content":"You update plotly code basis on the instructions, old plotly code and dataframe"
            },
            {
                "role":"user",
                "content":prompt
            }
            ]
        
        return messages
    
    def prompt_generate_plotly_code(self,**kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        prompt = [
            {
                "role": "system",
                "content": kwargs['system_msg']
            },
            {
                "role": "user",
                "content": f"""Can you generate {kwargs['plot_suggestion']} Python plotly code to chart the results of the dataframe? 
                                Assume the data is in a pandas dataframe called 'df'. 
                                Instruction:

                                1. If there is only **one value** in the dataframe, follow these rules:
                                        - If the value represents a **percentage**, display it as a **KPI Card** with:
                                            - The title indicating the metric.
                                            - The value formatted with `%` as a **suffix** (e.g., `85%`).
                                            - The formatted value should be **used as the displayed number** with the suffix applied separately.
                                        - If the value represents a **monetary amount**, display it as a **KPI Card** with:
                                            - The title indicating the metric.
                                            - The value formatted correctly with a **`$` prefix** as follows, make the changes in the orignal value itself **don't create another variable for it**:
                                                - If `< 1K` → Convert value normally and use **prefix: `$`** and **suffix: `""`** and **valueformat = "d" if it's an integer, otherwise ".2f"**.
                                                - If `1K - 999K` → Convert the value to **X.XXK** and use **prefix: `$`**, **suffix: `K`** and **valueformat = ".2f"**..
                                                - If `≥ 1M` → Convert the value to **X.XXM** and use **prefix: `$`**, **suffix: `M`** and **valueformat = ".2f"**..
                                            - The value should be used in `go.Indicator(value=value)`, with `prefix` and `suffix` applied separately.
                                        - For all other **Single numerical values** (counts, quantities, etc.), display the number as is, without a currency prefix ($). Apply the following format:
                                            - The value should be formatted correctly and make the changes in the **orignal value** itself **don't create another variable for it**:
                                                - If `< 10K` → Convert value normally and use **suffix: `""`** and **valueformat = "d" if it's an integer, otherwise ".2f"****.
                                                - If `10K - 999K` → Convert the value to **X.XXK** and use  **suffix: `K`** and **valueformat = ".2f"**.
                                                - If `≥ 1M` → Convert the value to **X.XXM** and use **suffix: `M`** and **valueformat = ".2f"**.
                                            - The value should be used in `go.Indicator(value=value)`, with `suffix` applied separately. 
                                        - Title Formatting:
                                            - Do NOT use "align" inside title.font to prevent errors.
                                        - **Donot** use the **width** argument in fig.update_layout().
                                        - **Donot** show title again in **update_layout** if it is shown in **go.Figure**.
                                        - Use fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20)) for proper spacing, **Dont restrict this height for other Charts with multiple values**.
                                                

                                    2. For PIE CHARTS, DONUT CHARTS, and WATERFALL CHARTS, if there are multiple values below 1, combine them into a single category named 'Others' before plotting.                                
                                    3. Do not use append function in dataframe instead use concat. for an example:
                                        Instead of "df = df.append('Profit_Center_Desc': 'Others', 'PercentageContribution': 'others_sum', ignore_index=True)"
                                        Use "pd.concat([df, pd.DataFrame('Profit_Center_Desc': ['Others'], 'PercentageContribution': ['others_sum'])], ignore_index=True)
                                    4. Include a title that summarizes the main insight from the data, make it bold, and left-justify it.
                                    5. If the chart has axes, make the axis titles bold. (For pie chart do not do.)
                                    6. Ensure **data labels are displayed** on all charts to enhance readability.
                                        - **Bar & Column Charts:** Display labels above the bars using `text` with `textposition='outside'`.
                                        - **Line Charts:** Show labels at each data point using `mode='lines+markers+text'` with `textposition='top center'`.
                                        - **Pie & Donut Charts Formatting:**
                                            - Ensure labels are visible on each slice by using `textinfo='label+value'` instead of `label+percent`.  
                                            - Use `texttemplate='% label: value:.2f'` to correctly display the actual percentage (e.g., `58.74` instead of `0.58%%`).  
                                            - **Do not use `percent:.2f`**, as it scales percentages incorrectly.
                                        - **Waterfall Charts:** Display labels above each bar using `textposition='outside'`.
                                        - **Scatter Plots:** Add labels next to each marker using `mode='markers+text'` with `textposition='top center'`.
                                        - **Other Chart Types:** If applicable, ensure data labels are present and well-positioned for clarity.
                                    7. **Data Labels** should be rounded off to **2 decimals** and include the '%' sign for percentages.
                                    8. When there are **more than 2 columns** and chart is **Bar Chart** then create **Grouped Bar Chart**. 
                                    9. **Color Refinement:**
                                        - **Use the following color scheme for consistency:**
                                            - Mars Blue: `#0000A0`
                                            - Mars Green: `#00D7B9`
                                            - Mars Yellow: `#FFDC00`
                                            - Sky Blue: `#00DCFA`
                                            - Deep Purple: `#9600FF`
                                            - Zesty Orange: `#FF8200`
                                            - Fresh Green: `#A6DB00`
                                            - Electric Pink: `#FF329F`
                                            - Grey (Other): `#EBEBEB`
                                            - Scarlet Red (Bad/-ve): `#FF2400`
                                        - **Categorical Plots:** Assign unique colors to each category.
                                        - **Pie/Donut Charts:** Ensure distinct colors; do **not** default to grey unless representing "Other".
                                        - **Negative Values:** Use Scarlet Red (`#FF2400`).
                                        - **KPI Cards:** Keep background color unchanged but apply **Mars Blue** color to label and **#0808bd color** to number.
                                        - **Grouped Bar Charts:** Ensure bars in a group have distinct colors (Mars blue, orange, red, green, purple). **Do not use grey** for any bar.
                                    10. Respond with only Python code **without any error**. Do not answer with any explanations -- just the code."""
                                    #6. Chart legends position should be bottom of the chart.
            }
        ]

        return prompt
    
    def prompt_follow_up_question_generation(self,**kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        messages = [
            {
                "role": "system", 
                "content": kwargs['system_prompt']
            },
            {
                "role": "user", 
                "content": "Give me a rephrased question based on the previous question and answers so that a TEXT TO SQL ENGINE CAN EXTRACT it. JUST GIVE ME THE QUESTION."
            }
        ]

        return messages

    def prompt_summary_clarity_score(self,**kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        system_prompt = """
        You are a helpful assistant highly skilled in checking a summary on verious guardrails by providing a score from 1 (POOR) - 10 (VERY GOOD) while providing clear rationale. Specifically, you can carefully evaluate the summaries across guardrails.
        GUARDRAILS: 
        1. Understand the asked question and the provided dataset carefully before summarizing.
        2. Do not merely list all the items in the dataframe as a summary.
        3. Format 'YearPeriod' as "Year Period". Example: 'YearPeriod' 2023001 should be shown as "Year 2023 Period 001".
        4. Display all numbers greater than 6 digits as millions.
        5. Display all numbers greater than 9 digits as billions.
        6. Limit all fractional numbers to a maximum of 2 decimal places.

        Your OUTPUT MUST BE A VALID JSON LIST OF OBJECTS in the format:

        ```[
        { "dimension":  "summary",  "score": x , "rationale": " .."}, 
        ]
        ```
            """
        
        
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant",
                    "content": f"The user query is: \n\n {kwargs['user_query']} and the generated summary is: \n\n ```{kwargs['summary']}``` \n\n. Now, compare the summary based on the guardrails above. \n.The score you assign must be meaningful and backed by clear rationale. A score of 1 means the summary is very different and a score of 10 means it is very similar. The structured evaluation is below:"},
                ]
        
        return messages
    
    def prompt_generate_summary_plotly(self, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        system_prompt="""
        You are an helpful agent that takes plotly figure json object and summarizes the data contained in it.
        1-GIVE SUMMARY on the DATA only.
        2-SUMMARY should be one paragraph only.
        3-Just give the data, do not provide any calculation
    """
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant",
                    "content": f"Generate an summary of the figure {kwargs['plotly_code']}. The user query is \n\n {kwargs['user_query']}"},
                ]
        
        return messages
    
    def prompt_compare_summaries(self, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        system_prompt = """
        You are a helpful assistant highly skilled in comparing the the two summaries given a by providing a score from 1 (different) - 10 (Same) while providing clear rationale. Specifically, you can carefully evaluate the summaries across the following dimensions
        - Data (data):  Is the facts and insights in both summaries are same.

        You must provide a score for each of the above dimensions. 

        Your OUTPUT MUST BE A VALID JSON LIST OF OBJECTS in the format:

        ```[
        { "dimension":  "data",  "score": x , "rationale": " .."}, 
        ]
        ```
        """
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant",
                    "content": f"Compare the summary {kwargs['summary']} with {kwargs['plot_summary']} . The user query is \n\n {kwargs['user_query']} and the visualization code is \n\n {kwargs['code']} \n\n. Now, compare the summaries based on the 1 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS different AND A SCORE OF 10 IS VERY similar. The structured evaluation is below ."},
                ]
        
        return messages
    
    def prompt_evaluate_visual_quality(self, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        system_prompt = f"""
                    You are a helpful assistant highly skilled in evaluating the quality of a given visualization code by providing a score from 1 (POOR) - 10 (VERY GOOD) while providing clear rationale. YOU MUST CONSIDER VISUALIZATION BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the code across the following dimensions:

                    Dimension 1:
                    - Visualization Compliance: Does the visualization meet the expectations as per the guardrails?
                        Provide a score from 1 (POOR) - 10 (VERY GOOD) based on the GUARDRAILS:
                        1. Should {kwargs['plot_suggestion']}.
                        2. If there is only one value in the dataframe, use an Indicator.
                        3. For PIE CHARTS, DONUT CHARTS, and WATERFALL CHARTS, if there are multiple values below 1, combine them into a single category named 'Others' before plotting.
                        4. Do not use the append function in the dataframe; instead, use concat. For example:
                            Instead of df = df.append({{'Profit_Center_Desc': 'Others', 'PercentageContribution': 'others_sum'}}, ignore_index=True),
                            Use pd.concat([df, pd.DataFrame({{'Profit_Center_Desc': ['Others'], 'PercentageContribution': ['others_sum']}})], ignore_index=True).
                        5. Include a title that summarizes the main insight from the data, make it bold, and left-justify it.
                        6. If the chart has axes, make the axis titles bold. (For pie charts, do not do this.)

                    Dimension 2:
                    - Aesthetics: Does the visualization effectively use design principles to enhance readability and interpretability?
                        Provide a score from 1 (POOR) - 10 (VERY GOOD) based on the AESTHETICS GUARDRAILS:
                        1. Use consistent and appropriate color schemes that enhance clarity and are accessible to color-blind users.
                        2. Ensure that all text, including labels, titles, and legends, is readable and appropriately sized.
                        3. Maintain a balanced layout with appropriate use of white space to avoid clutter.
                        4. Use grid lines sparingly and ensure they do not overpower the data being presented.
                        5. Ensure that the legend, if present, does not overlap with the chart elements and is placed in a clear, logical position.
                        6. Avoid using 3D effects that can distort data interpretation.
                        7. Ensure that the visualization elements (e.g., bars, lines, pie slices) are proportionate and accurately represent the data values.
                        8. Use annotations and data labels where necessary to highlight key insights without overcrowding the chart.

                    You must provide a score for each of the above dimensions.  Assume that data in chart = plot(data) contains a valid dataframe for the dataset. The `plot` function returns a chart (e.g., matplotlib, seaborn etc object).
 
                    Your OUTPUT MUST BE A VALID JSON LIST OF OBJECTS in the format:
                
                    ```[
                    {{ "dimension":  "visualization compliance",  "score": x , "rationale": " .."}}, {{ "dimension":  "aesthetics",  "score": x, "rationale": " .."}}
                    ]
                    ```
                    """
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant",
                    "content": f"Generate an evaluation given the goal and code below in {kwargs['library']}. The specified goal is \n\n {kwargs['user_query_ins']}, suggested plot is {kwargs['plot_suggestion']} \n\n and the visualization code is \n\n {kwargs['code']} \n\n. Now, evaluate the code based on the 2 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS POOR AND A SCORE OF 10 IS VERY GOOD. The structured evaluation is below ."},
                ]
        
        
        return messages
    
    def prompt_forecast(self, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        system_prompt = """
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

        messages = [
                        {"role": "system", 
                         "content": system_prompt},
                        {"role": "user",
                        "content": f"Generate a python code and build a model using arima on dataframe {kwargs['df']}. The user query is \n\n {kwargs['user_query']}. \n\n consider the data already in df as dataframe, dont create dataframe"},
                    ]
        
        return messages
    
    def prompt_create_summary(self, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        query = kwargs['query']
        df = kwargs['df']

        prompt = f"""
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
        return prompt
    
    def prompt_generate_summary(self, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        messages=[{"role":"system","content":"write a summary of dataframe given, do not give any table, summarize all information in paragraph."},
            {"role":"user","content":kwargs['prompt']}
            ]
        return messages
    
    def prompt_forecast_func(self, **kwargs):
        system_prompt = """
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

        messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                        "content": f"Generate a python code and build a model using arima on dataframe {kwargs['df']}. The user query is \n\n {kwargs['user_query']}. \n\n consider the data already in df as dataframe, dont create dataframe"},
                    ]

        return messages

