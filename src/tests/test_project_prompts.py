import pandas as pd
from src.consts.project_prompts import Prompts
def test_prompt_generate_summary_len_df_ge_two():
    expected_output = [
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

                            The user asked the question: 'THIS IS TEST QUESTION'

                            Below is a pandas DataFrame with the results of the query:
                            |    |   Number |
|---:|---------:|
|  0 |        1 |
|  1 |        2 |
|  2 |        3 |
|  3 |        4 |

                            Your task is to summarize the data in response to the question, incorporating the provided data description: THIS IS TEST DESCRIBE. Ensure that your summary is clear, concise, and follows the style of a professional financial analyst.Use bullet points for important points.
                            """
                        },
                        {
                            "role": "user",
                            "content": "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary."
                        }
                ]
    output = Prompts().prompt_generate_summary_len_df_ge_two(question="THIS IS TEST QUESTION", df=pd.DataFrame(data=[1,2,3,4],columns=['Number']), describe = "THIS IS TEST DESCRIBE")
    assert expected_output == output

def test_prompt_generate_summary_len_df_lt_two():
    expected_output = [
                {
                    "role": "system",
                    "content": f"""
                            You are a financial analyst. You have to adopt the style of a financial analyst. Few example summaries written by financial analysts are below: \n\n
                            1. In the current month, the Gross Sales Value (GSV) trend is showing positive indicators within the set confidence interval, as corroborated by the data from the last two years. Delving into the trend variances, we observe a slight deviation with the GSV declining by a marginal 0.50% from the confidence interval. Further analysis presents a promising picture with the rolling average for GSV standing at 538 million, which marks a significant milestone. On a more granular level, the month-to-month analysis indicates a robust growth of 10.30%, equating to an increase of 50 million from the preceding month. However, a year-to-year comparison showcases a slight dip, where the GSV fell by 8 million, representing a decrease of 1.30% compared to the same period last year.\n
                            2. For the Year-To-Date analysis of Gross Sales Value (GSV), the trend showcases that the value currently stands at 3318 Millions. Delving into the nuances of the trend variances, GSV has exhibited a commendable growth trajectory. The positive variation noted so far this year is an impressive 318 Millions, translating to a robust increase of 10.30%.\n
                            3. The year-on-year difference in Gross Sales Value (GSV) can be chiefly attributed to four major factors as outlined by the accounting profile. Quantity rebates are the most significant contributor, accounting for 60% of the variance. Returns also play a substantial role, making up 20% of the year-on-year change. GSV trading deductions and sales increases each contribute 10%, making them less impactful but still relevant to the overall fluctuation in GSV. Therefore, these four elements collectively explain the year-on-year variation in GSV.\n\n
                            The user asked the question: 'THIS IS TEST QUESTION'\n\nThe following is a pandas DataFrame with the results of the query: \n|    |   Number |
|---:|---------:|
|  0 |        1 |
|  1 |        2 |
|  2 |        3 |
|  3 |        4 |\n summarize the data as per the question.
                            """
                },
                {
                    "role": "user",
                    "content": "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary."
                }
        ]
    output = Prompts().prompt_generate_summary_len_df_lt_two(question="THIS IS TEST QUESTION", df=pd.DataFrame(data=[1,2,3,4],columns=['Number']), describe = "THIS IS TEST DESCRIBE")
    assert expected_output == output

def test_prompt_summary_of_summaries():
    prompt =f"""
            You are an expert Finance Analyst. Summarize the following list of summaries for the question provided.

            - Summaries: ```THIS IS SUMMARY LIST```
            - Question: ```THIS IS TEST QUERY```

            Please include only the information provided in the summary list. Do not add any other information or external context.
            
            follow this instructions:
            1. If the summary is full of numbers, focus on material details and explain key insights.
            2. Mention all major points. 
            3. Do not miss any information provided into the list of summaries.
        
            """
    
    expected_output =[
            {
                "role":"system",
                "content":"write a concise summary of given list of summaries, do not give any table, summarize all information in paragraph."
            },
            {
                "role":"user",
                "content":prompt}
        ]
    output = Prompts().prompt_summary_of_summaries(summary_list="THIS IS SUMMARY LIST", query="THIS IS TEST QUERY")
    assert expected_output == output

def test_prompt_plot_sugguestion():
    expected_output = [
            {
                "role": "system",
                "content": f"""You are an expert in data visualization using Plotly. Your task is to suggest the most suitable Plotly chart for visualizing the results of a query, based on the user's question and the given metadata.

                                    User Question: 'THIS IS TEST QUESTION'

                                    The resulting pandas DataFrame 'df' contains the following data: \n|    |   Number |
|---:|---------:|
|  0 |        1 |
|  1 |        2 |
|  2 |        3 |
|  3 |        4 |
                                    Metadata: THIS IS METADATA OF DATAFRAME

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
    output = Prompts().prompt_plot_sugguestion(question = "THIS IS TEST QUESTION", 
                                               df=pd.DataFrame(data=[1,2,3,4],columns=['Number']),
                                               df_metadata="THIS IS METADATA OF DATAFRAME", 
                                               query="THIS IS TEST QUERY")
    assert expected_output == output

def test_prompt_redraw_chart():
    prompt = f"""Update this plotly code ```THIS IS TEST PLOTLY CODE``` for the data:    Number
0       1
1       2
2       3
3       4 modified to new dataframe as per THIS IS TEST UPDATED PROMPT, Metadata: THIS IS METADATA OF DATAFRAME, based on the following instructions:

                            1. THIS IS TEST UPDATED PROMPT 
                            2. Our dataframe “df” only contains the following columns: Index(['Number'], dtype='object'). For any additional field or value required in plot, compute it accurately after understanding the requiremnet thoroughly.
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
    
    expected_output = [
            {
                "role":"system",
                "content":"You update plotly code basis on the instructions, old plotly code and dataframe"
            },
            {
                "role":"user",
                "content":prompt
            }
        ]
    output = Prompts().prompt_redraw_chart(plotly_code = "THIS IS TEST PLOTLY CODE", 
                                               dataframe=pd.DataFrame(data=[1,2,3,4],columns=['Number']),
                                               Updated_prompt = "THIS IS TEST UPDATED PROMPT",
                                               df_metadata="THIS IS METADATA OF DATAFRAME")
    
    assert expected_output == output

def test_prompt_generate_plotly_code():
    expected_output = [
            {
                "role": "system",
                "content": "THIS IS TEST SYSTEM MESSAGE"
            },
            {
                "role": "user",
                "content": f"""Can you generate THIS IS SUGGESTED PLOT Python plotly code to chart the results of the dataframe? 
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
                                #6. Chart legends position should be bottom of the chart.
            }
    ]
    output = Prompts().prompt_generate_plotly_code(system_msg = "THIS IS TEST SYSTEM MESSAGE", 
                                            plot_suggestion = "THIS IS SUGGESTED PLOT")
    
    assert expected_output == output

def test_prompt_follow_up_question_generation():
    expected_output = [
        {
            "role": "system", 
            "content": "THIS IS TEST SYSTEM PROMPT"
        },
        {
            "role": "user", 
            "content": "Give me a rephrased question based on the previous question and answers so that a TEXT TO SQL ENGINE CAN EXTRACT it. JUST GIVE ME THE QUESTION."
        }
    ]

    output = Prompts().prompt_follow_up_question_generation(system_prompt = "THIS IS TEST SYSTEM PROMPT")

    assert expected_output == output

def test_prompt_summary_clarity_score():
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
    expected_output = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant",
        "content": f"The user query is: \n\n THIS IS TEST USER QUERY and the generated summary is: \n\n ```THIS IS TEST SUMMARY``` \n\n. Now, compare the summary based on the guardrails above. \n.The score you assign must be meaningful and backed by clear rationale. A score of 1 means the summary is very different and a score of 10 means it is very similar. The structured evaluation is below:"},
            ]
    
    output = Prompts().prompt_summary_clarity_score(user_query = "THIS IS TEST USER QUERY",
                                                    summary = "THIS IS TEST SUMMARY")
    
    assert expected_output == output

def test_prompt_generate_summary_plotly():
    system_prompt="""
        You are an helpful agent that takes plotly figure json object and summarizes the data contained in it.
        1-GIVE SUMMARY on the DATA only.
        2-SUMMARY should be one paragraph only.
        3-Just give the data, do not provide any calculation
    """
    expected_output = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant",
                "content": f"Generate an summary of the figure THIS IS TEST PLOTLY CODE. The user query is \n\n THIS IS TEST USER QUERY"},
            ]
    
    output = Prompts().prompt_generate_summary_plotly(plotly_code = "THIS IS TEST PLOTLY CODE",
                                                    user_query = "THIS IS TEST USER QUERY")
    assert expected_output == output

def test_prompt_compare_summaries():
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
    expected_output = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant",
                "content": f"Compare the summary THIS IS TEST SUMMARY with THIS IS TEST PLOT SUMMARY . The user query is \n\n THIS IS TEST USER QUERY and the visualization code is \n\n THIS IS TEST CODE \n\n. Now, compare the summaries based on the 1 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS different AND A SCORE OF 10 IS VERY similar. The structured evaluation is below ."},
            ]
    
    output = Prompts().prompt_compare_summaries(summary = "THIS IS TEST SUMMARY",
                                                    plot_summary = "THIS IS TEST PLOT SUMMARY",
                                                    user_query = "THIS IS TEST USER QUERY",
                                                    code = "THIS IS TEST CODE")
    
    assert expected_output == output

def test_prompt_evaluate_visual_quality():
    system_prompt = f"""
                    You are a helpful assistant highly skilled in evaluating the quality of a given visualization code by providing a score from 1 (POOR) - 10 (VERY GOOD) while providing clear rationale. YOU MUST CONSIDER VISUALIZATION BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the code across the following dimensions:

                    Dimension 1:
                    - Visualization Compliance: Does the visualization meet the expectations as per the guardrails?
                        Provide a score from 1 (POOR) - 10 (VERY GOOD) based on the GUARDRAILS:
                        1. Should THIS IS TEST PLOT SUGGESTION.
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
    expected_output = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant",
                "content": f"Generate an evaluation given the goal and code below in THIS IS TEST LIBRARY. The specified goal is \n\n THIS IS TEST USER QUERY INS, suggested plot is THIS IS TEST PLOT SUGGESTION \n\n and the visualization code is \n\n THIS IS TEST CODE \n\n. Now, evaluate the code based on the 2 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS POOR AND A SCORE OF 10 IS VERY GOOD. The structured evaluation is below ."},
            ]

    output = Prompts().prompt_evaluate_visual_quality(library = "THIS IS TEST LIBRARY",
                                            user_query_ins = "THIS IS TEST USER QUERY INS",
                                            code = "THIS IS TEST CODE",
                                            plot_suggestion = 'THIS IS TEST PLOT SUGGESTION')
    
    assert expected_output == output

def test_prompt_forecast():
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

    expected_output = [
                    {"role": "system", 
                        "content": system_prompt},
                    {"role": "user",
                    "content": f"""Generate a python code and build a model using arima on dataframe    Number
0       1
1       2
2       3
3       4. The user query is \n\n THIS IS TEST USER QUERY. \n\n consider the data already in df as dataframe, dont create dataframe"""},
                ]
    
    output = Prompts().prompt_forecast(df = pd.DataFrame(data=[1,2,3,4],columns=['Number']),
                                        user_query = "THIS IS TEST USER QUERY",
                                        )

    assert expected_output == output

def test_prompt_create_summary():
    expected_output = f"""
            You are an expert Finance Analyst. Summarize the following data for the question: "THIS IS TEST QUERY". Provide the summary in markdown format. Adhere strictly to the given instructions:

            1. Understand the asked question and the provided dataset carefully before summarizing.
            2. Do not merely list all the items in the dataframe as a summary.
            3. Format 'YearPeriod' as "Year Period". Example: 'YearPeriod' 2023001 should be shown as "Year 2023 Period 001".
            4. Display all numbers greater than 6 digits as millions.
            5. Display all numbers greater than 9 digits as billions.
            6. Limit all fractional numbers to a maximum of 2 decimal places.

            Dataset:
            |    |   Number |
|---:|---------:|
|  0 |        1 |
|  1 |        2 |
|  2 |        3 |
|  3 |        4 |
            """
    output = Prompts().prompt_create_summary(df=pd.DataFrame(data=[1,2,3,4],columns=['Number']),
                                            query = "THIS IS TEST QUERY")
    
    assert expected_output == output

def test_prompt_generate_summary():
    expected_output=[{"role":"system","content":"write a summary of dataframe given, do not give any table, summarize all information in paragraph."},
    {"role":"user","content":"THIS IS TEST PROMPT"}
    ]

    output = Prompts().prompt_generate_summary(prompt = "THIS IS TEST PROMPT")

    assert expected_output == output