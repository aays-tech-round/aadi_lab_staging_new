# import sqlite3
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import argparse
import shlex
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.utils.aadi_call import *
import pandas as pd
import sqlparse
import uuid
from src.utils.mindmap import *
import concurrent.futures
from openai import AzureOpenAI
from src.utils.datastore import *
from src.utils.confidance_score import get_score
#Forecast Module
import src.utils.forecast as fc
from src.consts.project_prompts import Prompts
from src.consts.project_sql import SQLQUERY
from src.consts.project_constants import OPENAI_CONFIG,GEN_SQL

from pandas import DataFrame

app = Flask(__name__)
CORS(app)

# Instantiationg the Config Objects
PromptsObj = Prompts()
SQLQueryObj = SQLQUERY()
OpenAIConfigObj = OPENAI_CONFIG()

abc = aadi_genai()


# Creating a Client which is linked to Embedding Model
EmbedClient = AzureOpenAI(
        api_key=OpenAIConfigObj.OPENAI_EMBED_API_KEY,
        api_version=OpenAIConfigObj.OPENAI_EMBED_MODEL_API_VERSION,
        azure_endpoint=OpenAIConfigObj.OPENAI_EMBED_AZURE_ENDPOINT
    )

# Creating a Client which is linked to LLM Model
LLMClient = AzureOpenAI(
        azure_endpoint = OpenAIConfigObj.OPENAI_LLM_AZURE_ENDPOINT,
        api_key=OpenAIConfigObj.OPENAI_LLM_API_KEY,
        api_version=OpenAIConfigObj.OPENAI_LLM_API_VERSION
    )

def log(message: str, title: str = "Info"):
        print(f"{title}: {message}")

def generate_sql_and_df(query:str,is_mind_map:bool=False)->tuple:
    """


    Args:
        query : User input from front end of AADI app
        is_mind_map : Mindmap Question Defaults to False.

    Returns:
        Plot suggestions and confidence scores and SQL features
    """
    plot_suggestion = 'NA'
    # reply=""
    # excep=""
    retry = 2
    if is_mind_map == True:
        header = query[1]
        query = query[0]
    else:
        header = ''
    main_query = query
    while retry > 0:
        log(message="Entering SQL Generation Engine", title="SQL Generation")
        generated_sql = abc.generate_sql(query, allow_llm_to_see_data=True)
        log(message="SQL Generated Successfully", title="SQL Generation")
        if abc.is_sql_valid(generated_sql):
            log(message="SQL Schema Correct", title="SQL Schema Check")
            formatted_code = sqlparse.format(generated_sql, reindent=True)
            try:
                log(message="Entering into database to run generated SQL", title="SQL Syntax Check")
                df = abc.run_sql(generated_sql)

                df = df.apply(
                    lambda col: col.map(
                        lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x
                    )
                )
                # To Set a Threshhold
                threshhold_fallback=""
                if len(df) > 50:
                    log("Threshhold hit")
                    threshhold_fallback = (
                            """As the dataset is too large to be fully included in the response, 
                            we have limited the output to the top 50 rows to maintain readability 
                            and efficiency.
                            """
                        )
                    df = df.iloc[:50]
                log(message="SQL Syntactically Correct", title="SQL Syntax Check")
                #Changing datatypes
                df.fillna(0,inplace=True)
                df = abc.convert_yearperiod_to_string(df)
                if len(df) > 0:

                    questions_df = abc.get_training_data()
                    existing_ques = questions_df[questions_df['training_data_type'] == 'sql']['question']
                    log(message="Entering Summary Generation Engine", title="Summary Check")
                    summary = abc.generate_summary(
                        LLMClient,
                        OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME,
                        query,
                        df,
                        threshhold_fallback,
                    )
                    log(message="Summary Generation Successfull", title="Summary Check")
                    if abc.should_generate_chart(df):
                        log(message="Entering Visualization Generation Engine", title="Visualization Check")
                        fig_code, plot_suggestion = abc.generate_plotly_code(LLMClient=LLMClient, 
                                                                        LLMModelName=OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME, 
                                                                        question=query, 
                                                                        sql=generated_sql, 
                                                                        df=df)
                        fig = abc.get_plotly_figure(plotly_code=fig_code, df=df, dark_mode=False)

                        log(message="Visualization Generation Successfull", title="Visualization Check")

                        log(message="Entering Score Generation Engine", title="Score Generation Check")

                        score, viz_extracted_scores, sql_scores = get_score(EmbedClient, 
                                                                        PromptsObj, 
                                                                        LLMClient, 
                                                                        OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME, 
                                                                        query, 
                                                                        formatted_code, 
                                                                        existing_ques, 
                                                                        fig_code, 
                                                                        plot_suggestion, 
                                                                        summary)

                        log(message="Score Generation Successfull", title="Score Generation Check")
                        status = True
                        df = df.to_dict()
                        fig = fig.to_json()
                        reply = "NA"
                        exception = "NA"
                        fig_python = fig_code
                        con_score = score

                    else:
                        log(message="Entering Score Generation Engine except Visualization", title="Score Generation Check")
                        score, viz_extracted_scores, sql_scores = get_score(EmbedClient, 
                                                                        PromptsObj,
                                                                        LLMClient,
                                                                        OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME, 
                                                                        query, 
                                                                        formatted_code, 
                                                                        existing_ques)
                        
                        log(message="Score Generation Successfull", title="Score Generation Check")
                        status = True
                        df = df.to_dict()
                        fig = {}
                        reply = "NA"
                        exception = "NA"
                        fig_python = 'NA'
                        con_score = score
                    return generated_sql, formatted_code, df, summary, fig, status, reply, exception, fig_python, header, con_score, main_query, plot_suggestion, viz_extracted_scores, sql_scores
                else:

                    log(message="Generated Dataset is Empty", title="Empty DF(Retry)")
                    reply = "No records were found for this query."
                    query =f"\nGiven the previous sql: ```{generated_sql}```, please check all the details and create a new sql to answer the query ```{main_query}```"
                    status = False
                    summary = "NA"
                    df = {}
                    fig = {}
                    fig_python = 'NA'
                    exception = "NA"
                    con_score = "NA"
                    viz_extracted_scores = 0
                    sql_scores = 0
            except Exception as e:
                
                log(message=f"Error Occered: {e}", title="Except Block(Retry)")
                if hasattr(e, 'response'):
                    if e.response.status_code in [400, 429]:
                        reply = "There might be an issue with the llm in generating the answer. Please try again later."
                    elif e.response.status_code == 503:
                        reply = "The service is temporarily unable to process your request. Please try again later."
                    else:
                        reply = "I couldn't retrieve data from the database. Please re-ask the question."
                else:
                    reply = "I couldn't retrieve data from the database. Please try rephrasing your question or ask a different question."
                
                exception = str(e)
                query =f"\nGiven the previous sql: ```{generated_sql}```, failed to execute with error code : ```{exception}```, please generate a new sql to answer the query ```{main_query}```. Make sure the sql code is syntactically correct."
                status = False
                summary = "NA"
                df = {}
                fig = {}
                fig_python = "NA"
                con_score = "NA"
                viz_extracted_scores = 0
                sql_scores = 0
        else:
                log(message=f"SQL Contains Natural Language: {generated_sql}", title="SQL Schema missing (Retry)")
                reply = generated_sql
                formatted_code = {}
                df = {}
                summary = "NA"
                fig = {}
                fig_python = "NA"
                status = False
                exception = "NA"
                con_score = "NA"
                viz_extracted_scores = 0
                sql_scores = 0

        log(message=f"RETRY ENGINE -> {retry}", title="Retry Count")
        log(message=f"RETRY QUERY -> {query}", title="Retry Query")
        retry-=1
    generate_sql_and_df_object=GEN_SQL(generated_sql, formatted_code, df, summary, fig, status, reply, exception, fig_python, header, con_score, query, plot_suggestion, viz_extracted_scores, sql_scores)
    generate_sql_and_df_tuple=tuple(vars(generate_sql_and_df_object).values())
    #return sql, formatted_code, df, summary, fig, status, reply, exception, fig_python, header, con_score, query, plot_suggestion, viz_extracted_scores, sql_scores
 
    return generate_sql_and_df_tuple
#-----------------------------FORECAST-DATA-GENERATIOn-------------------------------------------------------#
def if_forecast(df:DataFrame, query:str):
    """
    Args:
        df : Generate summary for forecasted dataframe
        query : User input from front end of AADI app

    Returns:
       Plot suggestions and confidence scores and SQL features
    """  
    header='' 
    if len(df) > 0:
        if 'YearPeriod' in df.columns:
            df['YearPeriod'] = df['YearPeriod'].astype(str)
        if 'FY' in df.columns:
            df['FY'] = df['FY'].astype(str)
        log(message="Entering Summary Generation Engine", title="Summary Check")
        summary=abc.generate_summary(LLMClient, OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME, query, df)
        log(message="Summary Generation Successful", title="Summary Check")
        if abc.should_generate_chart(df):
            # query += '.Keep YearPeriod in x axis, predicted Values column has the predicted data, ignore zeros and ensure the plot shows both actual and predicted values in one line, with different colors for each. For example, use a line plot where actual values are in one color and predicted values are in another color.'
            log(message="Entering Visualization Generation Engine", title="Visualization Check")
            fig_code, plot_suggestion = abc.generate_plotly_code(LLMClient = LLMClient,
                                                                LLMModelName = OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME, 
                                                                question=query, 
                                                                df=df)
            fig = abc.get_plotly_figure(plotly_code=fig_code, df=df, dark_mode=False)

            log(message="Visualization Generation Successful", title="Visualization Check")

            status = True
            df = df.to_dict()
            fig = fig.to_json()
            reply = "NA"
            exception = "NA"
            fig_python = fig_code
            con_score = 'NA'
            viz_extracted_scores = 0
            sql_scores = 0

        else:
            log(message=f"The dataset is not meant to be visualized.{df}", title="No Visualization Block")
            status = True
            df = df.to_dict()
            fig = {}
            reply = "NA"
            exception = "NA"
            fig_python = 'NA'
            con_score = 'NA'
            viz_extracted_scores = 0
            sql_scores = 0
            plot_suggestion = 'NA'
    else:
        log(message="Generated Dataset is Empty", title="Empty DF")
        reply = "No records were found for this query."
        status = False
        summary = "NA"
        df = {}
        fig = {}
        fig_python = 'NA'
        exception = "NA"
        con_score = "NA"
        viz_extracted_scores = 0
        sql_scores = 0
        plot_suggestion = 'NA'
    
    return df, summary, fig, status, reply, exception, fig_python, con_score, query, plot_suggestion, viz_extracted_scores, sql_scores, header

@app.route('/ask', methods=['POST'])
def ask():
    """ This is the ask API , which is used for generating confidence and visualisation scores

    Returns:
       Json response parsed in front end
    """

    message_id = str(uuid.uuid4())
    data = request.get_json()
    prompt = data['query']
    uncache_llmeval = data.get('llmeval_flag',False)
    response = {"response":{
                            'query':prompt,
                            'message_id':message_id,
                            'summary_of_summaries': '',
                            'content': [],
                            }}
    cache_response = []
    
    # Checking for Cache only if fresh query from UI is generated and not from the ask_llmeval api
    if uncache_llmeval == False:
        cache_response = get_the_cached_response(SQLQueryObj, prompt.lower())
        iter = 1
    if len(cache_response) > 0 or cache_response==None:
        log(message="Query Cache Hit: Cache Query Found", title="Checking for Cached Queries")
        for row in cache_response:
            cached_from_response_id = row[0]
            child_question_id = row[1]
            summary = row[2]
            fig = json.loads(row[3])
            formatted_code = row[4]
            df = json.loads(row[5])
            exception = row[6]
            version = row[7]
            fig_python = row[11]
            query = row[12]
            child_query = row[13]
            header = row[14]
            con_score = row[15]
            all_score = row[16]
            summ_of_summ = row[17]
            status = row[18]
            reply = row[19]
            response['response']['content'].append({
                    'child_question_id':child_question_id,
                    'sql': formatted_code,
                    'summary': summary,
                    'header': header,
                    'dataframe': df,
                    'figure': fig,
                    'python_code_figure':fig_python,
                    'message':reply,
                    'error-statement':exception,
                    'status':status,
                    'confidence_score':con_score,
                    'all_score':json.loads(all_score),
                    'from_cache': True
                })
            store_response(SQLQueryObj,message_id, child_question_id, summary, json.dumps(fig), formatted_code, json.dumps(df), exception,fig_python,prompt, child_query, header, con_score, all_score, status, reply, summ_of_summ,version=version, cached_from_response_id=cached_from_response_id)
            log(message="Cached results stored in database successfully", title="Cached Data Storing")
        response['response']['summary_of_summaries'] = summ_of_summ
        return jsonify(response)
    elif len(cache_response)<1 or uncache_llmeval==True:
        log(message="Query Cache Miss: Cache Query Not Found. Redirecting to LLMs", title="Checking for Cached Queries")
        try:
            analysis_type = extract_analysis(prompt)['Analysis Type']
        except:
            analysis_type = 'aadi_main'
        ## Redirecting to Mind Map
        if (analysis_type == "trend analysis") | (analysis_type == "decomposition analysis") | (
                analysis_type == "variance analysis"):
            
            log(message="Entering Sub-Question Generation Engine", title="Sub-question Generation")
            queries = sub_queries(prompt)
            log(message="Sub-Question Generation Successful", title="Sub-question Generation")
            # For Summary of Summary
            multi_summary = []
            with concurrent.futures.ThreadPoolExecutor() as executor:

                log(message="Entering 'generate_sql_and_df' to generate answer", title="API - ANSWER Generation - Mindmap")
                generate_sql_and_run_fobjects = [executor.submit(generate_sql_and_df, query, True) for query in queries]
                answer_list = []
                for id,obj in enumerate(generate_sql_and_run_fobjects):
                    child_question_id = id+1
                    sql, formatted_code, df, summary, fig, status, reply, exception, fig_python, header, con_score, query, plot_suggesion, viz_extracted_scores, sql_scores = obj.result()
                    multi_summary.append(summary)
                    response['response']['content'].append({
                        'child_question_id':child_question_id,
                        'sql': formatted_code,
                        'summary': summary,
                        'header': header,
                        'dataframe': df,
                        'figure': fig,
                        'python_code_figure':fig_python,
                        'message':reply,
                        'error-statement':exception,
                        'status':status,
                        'confidence_score':con_score,
                        'all_score':{
                            'visualization_score': viz_extracted_scores,
                            'Summary_scores':sql_scores
                        },
                        'from_cache': False
                    }) 
                    log(message="Answer Generation successful", title="API - ANSWER Generation - Mindmap")
                    all_score = {
                                'Summary_scores':sql_scores,
                                'visualization_score': viz_extracted_scores
                            }
                    answer_list.append([message_id, child_question_id, summary, fig, formatted_code, df, exception, fig_python, prompt, query, header, con_score, all_score, status, reply])
                summ_of_summ = abc.summary_of_summaries(LLMClient,
                                                    OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME,
                                                    multi_summary, 
                                                    prompt)
                log(message="Storing Generated answer to SQL Database", title="Data Storeing - MindMap")

                for items in answer_list:
                    message_id, child_question_id, summary, fig, formatted_code, df, exception, fig_python, prompt, query, header, con_score, all_score, status, reply = items
                    store_response(SQLQueryObj,message_id, child_question_id, summary, json.dumps(fig), formatted_code, json.dumps(df), exception, fig_python, prompt, query, header, con_score, json.dumps(all_score), status, reply, summ_of_summ)
                log(message="Answer storing is successful", title="Data Storeing - Mindmap")
                response['response']['summary_of_summaries'] = summ_of_summ

            return jsonify(response)

        ## Redirecting to AADI Main App
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:

                log(message="Entering 'generate_sql_and_df' to generate answer", title="API - ANSWER Generation")
                generate_sql_and_run_fobjects = executor.submit(generate_sql_and_df, prompt, False)

                sql, formatted_code, df, summary, fig, status, reply, exception, fig_python, header, con_score, query, plot_suggesion, viz_extracted_scores, sql_scores = generate_sql_and_run_fobjects.result()
                child_question_id = 1

                response['response']['content'].append({
                        'child_question_id':child_question_id,
                        'sql': formatted_code,
                        'summary': summary,
                        'header': header,
                        'dataframe': df,
                        'figure': fig,
                        'python_code_figure':fig_python,
                        'message':reply,
                        'error-statement':exception,
                        'status':status,
                        'confidence_score':con_score,
                        'all_score':{
                            'visualization_score': viz_extracted_scores,
                            'sql_scores':sql_scores
                        },
                        'from_cache': False
                    })
                all_score = {
                                'sql_scores':sql_scores,
                                'visualization_score': viz_extracted_scores
                            }
                log(message="Answer Generation successful", title="API - ANSWER Generation - IND")
                if uncache_llmeval != True:
                    log(message="Storing Generated answer to SQL Database", title="Data Storeing - IND")
                    store_response(SQLQueryObj,message_id, child_question_id, summary, json.dumps(fig), formatted_code, json.dumps(df), exception, fig_python, prompt, query, header, con_score, json.dumps(all_score), status, reply)
                    log(message="Answer storing is successful", title="Data Storeing - IND")
                
                
            return jsonify(response)


@app.route('/update_plot', methods=['POST'])
def update_plot():
    """ This function is used to update Plot if user is not content with the plot shown in front end 

    Returns:
        Json response which gives the figure object
    """
    data = request.get_json()
    Response_id=data['message_id']
    Section_id=data['child_question_id']
    Updated_prompt=data["Change_prompt"]
    version = data["Version"]
    version_history = int(version) + 1

    try:
        log(message="Entering Redraw chart function to generate new chart.", title="Redraw Chart")
        python_code_figure, df = abc.redraw_chart(LLMClient, 
                                                 OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME, 
                                                 Response_id, 
                                                 Section_id, 
                                                 Updated_prompt)
        figure = abc.get_plotly_figure(python_code_figure, df,dark_mode=False)
        log(message="New chart generation successful.", title="Redraw Chart")
        figure = figure.to_json()
          
        log(message="Storing Generated visualization code to db", title="Data Storing - Redraw")
        re_draw_data_store(SQLQueryObj, Response_id, Section_id, python_code_figure, figure, version_history)
        log(message="Generated visualization code is successfully stored", title="Data Storing - Redraw")

        return jsonify({'python_code_figure':python_code_figure, 'figure':figure, 'version':version_history})
    
    except Exception as e:
        print(e)
        # return jsonify({'reply':"Sorry! not able to re-draw chart"}),401    


@app.route("/reset", methods=['POST'])
def reset_plot():
    """Resets the plot to base plot 

    Returns:
        Json response
    """
    try:
        data = request.get_json()
        type = data.get('type')
        response_id = data.get('message_id')
        child_id = data.get('child_id')
        version = data.get('version')

        if type == 'reset' and response_id is not None:
            return reset(SQLQueryObj, response_id, child_id)
        elif type == 'undo' and response_id is not None:
            return undo(SQLQueryObj, response_id, child_id, version)
        else:
            return jsonify({'error': 'Invalid request parameters'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500  


@app.route("/ask_followup", methods=['POST'])
def ask_followup():
    """ This functions is called when a follow up question is prompted in front end by end user which is captured by
    Response_ID and Section_ID

    Returns:
        Plot suggestions and confidence scores and SQL features
    """
    data = request.get_json()
    follow_query = data['query']
    Response_ID = data['Response_ID']
    Section_ID = data['Section_ID']

    log(message="As per Response id this will fetch data", title="Data Extraction for Previous Question")
    summary, sql, df, main_question, child_question, header = get_data_followup(SQLQueryObj, Response_ID, Section_ID)
    log(message="Data Fetched Successfully.", title="Data Extraction for Previous Question")
    log(message=f"Raw question will pass to generate new question, Question: {follow_query}", title="Followup Question Generation Engine")
    follow_up_prompt = abc.follow_up_question_generation(LLMClient, 
                                                        OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME, 
                                                        child_question, 
                                                        follow_query, 
                                                        sql, 
                                                        summary)
    log(message=f"New Question generation Successfull. Question: {follow_up_prompt}", title="Followup Question Generation Engine")
    if fc.forecast_extract_query(follow_up_prompt):
        log(message="This question is a forecast question", title="Forecest-Engine")
        message_id = str(uuid.uuid4())
        response = {"response":{
                            'query':follow_up_prompt,
                            'message_id':message_id,
                            'summary_of_summaries': '',
                            'content': [],
                            }}
        
        df_to_forecast = pd.read_json(df)
        log(message="Passing Previous question's data to generate forecasted dataframe", title="Forecest-Engine")
        forecast_df, forecast_code = fc.forecast_func(df_to_forecast, 
                                                      follow_up_prompt, 
                                                      PromptsObj, 
                                                      LLMClient,OpenAIConfigObj.OPENAI_LLM_DEPLOYMENT_NAME)
        log(message="Forecast data generation successful.", title="Forecest-Engine")                                              
        # result = pd.concat([df_to_forecast.reset_index(drop=True), forecast_df], ignore_index=True
        # result.fillna(0, inplace=True)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            log(message="Answer Creation using 'if_forecast' function.", title="Answer Creation-Forecest-Engine")
            generate_forecast = executor.submit(if_forecast, forecast_df, follow_up_prompt)

            df, summary, fig, status, reply, exception, fig_python, con_score, query, plot_suggestion, viz_extracted_scores, sql_scores, header = generate_forecast.result()
            child_question_id = 1
            log(message="Answer Creation using 'if_forecast' function is successful.", title="Answer Creation-Forecest-Engine")
            formatted_code = 'NA'
            response['response']['content'].append({
                    'child_question_id':child_question_id,
                    'sql': formatted_code,
                    'summary': summary,
                    'header': "NA",
                    'dataframe': df,
                    'figure': fig,
                    'python_code_figure':fig_python,
                    'message':reply,
                    'error-statement':exception,
                    'status':status,
                    'confidence_score':con_score,
                    'all_score':{
                            'visualization_score': viz_extracted_scores,
                            'sql_scores':sql_scores
                        }
                })
            log(message="Storing Generated answer to SQL Database", title="Data Storeing - Forecast")
            all_score = {
                            'sql_scores':sql_scores,
                            'visualization_score': viz_extracted_scores
                        }
            all_score = json.dumps(all_score)
            store_response(SQLQueryObj, message_id, child_question_id,summary, json.dumps(fig), formatted_code, json.dumps(df), exception, fig_python, child_question, follow_query,header, con_score,all_score, status, reply)
            log(message="Answer storing is successful", title="Data Storeing - Forecast")

        return jsonify(response)
    
    else:
        ask_payload = {
            'query': follow_up_prompt
        }
        log(message="Passing Followup Question to ask Api to generate answer", title="Followup - Ask API")
        with app.test_request_context('/ask', method='POST', json=ask_payload):
            ask_response = ask()
            ask_response_json = ask_response.get_json()  # Get the JSON data from the response
            log(message="ask Api to answer generation is successful.", title="Followup - Ask API")

            log(message="Manipulating Response Id.", title="Followup - Ask API")
            old_msg_id = ask_response_json['response']['message_id']
            ask_response_json['response']['message_id'] += "-followup"  # Modify the message_id
            new_msg_id = ask_response_json['response']['message_id']
            response = jsonify(ask_response_json)
            update_table(SQLQueryObj,old_response_id=old_msg_id, new_response_id=new_msg_id)
            log(message="Manipulating Response Id Successful.", title="Followup - Ask API")

            return response
        
@app.route('/ask_llmeval', methods=['POST'])
def ask_llmeval():
    """ This is the ask API to uncache and re-evaluate a query using LLM

    Returns:
       Json response parsed in front end
    """

    # message_id = str(uuid.uuid4())
    data = request.get_json()
    response_id = data['Response_ID']
    section_id = data['Section_ID']
    summary, sql, df, main_question, child_question, header_old = get_data_followup(SQLQueryObj,response_id, section_id)    
    ask_payload = {
            'query': child_question,
            'llmeval_flag': True
    }
    log(message="Passing Question to ask Api to uncache and re-evaluate using LLM", title="Uncache - Ask API")
    with app.test_request_context('/ask', method='POST', json=ask_payload):
        ask_response = ask()
        ask_response_json = ask_response.get_json()  # Get the JSON data from the response
        log(message="Response Generated for LLM Evauluation of Uncache query", title="Uncache: Response Generated")
        new_response_id = f'{response_id}_{section_id}_llmeval'
        ask_response_json['response']['message_id'] = new_response_id
        ask_response_json['response']['content'][0]['header'] = header_old
        response = jsonify(ask_response_json)
        response_content = ask_response_json['response']['content'][0]
        summary = response_content['summary']
        fig = response_content['figure']
        formatted_code = response_content['sql']
        df = response_content['dataframe']
        exception = response_content['error-statement']
        fig_python = response_content['python_code_figure']
        prompt = main_question  
        query = child_question
        header = header_old
        con_score = response_content['confidence_score']
        all_score = response_content['all_score']
        status = response_content['status']
        reply = response_content['message']

        log(message="Writing the Uncache Records in Table", title="Storing in Database")
        store_response(SQLQueryObj,new_response_id, section_id, summary, json.dumps(fig), formatted_code, json.dumps(df), exception, fig_python, prompt, query, header, con_score, json.dumps(all_score), status, reply)
        log(message="Uncaching is successful", title="Uncache-LLM Evaluation Successful")

        return response


    
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        help="The environment the process is being run in. Values ['DEV','PROD']. Default is 'PROD'",
        default="PROD",
        choices=["DEV","PROD"]
    )

    args = parser.parse_args(shlex.split(" ".join(sys.argv[1:])))

    env = args.env
    if env=="PROD":
     
      from src.utils.deployment_libs import * 
      
    
    app.run(debug=False)