import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
import sqlparse
from sqlparse import sql, tokens
import numpy as np
import requests
import json
import pandas as pd
from src.consts.project_prompts import Prompts
from src.consts.project_constants import OPENAI_CONFIG

#ivaopenaiembed
def get_embedding(EmbedClient, text, model="ivaopenaiembed"):
    """This function will help to get the embedding of the provided text

    Args:
        EmbedClient (_type_): This is an Object of Embedding Model which will help to establish the connection
        text (_type_): This will contain the text on which Embeddings need to be performed
        model (str, optional): Name of Deployed Embedding Model Name. Defaults to "ivaopenaiembed".

    Returns:
        _type_: Return the Embeddings which are in form of vector.
    """
    text = text.replace("\n", " ")
    return EmbedClient.embeddings.create(input = [text], model=model).data[0].embedding


def cosinesimilarity(a, b):
    """This function calculated the Cosine Similarity Score

    Args:
        a (_type_): This contains First Feature Vector
        b (_type_): This contains Second Feature Vector

    Returns:
        _type_: Return a numerical value which signifies the cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def preprocess(text):
    """This function do the preprocessing to remove the SQL words from a text object

    Args:
        text (_type_): This contains complete text which needs to be preprocessed to remove the keywords related to SQL Language

    Returns:
        _type_: This contains the text which doesn't contain any SQL keywords
    """
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove SQL keywords
    sql_keywords = ['select', 'from', 'where', 'group', 'by', 'having', 'order', 'join', 'on', 'case', 'left', 'right', 'then', 'like', 'in', 'else', 'if', 'when']
    return ' '.join([word for word in text.split() if word not in sql_keywords])

def jaccard_similarity(set1, set2):
    """This function helps to calculate the Jaccard Similarity Score

    Args:
        set1 (_type_): Contains First Set of Words
        set2 (_type_): Contains Second Set of Words

    Returns:
        _type_: Returns a Jaccard Similarity Score
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def calculate_fuzzysimilarity(question, sql):
    """This function helps to calculate the Fuzzy Similarity Score

    Args:
        question (_type_): This contains the question which was provided in form of the user query
        sql (_type_): Tgis contains the corresponding SQL Query 

    Returns:
        _type_: Returns a combined score
    """
    # Preprocess
    proc_question = preprocess(question)
    proc_sql = preprocess(sql)
    
    # TF-IDF and Cosine Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([proc_question, proc_sql])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    # Jaccard Similarity
    question_set = set(proc_question.split())
    sql_set = set(proc_sql.split())
    jaccard_sim = jaccard_similarity(question_set, sql_set)
    
    # Combine similarities (you can adjust weights as needed)
    combined_sim = (cosine_sim + jaccard_sim) / 2
    
    return combined_sim  # Convert to percentage

def is_syntactically_correct(query):
    """This function will check if the provided SQL query is syntax wise correct or not

    Args:
        query (_type_): This contains an SQL Query

    Returns:
        _type_: Returns True if the SQL Query is Syntatically correct else False
    """
    try:
        parsed = sqlparse.parse(query)
        return len(parsed) > 0 and all([type(stmt) == sql.Statement for stmt in parsed])
    except Exception as e:
        return False
    
    
def preprocess_text(text):
    """This function will return the provided text in lower case.

    Args:
        text (_type_): This contains a text

    Returns:
        _type_: Returns a lowered case of complete text
    """
    # Add any necessary preprocessing steps (lowercasing, removing punctuation, etc.)
    return text.lower()

def calculate_similarity(existing_questions, new_question):
    """This function helps to identify the similarity score among the provided list of existing and new questions.

    Args:
        existing_questions (_type_): This contains a list of all the existing questions
        new_question (_type_): This contains a list of all the new questions which are asked in the form of user query in new session.

    Returns:
        _type_: Return a Similarity Score
    """
    # Preprocess all questions
    all_questions = [preprocess_text(q) for q in existing_questions + [new_question]]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_questions)
    
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    return np.max(similarity_scores[0]) 


def extract_scores(formatted_json_string):
    """This function extracts the scores of different dimension from a JSON string which is generated using an LLM Prompt

    Args:
        formatted_json_string (_type_): This contains a JSON string which consists of scores of different dimensons on which we are evaluting.

    Returns:
        _type_: Returns a dictionary of scores of different dimension
    """
    # Remove the ```json and ``` tags, and strip whitespace
    json_string = formatted_json_string.strip().lstrip('```json').rstrip('```')
  
    # Parse the JSON string
    data = json.loads(json_string)
    
    # Extract scores
    scores = {}
    for item in data:
        dimension = item['dimension']
        score = item['score']
        scores[dimension] = score

    
    return scores
#------------------------------SUMMARY_CLARITY_SCORE-------------------------------------------------------#
def summary_clarity_score(PromptsObj, LLMClient, LLMModelName, user_query, summary):
    """This function helps to evaluate the clarity of LLM generated summary 

    Args:
        PromptsObj (_type_): This is an instantiated Object of Prompts which have different functions related to different prompts
        LLMClient (_type_): This is a connection client which is linked to an LLM Model
        LLMModelName (_type_): This contains the deployed LLM Model Name
        user_query (_type_): This contains a text of user query
        summary (_type_): This contains summary of the datas

    Returns:
        _type_: Returns a clarity score which signifies how concise the summary is with related to user query
    """

    messages = PromptsObj.prompt_summary_clarity_score(user_query = user_query,
                                                      summary = summary)

    response = LLMClient.chat.completions.create(
                    model=LLMModelName,
                    messages=messages,
                    stop=None,
                    temperature=0.1,
                )
    
    clarity_score = extract_scores(response.choices[0].message.content)['summary']
    clarity_score = clarity_score/10
    
    return clarity_score

#----------------------------------------VISUALIZATION-SCORE-------------------------------------------------------------#

def generate_summary_plotly(plotly_code, user_query, PromptsObj, LLMClient, LLMModelName):
    """This functions helps to generate the summary based on the plotly visualization code with the help of an LLM Model

    Args:
        plotly_code (_type_): This will contain an LLM generated plotly code
        user_query (_type_): This contains an user query which conveys what user is asking
        PromptsObj (_type_): This is an instantiated Object of Prompts which have different functions related to different prompts
        LLMClient (_type_): This is a connection client which is linked to an LLM Model
        LLMModelName (_type_): This contains the deployed LLM Model Name

    Returns:
        _type_: Return an LLM generated summary about Visualization
    """

    messages = PromptsObj.prompt_generate_summary_plotly(plotly_code = plotly_code,
                                                        user_query = user_query)

    response = LLMClient.chat.completions.create(
                    model=LLMModelName,
                    messages=messages,
                    stop=None,
                    temperature=0.2,
                )

    viz_summary =(response.choices[0].message.content)
    return viz_summary

def compare_summaries(summary, user_query, code, PromptsObj, LLMClient, LLMModelName):
    """This function will help to compare the summaries and will provide a relevant score with proper explanation as well

    Args:
        summary (_type_): This contains an LLM generated summary
        user_query (_type_): This contains an user query which conveys what user is asking
        code (_type_): This contains a python code generated by LLM Call
        PromptsObj (_type_): This is an instantiated Object of Prompts which have different functions related to different prompts
        LLMClient (_type_): This is a connection client which is linked to an LLM Model
        LLMModelName (_type_): This contains the deployed LLM Model Name

    Returns:
        _type_: Returns an LLM generated response which compares summaries 
    """

    plot_summary = generate_summary_plotly(code, user_query, PromptsObj, LLMClient, LLMModelName)

    messages = PromptsObj.prompt_compare_summaries(code = code,
                                                  user_query = user_query,
                                                  summary = summary,
                                                  plot_summary = plot_summary)

    response = LLMClient.chat.completions.create(
                    model=LLMModelName,
                    messages=messages,
                    stop=None,
                    temperature=0.2,
                )

    comparsion_score=(response.choices[0].message.content)
    extracted_scores = extract_scores(comparsion_score)
    extracted_score = (sum(extracted_scores.values())/len(extracted_scores))

    

    return extracted_score

def visual_cs(library,code,user_query,plot_suggestion,summary, PromptsObj, LLMClient, LLMModelName):
    """This function will help to evaluate the confidence score from the generated visual code perspective.

    Args:
        library (_type_): This contains the name of the library/package
        code (_type_): This contains a code snippet which is LLM generated
        user_query (_type_): This contains an user query which conveys what user is asking
        plot_suggestion (_type_): This contains the name of the suggested plot like Bar chart, Pie chart etc.
        summary (_type_): This contains the summary of the data in human readable form
        PromptsObj (_type_): This is an instantiated Object of Prompts which have different functions related to different prompts
        LLMClient (_type_): This is a connection client which is linked to an LLM Model
        LLMModelName (_type_): This contains the deployed LLM Model Name

    Returns:
        _type_: Returns a score which helps to identify how relevant visual is with the user query and generated summary data
    """
         
    messages = PromptsObj.prompt_evaluate_visual_quality(library = library,
                                                        code = code,
                                                        user_query_ins = user_query,
                                                        plot_suggestion = plot_suggestion
                                                    )
    response = LLMClient.chat.completions.create(
                model=LLMModelName,
                messages=messages,
                stop=None,
                temperature=0.2,
            )
    
    
    
    extracted_scores = extract_scores(response.choices[0].message.content)
    

    # Calculating Visualization Summary Comparison Score
    viz_summary_comp_score = compare_summaries(summary, user_query, code, PromptsObj, LLMClient, LLMModelName)

    # Appending Visualization Summary Comparison Score into dictionary
    extracted_scores['visual_summary_comparison_score'] = viz_summary_comp_score

    # Calculating Avg
    average_score = (sum(extracted_scores.values()) / len(extracted_scores))
    average_score = average_score/10


    extracted_scores = {key : value/10 for key, value in extracted_scores.items()}
    extracted_scores['Total'] = average_score

    return average_score, viz_summary_comp_score, extracted_scores
    
    
    
def get_score(EmbedClient: object, 
              PromptsObj, 
              LLMClient, 
              LLMModelName, 
              user_input: str, 
              generated_sql: str, 
              existing_questions: list, 
              summary: str, 
              plotly_code: str = None, 
              plot_suggestion:str = None, 
              status=True):

    """This function will help to generate properly formatted over score in terms of visualization, sql 

    Returns:
        _type_: Returns a tuple of different relevant scores.
    """
    print("Entering Get Score")
    fuzz_score = calculate_fuzzysimilarity(user_input, generated_sql)

    print("Semantic Score")
    question_embedding = get_embedding(EmbedClient, user_input)
    query_embedding = get_embedding(EmbedClient, generated_sql)
    semantic_score = cosinesimilarity(question_embedding, query_embedding)

    print("Syntax Score")
    syntax_check = is_syntactically_correct(generated_sql)
    syntax_check_score = 1.0 if syntax_check else 0.0

    print("Semantic Check Score")
    if status:
        semantic_check_score = 1.0 
    else:
        semantic_check_score = 0

    print("Fuzzy Alter Score")
    # Combining scores with different weights
    if fuzz_score==0:
        fuzz_score=-0.50

    print("Question Similarity Score")
    question_similarity_score = calculate_similarity(existing_questions, user_input)
    # Summary Clarity Score
    summary_clarity = summary_clarity_score(PromptsObj, LLMClient, LLMModelName, user_input, summary)

    print("Generating Overall SQL Score")
    #Get sql score
    sql_conf_score = (question_similarity_score*0.30+(fuzz_score + semantic_score)*0.20+(syntax_check_score+semantic_check_score)*0.10+summary_clarity*0.40)
    
    final_score_sql = {
        'Intent Translation':semantic_score + fuzz_score,
        'Entity Mapping':(semantic_check_score + syntax_check_score)/2,
        'Code Generation':syntax_check_score,
        'Question Similarity':question_similarity_score,
        'Summary Quality':summary_clarity,
        'Total':sql_conf_score  
    }
    print("All Score calculation")
    #get visual score
    if plotly_code is not None and plot_suggestion is not None:
        print("Vizual Score calculation")
        viz_average_score, viz_summary_comp_score, extracted_scores = visual_cs('plotly', plotly_code, user_input, plot_suggestion, summary, PromptsObj, LLMClient, LLMModelName)
        print("Vizual Score Dict")
        final_score_viz = {
                    'Visualization Compliance Score':extracted_scores['visualization compliance'],
                    'Visualization Aesthetics':extracted_scores['aesthetics'],
                    'Insight Alignment':extracted_scores['visual_summary_comparison_score'],
                    'Total':extracted_scores['Total']
                }
        
        print("Final Conf Score")
        
        confidence_score = (sql_conf_score + viz_average_score)/2
        return confidence_score, final_score_viz, final_score_sql
    
    else:
        extracted_scores=0
        return sql_conf_score, extracted_scores, final_score_sql