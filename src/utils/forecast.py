import re

def forecast_extract_query(query):
    """_summary_

    Args:
        query (_type_): _description_

    Returns:
        _type_: _description_
    """
    query = query.lower()

    keywords = ['forecast', 'predict']

    for key in keywords:
        if key in query:
            return True
        else:
            continue
        
    return False

def extract_python_code(text):
    """_summary_

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def forecast_func(df,user_query,PromptsObj,LLMClient, LLMModelName):
    """_summary_

        Args:
            LLMClient (_type_): _description_
            LLMModelName (_type_): _description_
            query (_type_): _description_
            df (_type_): _description_

        Returns:
            _type_: _description_
        """

    messages = PromptsObj.prompt_forecast_func(df = df,
                                               user_query = user_query)

     
    response = LLMClient.chat.completions.create(
                        model=LLMModelName,
                        messages=messages,
                        stop=None,
                        temperature=0.0,
                    )

    code = extract_python_code((response.choices[0].message.content))[0]
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