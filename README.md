# AaDI: Interactive AI for Data Insights

AaDI (AAYS Assistant of Dashboard Interface) is an interactive AI application designed to generate answers from your data using Text-to-SQL and vector store techniques. AaDI not only constructs SQL queries but also executes them to generate dataframes and visualizes the results in an intuitive manner.

## Features

- **Text-to-SQL Generation**: Converts user queries into SQL statements.
- **Data Retrieval**: Executes the generated SQL on your database to fetch the data.
- **Data Visualization**: Creates interactive visualizations from the retrieved data.

## How It Works

1. **User Query**: The user inputs a question.
2. **SQL Generation**: AaDI uses Text-to-SQL techniques to convert the question into an SQL query.
3. **Data Retrieval**: The generated SQL query is executed against the database to retrieve the data.
4. **Data Visualization**: The retrieved data is visualized using interactive charts and graphs.

## Getting Started

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/aays-tech-round/aadi_api.git
    cd aadi_api
    ```

2. Create a Virtual Environment using venv:
    ```bash
    python -m venv aadi_venv
    ```

3. Activate the aadi_venv environment:
    ```bash
    cd aadi_venv/Scripts
    ```
    ```bash
    activate
    ```

4. Install the packages using `requirements.txt` residing in root directory. Make sure you run this after activating your newly created virtual environment
   ```
    pip install -r requirements.txt
    ```

### Configuration of `.env` file
1. Create a new `.env` file in the root directory/Project Directory (i.e aadi_api)
2. Configure the below variables in `.env` file
   1. `SQL_SERVER`: This will contain the value of SQL Server, Database Port in the form `tcp:<FQDN_SQL_SERVER>,<DATABASE_SERVER_PORT>`
   2. `SQL_DATABASE`: This will contain the Database Name
   3. `SQL_USER_ID`: This will contain the SQL User Name
   4. `SQL_PASSWORD`: This will contain the Password of the SQL User.
   5. `OPENAI_LLM_AZURE_ENDPOINT`: This will contain the Endpoint URL of the Open AI LLM Model you want to use.
   6. `OPENAI_LLM_DEPLOYMENT_NAME`: This will contain the deployment name of the deployed Open AI model in Azure
   7. `OPENAI_LLM_API_KEY`: This will contain the API key of Azure Open AI LLM Endpoint
   8. `OPENAI_EMBED_AZURE_ENDPOINT`: This will contain the Endpoint URL of the Open AI Embedding Model you want to use.
   9. `OPENAI_EMBED_API_KEY`: This will contain the API key of Azure Open AI Embed Endpoint
   10. `RESPONSE_USER_DATA_TABLE`: This will contain the Table name where you want to store all the responses of the user session.

### Running the App
***Note***: Before executing below commands, you should have your virtual environment activated

1. If you want to run the app on your local machine then use an argument as shown below
   ```
   python app.py --env DEV
   ```
2. If the app needs to be run on Prod Server.
   ```
   python app.py
   ```

