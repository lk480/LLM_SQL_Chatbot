import os
import getpass
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, inspect
from operator import itemgetter
import re

# Request OpenAI API key
os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Create SQLAlchemy engine
db_path = 'sqlite:///substrata_interview.sqlite3'
engine = create_engine(db_path)

# Connect to the SQLite database using the engine
db = SQLDatabase(engine)

# Create the SQL query chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
write_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)

# Define the answer prompt template
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()

# Create the chain for query execution and response generation
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

# Memory to store job_name
memory = {}

# Function to extract job_name from user input
def extract_job_name(user_input):
    match = re.search(r'for\s+the\s+job\s+named\s+"([^"]+)"', user_input)
    return match.group(1) if match else None

# Function to inspect the table schema and get column names
def get_table_columns(table_name):
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return [column['name'] for column in columns]

# Function to handle user input and generate responses
def chatbot(user_input):
    table_columns = get_table_columns('substrata_api_jobparameters')

    if "Summarize the parameters for the job named" in user_input:
        job_name = extract_job_name(user_input)
        if job_name:
            memory["job_name"] = job_name
        elif "job_name" not in memory:
            return "No job name found in your input and no job name stored in memory. Please provide a job name."
        else:
            job_name = memory["job_name"]

        # Generate dynamic query to select all columns for the given job_name
        non_null_columns = [col for col in table_columns if col != 'job_name']
        select_clause = ", ".join(non_null_columns)
        query = f"SELECT {select_clause} FROM substrata_api_jobparameters WHERE job_name = '{job_name}'"
        
        #print(f"Generated SQL Query: {query}")  # Debug statement to print the SQL query
        
        # Execute the SQL query
        result = execute_query.invoke({"query": query})
        
        # Initialize a dictionary to store non-NULL values
        non_null_results = []
        
        # Process the results to include only non-NULL values
        for row in result:
            row_data = {}
            for col in non_null_columns:
                if row[col] is not None:
                    row_data[col] = row[col]
            if row_data:
                non_null_results.append(row_data)
        
        # Filter results if "AWI" is in the user input
        if "AWI" in user_input:
            awi_results = [{col: value for col, value in row.items() if col.startswith('awi_')} for row in non_null_results]
            formatted_awis = "\n".join([f"{col}: {value}" for row in awi_results for col, value in row.items()])
            formatted_answer = f"AWI Parameters for job_name {job_name}:\n{formatted_awis}"
        else:
            formatted_params = "\n".join([f"{col}: {value}" for row in non_null_results for col, value in row.items()])
            formatted_answer = f"Parameters for job_name {job_name}:\n{formatted_params}"
    else:
        query = write_query.invoke({"question": user_input})

        #print(f"Generated SQL Query: {query}")  # Debug statement to print the SQL query

        result = execute_query.invoke({"query": query})
    
        formatted_answer = answer.invoke({
            "question": user_input,
            "query": query,
            "result": result
        })
    
    return formatted_answer

# Main function to run the chatbot
def main():
    while True:
        user_input = input("Enter your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        result = chatbot(user_input)
        print(result)

if __name__ == "__main__":
    main()