import getpass
import os
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.agent_toolkits import create_sql_agent

os.environ["OPENAI_API_KEY"] = getpass.getpass()


#View DB Schema
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
result = db.run("SELECT * FROM Artist LIMIT 10;")
#print(result)


#LLM SQL Query Generation 
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
#print(response)

#Query Execution 
db.run(response)
#chain.get_prompts()[0].pretty_print()
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
result = chain.invoke({"question": "How many employees are there"})
#print(result)


#Response Generation

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

final_reponse = chain.invoke({"question": "How many employees are there"})
#print(final_reponse)

#SQL Agents
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

agent_executor.invoke(
    {
        "input": "List the total sales per country. Which country's customers spent the most?"
    }
)

