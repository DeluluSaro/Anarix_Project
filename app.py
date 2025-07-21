import streamlit as st
from pathlib import Path
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import re

# --- Streamlit Setup ---
st.set_page_config(page_title="Chat with SQL DB", page_icon='💻')
st.title('Chat with your SQL DB Dynamically!!')

# --- API Key Sidebar ---
api_key = st.sidebar.text_input(label='Groq API Key', type='password')
if not api_key:
    st.info('Please add the Groq API Key')
    st.stop()

# --- Database Setup ---
dbfilepath = (Path(__file__).parent / 'pr_report.db').absolute()
llm = ChatGroq(groq_api_key=api_key, model_name='Llama3-8b-8192', streaming=True)
db = SQLDatabase(create_engine(f'sqlite:///{dbfilepath}'))
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    handle_parsing_errors=True,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# --- Session State Setup ---
if 'messages' not in st.session_state or st.sidebar.button('Clear message history'):
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you?'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# --- User Query ---
user_query = st.chat_input(placeholder='Ask anything from the database')
if user_query:
    st.session_state.messages.append({'role': 'user', 'content': user_query})
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)

        # --- SQL Extraction Logic ---
        sql_query = None
        match_blocks = [
            re.search(r"```sql\n(.*?)```", response, re.DOTALL),
            re.search(r"```sql\r?\n(.*?)```", response, re.DOTALL),
            re.search(r"SELECT[\s\S]+?;", response, re.IGNORECASE)
        ]
        for match in match_blocks:
            if match:
                sql_query = match.group(1).strip() if match.lastindex else match.group(0).strip()
                break

        # --- SQL Execution and Visualization ---
        if sql_query:
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1]

            try:
                engine = create_engine(f'sqlite:///{dbfilepath}')
                df = pd.read_sql_query(sql_query, engine)

                if not df.empty:
                    st.write("### Data Table")
                    st.dataframe(df)

                    if len(df.columns) >= 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
                        st.write("### Bar Chart")
                        st.bar_chart(df.set_index(df.columns[0])[df.columns[1]])
                    elif len(df.columns) == 1:
                        st.write("### Line Chart (Single Column)")
                        st.line_chart(df)
                    else:
                        st.info("Data available but not suitable for bar chart.")
                else:
                    st.info("The SQL query returned no data.")
            except Exception as e:
                st.error(f"Error executing SQL or generating visualization: {e}")

        # --- Fallback: Visualize Text Response Like "Product ID X with Y purchases" ---
        else:
            matches = re.findall(r"Product ID\s*(\d+)\s*with\s*(\d+)\s*purchases", response)
            if matches:
                product_ids = [f"Product {pid}" for pid, _ in matches]
                purchase_counts = [int(purchase) for _, purchase in matches]

                df = pd.DataFrame({
                    "Product": product_ids,
                    "Purchases": purchase_counts
                })

                st.write("### Data Table from Text Response")
                st.dataframe(df)

                st.write("### Bar Chart from Text Response")
                st.bar_chart(df.set_index("Product")["Purchases"])
            else:
                st.info("No SQL or structured product data found to visualize.")
