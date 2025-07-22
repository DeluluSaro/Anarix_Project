import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import re

# Page configuration
st.set_page_config(
    page_title="Chat with SQL DB", 
    page_icon='💻',
    layout="wide"
)
st.title('Chat with your SQL DB Dynamically!!')

# Sidebar configuration
with st.sidebar:
    api_key = st.text_input(label='Groq API Key', type='password')
    if not api_key:
        st.info('Please add the Groq API Key')
        st.stop()
    
    st.markdown("### Visualization Options")
    show_visualizations = st.checkbox("Enable Auto-Visualizations", value=True)
    chart_types = st.multiselect(
        "Select preferred chart types:",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Heatmap", "Box Plot"],
        default=["Bar Chart", "Line Chart", "Pie Chart"]
    )
    
    if st.button('Clear message history'):
        st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you?'}]

# Initialize components
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

def detect_chart_type(df):
    """Intelligently detect the best chart type for the data"""
    n_rows, n_cols = df.shape
    
    if n_cols == 1:
        return "line"
    elif n_cols == 2:
        col1, col2 = df.columns
        if pd.api.types.is_numeric_dtype(df[col2]):
            if df[col1].nunique() <= 10:  # Categorical with few categories
                return "bar"
            else:
                return "line"
        else:
            return "bar"
    elif n_cols >= 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            return "scatter"
        else:
            return "bar"
    
    return "table"

def create_plotly_chart(df, chart_type):
    """Create various Plotly charts based on data and type"""
    if df.empty:
        return None
    
    try:
        if chart_type == "bar":
            if len(df.columns) >= 2:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], 
                           title=f"{df.columns[1]} by {df.columns[0]}")
            else:
                fig = px.bar(df, y=df.columns[0], title=f"Distribution of {df.columns[0]}")
        
        elif chart_type == "line":
            if len(df.columns) >= 2:
                fig = px.line(df, x=df.columns[0], y=df.columns[1], 
                            title=f"{df.columns[1]} over {df.columns[0]}")
            else:
                fig = px.line(df, y=df.columns[0], title=f"Trend of {df.columns[0]}")
        
        elif chart_type == "scatter":
            if len(df.columns) >= 2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                   title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
                else:
                    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], 
                                   title=f"{df.columns[1]} vs {df.columns[0]}")
        
        elif chart_type == "pie":
            if len(df.columns) >= 2:
                fig = px.pie(df, names=df.columns[0], values=df.columns[1], 
                           title=f"Distribution of {df.columns[1]} by {df.columns[0]}")
            else:
                value_counts = df[df.columns[0]].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Distribution of {df.columns[0]}")
        
        elif chart_type == "histogram":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(df, x=numeric_cols[0], 
                                 title=f"Distribution of {numeric_cols[0]}")
            else:
                fig = px.histogram(df, x=df.columns[0], 
                                 title=f"Distribution of {df.columns[0]}")
        
        elif chart_type == "box":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.box(df, y=numeric_cols[0], 
                           title=f"Box Plot of {numeric_cols[0]}")
            else:
                return None
        
        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Correlation Heatmap")
            else:
                return None
        
        else:
            return None
        
        # Apply Streamlit theme and styling
        fig.update_layout(
            template="plotly_white",
            height=500,
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating {chart_type} chart: {e}")
        return None

def create_multiple_visualizations(df):
    """Create multiple visualizations for comprehensive data analysis"""
    if df.empty:
        return
    
    st.markdown("### 📊 Data Visualizations")
    
    # Create tabs for different visualizations
    chart_tabs = []
    available_charts = []
    
    # Determine which charts are applicable
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(df.columns) >= 2 and "Bar Chart" in chart_types:
        available_charts.append("Bar Chart")
    if len(numeric_cols) > 0 and "Line Chart" in chart_types:
        available_charts.append("Line Chart")
    if len(numeric_cols) >= 2 and "Scatter Plot" in chart_types:
        available_charts.append("Scatter Plot")
    if len(df.columns) >= 2 and "Pie Chart" in chart_types:
        available_charts.append("Pie Chart")
    if len(numeric_cols) > 0 and "Histogram" in chart_types:
        available_charts.append("Histogram")
    if len(numeric_cols) >= 2 and "Heatmap" in chart_types:
        available_charts.append("Heatmap")
    if len(numeric_cols) > 0 and "Box Plot" in chart_types:
        available_charts.append("Box Plot")
    
    if available_charts:
        tabs = st.tabs(available_charts + ["Data Summary"])
        
        for i, chart_name in enumerate(available_charts):
            with tabs[i]:
                chart_type_map = {
                    "Bar Chart": "bar",
                    "Line Chart": "line", 
                    "Scatter Plot": "scatter",
                    "Pie Chart": "pie",
                    "Histogram": "histogram",
                    "Heatmap": "heatmap",
                    "Box Plot": "box"
                }
                
                fig = create_plotly_chart(df, chart_type_map[chart_name])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"{chart_name} not applicable for this data structure.")
        
        # Data Summary tab
        with tabs[-1]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Data Info:**")
                st.write(f"• Rows: {len(df)}")
                st.write(f"• Columns: {len(df.columns)}")
                st.write(f"• Numeric columns: {len(numeric_cols)}")
                st.write(f"• Categorical columns: {len(categorical_cols)}")
            
            with col2:
                st.markdown("**📊 Statistical Summary:**")
                if len(numeric_cols) > 0:
                    st.dataframe(df[numeric_cols].describe())
                else:
                    st.write("No numeric columns for statistical summary")

def extract_sql_from_response(response):
    """Enhanced SQL extraction from agent response"""
    sql_query = None
    
    # Multiple patterns to catch SQL queries
    patterns = [
        r"```sql\n(.*?)```",
        r"```sql\r?\n(.*?)```", 
        r"```\n(SELECT[\s\S]+?)```",
        r"(SELECT[\s\S]+?;)",
        r"Action Input:\s*[\"']?(SELECT[\s\S]+?)[\"']?(?:\n|$)",
        r"Query:\s*[\"']?(SELECT[\s\S]+?)[\"']?(?:\n|$)",
        r"SQL:\s*[\"']?(SELECT[\s\S]+?)[\"']?(?:\n|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            sql_query = match.group(1).strip()
            break
    
    if sql_query:
        # Clean up the SQL query
        sql_query = sql_query.replace('\\n', ' ').replace('\n', ' ')
        sql_query = re.sub(r'\s+', ' ', sql_query)  # Remove extra whitespace
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
    
    return sql_query

def extract_structured_data_from_response(response):
    """Extract structured data from various text response formats"""
    
    # Pattern 1: Tuple/List data like [('2025-06-13', 21, 46654.18), ...]
    tuple_pattern = r"\[\s*(?:\([^)]+\),?\s*)+\]"
    tuple_match = re.search(tuple_pattern, response)
    
    if tuple_match:
        try:
            # Extract the list string and evaluate it safely
            data_str = tuple_match.group(0)
            # Use ast.literal_eval for safe evaluation
            import ast
            data_list = ast.literal_eval(data_str)
            
            # Convert to DataFrame
            if data_list and len(data_list) > 0:
                # Determine number of columns from first tuple
                if isinstance(data_list[0], tuple):
                    n_cols = len(data_list[0])
                    
                    # Create column names based on context and data types
                    columns = []
                    sample_row = data_list[0]
                    
                    for i, val in enumerate(sample_row):
                        if isinstance(val, str) and '-' in val and len(val) == 10:
                            columns.append(f"Date")
                        elif isinstance(val, (int, float)) and val > 1000:
                            columns.append(f"Amount")
                        elif isinstance(val, (int, float)) and val < 1000:
                            columns.append(f"ID/Count")
                        else:
                            columns.append(f"Column_{i+1}")
                    
                    # Handle common patterns
                    if "top" in response.lower() and "product" in response.lower():
                        if len(columns) >= 2:
                            columns[0] = "Product_Info"
                            columns[-1] = "Sales_Amount"
                    
                    df = pd.DataFrame(data_list, columns=columns)
                    return df
        except:
            pass
    
    # Pattern 2: Key-value pairs like "Product ID 1 with 500 purchases"
    kv_patterns = [
        r"Product ID\s*(\d+)\s*with\s*(\d+)\s*purchases",
        r"Product\s*(\w+)[:\s]*(\d+\.?\d*)",
        r"Customer\s*(\w+)[:\s]*(\d+\.?\d*)",
        r"(\w+)[:\s]*(\d+\.?\d*)",
    ]
    
    for pattern in kv_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches and len(matches) > 2:  # At least 3 data points
            df = pd.DataFrame(matches, columns=["Item", "Value"])
            df["Value"] = pd.to_numeric(df["Value"], errors='coerce')
            return df
    
    # Pattern 3: Simple number lists
    number_list_pattern = r"\[(\d+(?:\.\d+)?(?:,\s*\d+(?:\.\d+)?)*)\]"
    number_match = re.search(number_list_pattern, response)
    
    if number_match:
        numbers = [float(x.strip()) for x in number_match.group(1).split(',')]
        df = pd.DataFrame({"Index": range(len(numbers)), "Value": numbers})
        return df
    
    return None

def refine_query_for_visualization(original_query, response):
    """Generate a refined query to get better structured data"""
    
    # Analyze the original query to understand intent
    query_lower = original_query.lower()
    
    suggestions = []
    
    if "top" in query_lower and "product" in query_lower:
        suggestions = [
            "SELECT product_name, SUM(sales_amount) as total_sales FROM sales_table GROUP BY product_name ORDER BY total_sales DESC LIMIT 10",
            "Can you show me the product names and their total sales amounts in a table format?",
            "List the top 10 products with their names and sales amounts as separate columns"
        ]
    
    elif "customer" in query_lower:
        suggestions = [
            "SELECT customer_name, SUM(purchase_amount) as total_purchases FROM customer_table GROUP BY customer_name ORDER BY total_purchases DESC LIMIT 10",
            "Show me customer names and their total purchase amounts in table format"
        ]
    
    elif "sales" in query_lower or "revenue" in query_lower:
        suggestions = [
            "SELECT date, product_category, SUM(amount) as total_amount FROM sales GROUP BY date, product_category ORDER BY date",
            "Show me sales data with clear column headers for dates, products, and amounts"
        ]
    
    return suggestions

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you analyze your database? 🔍'}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# User input
user_query = st.chat_input(placeholder='Ask anything from the database (e.g., "Show me top 10 products by sales")')

if user_query:
    st.session_state.messages.append({'role': 'user', 'content': user_query})
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        # Create containers for organized output
        response_container = st.container()
        
        with response_container:
            # Show agent thinking process
            streamlit_callback = StreamlitCallbackHandler(st.container())
            
            with st.spinner("🤖 Analyzing your query..."):
                response = agent.run(user_query, callbacks=[streamlit_callback])
            
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            
            # Display the response
            st.markdown("### 🤖 Assistant Response")
            st.write(response)
            
            # Extract and execute SQL if present
            sql_query = extract_sql_from_response(response)
            
            if sql_query:
                st.markdown("### 🔍 Extracted SQL Query")
                st.code(sql_query, language='sql')
                
                try:
                    engine = create_engine(f'sqlite:///{dbfilepath}')
                    df = pd.read_sql_query(sql_query, engine)
                    engine.dispose()
                    
                    if not df.empty:
                        # Display data table
                        st.markdown("### 📋 Query Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Create visualizations if enabled
                        if show_visualizations and len(chart_types) > 0:
                            create_multiple_visualizations(df)
                        
                        # Option to download data
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download data as CSV",
                            data=csv,
                            file_name=f"query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    else:
                        st.info("🔍 The SQL query executed successfully but returned no data.")
                
                except Exception as e:
                    st.error(f"❌ Error executing SQL query: {e}")
            
            else:
                # No SQL found - try to extract structured data from text response
                st.markdown("### 🔄 Extracting Data from Response...")
                
                extracted_df = extract_structured_data_from_response(response)
                
                if extracted_df is not None and not extracted_df.empty:
                    st.success("✅ Successfully extracted structured data from response!")
                    
                    # Display extracted data
                    st.markdown("### 📊 Extracted Data")
                    st.dataframe(extracted_df, use_container_width=True)
                    
                    # Create visualizations
                    if show_visualizations and len(chart_types) > 0:
                        create_multiple_visualizations(extracted_df)
                    
                    # Download option
                    csv = extracted_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download extracted data as CSV",
                        data=csv,
                        file_name=f"extracted_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    # Provide suggestions for better queries
                    st.warning("🤔 No structured data detected. Let me help you get better results!")
                    
                    suggestions = refine_query_for_visualization(user_query, response)
                    
                    if suggestions:
                        st.markdown("### 💡 Try these refined queries:")
                        
                        for i, suggestion in enumerate(suggestions):
                            if suggestion.upper().startswith('SELECT'):
                                st.code(suggestion, language='sql')
                                if st.button(f"🚀 Run this query", key=f"sql_btn_{i}"):
                                    # Auto-run the suggested query
                                    try:
                                        engine = create_engine(f'sqlite:///{dbfilepath}')
                                        auto_df = pd.read_sql_query(suggestion, engine)
                                        engine.dispose()
                                        
                                        if not auto_df.empty:
                                            st.markdown("### 📋 Auto-Generated Results")
                                            st.dataframe(auto_df, use_container_width=True)
                                            
                                            if show_visualizations:
                                                create_multiple_visualizations(auto_df)
                                    except Exception as e:
                                        st.error(f"Error with suggested query: {e}")
                            else:
                                st.info(f"💬 **Refined prompt:** {suggestion}")
                                if st.button(f"✨ Ask this instead", key=f"prompt_btn_{i}"):
                                    st.session_state.refined_query = suggestion
                                    st.rerun()
                    
                    # Manual refinement option
                    with st.expander("🔧 Manual Query Refinement"):
                        st.markdown("**Common issues and solutions:**")
                        st.markdown("""
                        - **Issue**: Data returned as tuples/lists
                        - **Solution**: Ask for "table format" or "with column names"
                        - **Better prompt**: "Show me the top 10 products with product names and sales amounts in separate columns"
                        
                        **Examples of good prompts:**
                        - ✅ "Display product names and their sales in a table"
                        - ✅ "Show customer data with clear column headers"
                        - ❌ "Get me the data" (too vague)
                        """)
                        
                        manual_query = st.text_area(
                            "✏️ Refine your query:",
                            value=user_query,
                            help="Try rephrasing to be more specific about wanting tabular data"
                        )
                        
                        if st.button("🔄 Retry with refined query") and manual_query != user_query:
                            st.session_state.refined_query = manual_query
                            st.rerun()
            
            # Handle refined queries
            if hasattr(st.session_state, 'refined_query') and st.session_state.refined_query:
                refined_query = st.session_state.refined_query
                delattr(st.session_state, 'refined_query')
                
                st.markdown("### 🔄 Retrying with refined query...")
                st.info(f"New query: {refined_query}")
                
                with st.spinner("🤖 Processing refined query..."):
                    refined_response = agent.run(refined_query, callbacks=[streamlit_callback])
                
                st.session_state.messages.append({'role': 'assistant', 'content': refined_response})
                st.write(refined_response)
                
                # Try to extract data from refined response
                refined_sql = extract_sql_from_response(refined_response)
                if refined_sql:
                    try:
                        engine = create_engine(f'sqlite:///{dbfilepath}')
                        refined_df = pd.read_sql_query(refined_sql, engine)
                        engine.dispose()
                        
                        if not refined_df.empty:
                            st.markdown("### 📊 Refined Results")
                            st.dataframe(refined_df, use_container_width=True)
                            
                            if show_visualizations:
                                create_multiple_visualizations(refined_df)
                    except Exception as e:
                        st.error(f"Error with refined query: {e}")
                else:
                    refined_extracted_df = extract_structured_data_from_response(refined_response)
                    if refined_extracted_df is not None:
                        st.markdown("### 📊 Refined Extracted Data")
                        st.dataframe(refined_extracted_df, use_container_width=True)
                        
                        if show_visualizations:
                            create_multiple_visualizations(refined_extracted_df)

# Footer with helpful tips
st.markdown("---")
st.markdown("""
### 💡 Tips for better queries:
- **Data exploration**: "Show me all table names" or "Describe the structure of [table_name]"
- **Specific analysis**: "Find top 10 customers by total purchases"
- **Comparisons**: "Compare sales by region for the last quarter"
- **Trends**: "Show monthly revenue trends for 2023"
""")