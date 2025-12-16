import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Simple Streamlit App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🚀 Welcome to My Streamlit App")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page:",
        ["Home", "Data Visualization", "Calculator"]
    )

# Home page
if page == "Home":
    st.header("Home")
    st.write("This is a simple Streamlit application with multiple features.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Users", "1,234", "+123")
    
    with col2:
        st.metric("Revenue", "$50,000", "+12%")
    
    with col3:
        st.metric("Growth", "45%", "+5%")
    
    st.markdown("---")
    st.subheader("Features")
    st.write("""
    - **Data Visualization**: Create interactive charts and graphs
    - **Calculator**: Perform basic calculations
    - **Modern UI**: Clean and responsive design
    """)

# Data Visualization page
elif page == "Data Visualization":
    st.header("Data Visualization")
    
    # Generate sample data
    chart_type = st.selectbox("Select chart type", ["Line Chart", "Bar Chart", "Area Chart"])
    
    # Create sample data
    data = pd.DataFrame({
        'x': np.arange(1, 21),
        'y': np.random.randn(20).cumsum()
    })
    
    if chart_type == "Line Chart":
        st.line_chart(data.set_index('x'))
    elif chart_type == "Bar Chart":
        st.bar_chart(data.set_index('x'))
    elif chart_type == "Area Chart":
        st.area_chart(data.set_index('x'))
    
    # Display data table
    st.subheader("Data Table")
    st.dataframe(data, use_container_width=True)

# Calculator page
elif page == "Calculator":
    st.header("Simple Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num1 = st.number_input("First number", value=0.0)
        num2 = st.number_input("Second number", value=0.0)
    
    with col2:
        operation = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide"])
    
    if st.button("Calculate"):
        if operation == "Add":
            result = num1 + num2
        elif operation == "Subtract":
            result = num1 - num2
        elif operation == "Multiply":
            result = num1 * num2
        elif operation == "Divide":
            if num2 != 0:
                result = num1 / num2
            else:
                st.error("Cannot divide by zero!")
                result = None
        
        if result is not None:
            st.success(f"Result: **{result}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")

