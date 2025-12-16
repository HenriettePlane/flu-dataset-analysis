import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Flu Dataset Analysis",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data with caching
@st.cache_data
def load_data():
    """Load the flu dataset"""
    df = pd.read_csv('flu_border_states_last_20_years.csv', low_memory=False)
    # Convert Collection_Date to datetime
    df['Collection_Date'] = pd.to_datetime(df['Collection_Date'], errors='coerce')
    # Extract year from collection date
    df['Year'] = df['Collection_Date'].dt.year
    return df

# Load data
df = load_data()

# Sidebar
with st.sidebar:
    st.header("🦠 Flu Dataset Analysis")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "Data Exploration", "Visualizations", "Data Table"]
    )
    st.markdown("---")
    st.info(f"**Total Records:** {len(df):,}")

# Overview page
if page == "Overview":
    st.title("🦠 Flu Border States Dataset Analysis")
    st.markdown("Analysis of influenza data from border states over the last 20 years")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        unique_states = df['state'].nunique()
        st.metric("States", unique_states)
    
    with col3:
        unique_genotypes = df['Genotype'].nunique()
        st.metric("Genotypes", unique_genotypes)
    
    with col4:
        date_range = f"{int(df['Year'].min())} - {int(df['Year'].max())}"
        st.metric("Year Range", date_range)
    
    st.markdown("---")
    
    # States breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Records by State")
        state_counts = df['state'].value_counts()
        st.bar_chart(state_counts)
    
    with col2:
        st.subheader("🧬 Top Genotypes")
        genotype_counts = df['Genotype'].value_counts().head(10)
        st.bar_chart(genotype_counts)
    
    st.markdown("---")
    
    # Yearly trends
    st.subheader("📈 Yearly Trends")
    yearly_counts = df.groupby('Year').size().reset_index(name='Count')
    yearly_counts = yearly_counts[yearly_counts['Year'].notna()]
    st.line_chart(yearly_counts.set_index('Year'))
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📋 Dataset Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**States in Dataset:**")
        states_list = sorted(df['state'].dropna().unique())
        for state in states_list:
            count = len(df[df['state'] == state])
            st.write(f"- {state}: {count:,} records")
    
    with col2:
        st.write("**Genotypes in Dataset:**")
        genotypes_list = df['Genotype'].value_counts().head(10)
        for genotype, count in genotypes_list.items():
            if pd.notna(genotype):
                st.write(f"- {genotype}: {count:,} records")

# Data Exploration page
elif page == "Data Exploration":
    st.title("🔍 Data Exploration")
    st.markdown("Filter and explore the flu dataset")
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        states_filter = st.multiselect(
            "Select States",
            options=sorted(df['state'].dropna().unique()),
            default=None
        )
    
    with col2:
        genotypes_filter = st.multiselect(
            "Select Genotypes",
            options=sorted(df['Genotype'].dropna().unique()),
            default=None
        )
    
    with col3:
        year_range = st.slider(
            "Year Range",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max()))
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if states_filter:
        filtered_df = filtered_df[filtered_df['state'].isin(states_filter)]
    
    if genotypes_filter:
        filtered_df = filtered_df[filtered_df['Genotype'].isin(genotypes_filter)]
    
    filtered_df = filtered_df[
        (filtered_df['Year'] >= year_range[0]) & 
        (filtered_df['Year'] <= year_range[1])
    ]
    
    # Display filtered results
    st.markdown("---")
    st.subheader(f"Filtered Results: {len(filtered_df):,} records")
    
    if len(filtered_df) > 0:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Filtered Records", f"{len(filtered_df):,}")
        
        with col2:
            states_in_filter = filtered_df['state'].nunique()
            st.metric("States", states_in_filter)
        
        with col3:
            genotypes_in_filter = filtered_df['Genotype'].nunique()
            st.metric("Genotypes", genotypes_in_filter)
        
        # Breakdown charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Records by State**")
            state_filtered_counts = filtered_df['state'].value_counts()
            st.bar_chart(state_filtered_counts)
        
        with col2:
            st.write("**Records by Genotype**")
            genotype_filtered_counts = filtered_df['Genotype'].value_counts().head(10)
            st.bar_chart(genotype_filtered_counts)
        
        # Yearly trend for filtered data
        st.write("**Yearly Trend (Filtered)**")
        yearly_filtered = filtered_df.groupby('Year').size().reset_index(name='Count')
        yearly_filtered = yearly_filtered[yearly_filtered['Year'].notna()]
        if len(yearly_filtered) > 0:
            st.line_chart(yearly_filtered.set_index('Year'))
    else:
        st.warning("No records match the selected filters.")

# Visualizations page
elif page == "Visualizations":
    st.title("📊 Visualizations")
    st.markdown("Interactive charts and graphs")
    st.markdown("---")
    
    viz_type = st.selectbox(
        "Select Visualization",
        ["Genotype Distribution", "State Comparison", "Temporal Trends", "Genotype by State"]
    )
    
    st.markdown("---")
    
    if viz_type == "Genotype Distribution":
        st.subheader("Genotype Distribution")
        genotype_counts = df['Genotype'].value_counts().head(15)
        genotype_counts = genotype_counts[genotype_counts.index.notna()]
        st.bar_chart(genotype_counts)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top 10 Genotypes**")
            st.dataframe(genotype_counts.head(10).reset_index().rename(columns={'index': 'Genotype', 'Genotype': 'Count'}))
    
    elif viz_type == "State Comparison":
        st.subheader("State Comparison")
        state_counts = df['state'].value_counts()
        st.bar_chart(state_counts)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Records by State**")
            st.dataframe(state_counts.reset_index().rename(columns={'index': 'State', 'state': 'Count'}))
    
    elif viz_type == "Temporal Trends":
        st.subheader("Temporal Trends Over Time")
        
        # Yearly counts
        yearly_counts = df.groupby('Year').size().reset_index(name='Count')
        yearly_counts = yearly_counts[yearly_counts['Year'].notna()]
        st.line_chart(yearly_counts.set_index('Year'))
        
        # Genotype trends over time
        st.write("**Genotype Trends Over Time**")
        top_genotypes = df['Genotype'].value_counts().head(5).index.tolist()
        top_genotypes = [g for g in top_genotypes if pd.notna(g)]
        
        genotype_trends = df[df['Genotype'].isin(top_genotypes)].groupby(['Year', 'Genotype']).size().reset_index(name='Count')
        genotype_trends = genotype_trends[genotype_trends['Year'].notna()]
        
        if len(genotype_trends) > 0:
            pivot_trends = genotype_trends.pivot(index='Year', columns='Genotype', values='Count').fillna(0)
            st.line_chart(pivot_trends)
    
    elif viz_type == "Genotype by State":
        st.subheader("Genotype Distribution by State")
        
        # Create cross-tabulation
        crosstab = pd.crosstab(df['state'], df['Genotype'])
        crosstab = crosstab.loc[:, crosstab.columns.notna()]  # Remove NaN columns
        
        # Show top genotypes per state
        st.bar_chart(crosstab)
        
        st.write("**Cross-tabulation Table**")
        st.dataframe(crosstab, use_container_width=True)

# Data Table page
elif page == "Data Table":
    st.title("📋 Data Table")
    st.markdown("Browse and search the dataset")
    st.markdown("---")
    
    # Search functionality
    search_term = st.text_input("Search (Accession, Organism Name, or Genotype)", "")
    
    # Filter by search
    display_df = df.copy()
    
    if search_term:
        mask = (
            display_df['Accession'].astype(str).str.contains(search_term, case=False, na=False) |
            display_df['Organism_Name'].astype(str).str.contains(search_term, case=False, na=False) |
            display_df['Genotype'].astype(str).str.contains(search_term, case=False, na=False)
        )
        display_df = display_df[mask]
    
    # Column selection
    st.write("**Select columns to display:**")
    default_cols = ['Accession', 'Genotype', 'state', 'Collection_Date', 'Year', 'Nuc_Completeness']
    available_cols = [col for col in default_cols if col in display_df.columns]
    selected_cols = st.multiselect(
        "Columns",
        options=display_df.columns.tolist(),
        default=available_cols
    )
    
    if selected_cols:
        display_df = display_df[selected_cols]
    
    # Display table
    st.markdown("---")
    st.write(f"**Showing {len(display_df):,} records**")
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name=f"flu_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Flu Border States Dataset Analysis")
