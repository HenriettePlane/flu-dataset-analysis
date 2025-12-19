import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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

@st.cache_data
def load_dengue_data():
    """Load the dengue dataset"""
    try:
        df_dengue = pd.read_csv('border_data.csv', low_memory=False)
        # Convert Collection_Date to datetime
        df_dengue['Collection_Date'] = pd.to_datetime(df_dengue['Collection_Date'], errors='coerce')
        # Extract year from collection date
        df_dengue['Year'] = df_dengue['Collection_Date'].dt.year
        # Extract state from Geo_Location if available
        if 'Geo_Location' in df_dengue.columns:
            df_dengue['state'] = df_dengue['Geo_Location'].str.extract(r'(?:USA|Mexico):\s*([^,]+)', expand=False)
        return df_dengue
    except Exception as e:
        st.error(f"Error loading dengue data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()
df_dengue = load_dengue_data()

# Sidebar
with st.sidebar:
    st.header("🦠 Flu Dataset Analysis")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Flu Overview", "Flu Data Exploration", "Flu Visualizations", "Flu Map", "Flu Data Table",
         "Dengue Overview", "Dengue Analysis", "Dengue Map"]
    )
    st.markdown("---")
    if page in ["Dengue Overview", "Dengue Analysis", "Dengue Map"]:
        if len(df_dengue) > 0:
            st.info(f"**Dengue Records:** {len(df_dengue):,}")
        else:
            st.warning("Dengue data not available")
    elif page in ["Flu Overview", "Flu Data Exploration", "Flu Visualizations", "Flu Map", "Flu Data Table"]:
        st.info(f"**Flu Records:** {len(df):,}")
    else:
        st.info(f"**Flu Records:** {len(df):,}")

# Overview page
if page == "Flu Overview":
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
elif page == "Flu Data Exploration":
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
        
        # Yearly trend by genotype
        st.markdown("---")
        st.subheader("📈 Yearly Trend by Genotype")
        st.write("Track the evolution of each genotype over time")
        
        # Prepare data for genotype trends
        genotype_trends = filtered_df.groupby(['Year', 'Genotype']).size().reset_index(name='Count')
        genotype_trends = genotype_trends[genotype_trends['Year'].notna()]
        genotype_trends = genotype_trends[genotype_trends['Genotype'].notna()]
        
        if len(genotype_trends) > 0:
            # Create pivot table for line chart
            pivot_trends = genotype_trends.pivot(index='Year', columns='Genotype', values='Count').fillna(0)
            
            # Create interactive line chart with Plotly
            fig_genotype_trends = px.line(
                pivot_trends.reset_index(),
                x='Year',
                y=[col for col in pivot_trends.columns],
                title='Yearly Trend by Genotype',
                labels={'value': 'Number of Records', 'Year': 'Year'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_genotype_trends.update_layout(
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    x=1.02
                ),
                xaxis_title="Year",
                yaxis_title="Number of Records"
            )
            
            st.plotly_chart(fig_genotype_trends, use_container_width=True)
            
            # Show data table
            with st.expander("View Data Table"):
                st.dataframe(
                    genotype_trends.sort_values(['Year', 'Count'], ascending=[True, False]),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No genotype trend data available for the selected filters.")
    else:
        st.warning("No records match the selected filters.")

# Visualizations page
elif page == "Flu Visualizations":
    st.title("📊 Visualizations")
    st.markdown("Interactive charts and graphs")
    st.markdown("---")
    
    viz_type = st.selectbox(
        "Select Visualization",
        ["Genotype Distribution", "State Comparison", "Temporal Trends", "Genotype by State"],
        index=3
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

# Map page
elif page == "Flu Map":
    st.title("🗺️ Genotype Distribution Map")
    st.markdown("Visualize flu genotypes by geographic location")
    st.markdown("---")
    
    # State coordinates (approximate centroids)
    state_coords = {
        'California': {'lat': 36.7783, 'lon': -119.4179},
        'Arizona': {'lat': 34.0489, 'lon': -111.0937},
        'New Mexico': {'lat': 34.5199, 'lon': -105.8701},
        'Texas': {'lat': 31.9686, 'lon': -99.9018},
        'Baja California': {'lat': 30.8406, 'lon': -115.2838},
        'Sonora': {'lat': 29.2972, 'lon': -110.3309},
        'Chihuahua': {'lat': 28.6353, 'lon': -106.0889},
        'Nuevo Leon': {'lat': 25.6866, 'lon': -100.3161}
    }
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        selected_genotype = st.selectbox(
            "Select Genotype to Visualize",
            options=['All'] + sorted([g for g in df['Genotype'].dropna().unique() if pd.notna(g)]),
            index=0
        )
    
    with col2:
        map_type = st.selectbox(
            "Map Type",
            options=["Scatter Map (by State)", "Choropleth (by State)"],
            index=0
        )
    
    # Prepare data for map
    map_df = df.copy()
    
    if selected_genotype != 'All':
        map_df = map_df[map_df['Genotype'] == selected_genotype]
    
    # Aggregate by state
    state_data = map_df.groupby('state').agg({
        'Genotype': 'count',
        'Accession': 'count'
    }).reset_index()
    state_data.columns = ['state', 'count', 'total']
    
    # Add coordinates
    state_data['lat'] = state_data['state'].map(lambda x: state_coords.get(x, {}).get('lat', np.nan))
    state_data['lon'] = state_data['state'].map(lambda x: state_coords.get(x, {}).get('lon', np.nan))
    
    # Remove states without coordinates
    state_data = state_data.dropna(subset=['lat', 'lon'])
    
    st.markdown("---")
    
    if len(state_data) > 0:
        if map_type == "Scatter Map (by State)":
            # Create scatter map
            fig = px.scatter_mapbox(
                state_data,
                lat='lat',
                lon='lon',
                size='count',
                hover_name='state',
                hover_data={'count': True, 'lat': False, 'lon': False},
                color='count',
                color_continuous_scale='Viridis',
                size_max=50,
                zoom=4,
                height=600,
                mapbox_style="open-street-map",
                title=f"Genotype Distribution by State{' - ' + selected_genotype if selected_genotype != 'All' else ''}"
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                mapbox=dict(
                    center=dict(lat=32.0, lon=-110.0),
                    zoom=4
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Choropleth
            st.info("Choropleth maps require GeoJSON boundaries. Using scatter map instead.")
            fig = px.scatter_mapbox(
                state_data,
                lat='lat',
                lon='lon',
                size='count',
                hover_name='state',
                hover_data={'count': True},
                color='count',
                color_continuous_scale='Viridis',
                size_max=50,
                zoom=4,
                height=600,
                mapbox_style="open-street-map"
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                mapbox=dict(
                    center=dict(lat=32.0, lon=-110.0),
                    zoom=4
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart visualization for genotype breakdown by state
        st.markdown("---")
        st.subheader("🧬 Genotype Breakdown by State")
        
        # State selector for pie chart
        available_states = sorted([s for s in df['state'].dropna().unique() if s in state_coords])
        selected_state_pie = st.selectbox(
            "Select State to View Genotype Breakdown",
            options=['All States'] + available_states,
            index=0
        )
        
        # Prepare data for pie chart
        pie_df = df.copy()
        
        # Apply genotype filter if selected
        if selected_genotype != 'All':
            pie_df = pie_df[pie_df['Genotype'] == selected_genotype]
        
        # Filter by selected state for pie chart
        if selected_state_pie != 'All States':
            pie_df = pie_df[pie_df['state'] == selected_state_pie]
        
        # Get genotype counts
        genotype_counts = pie_df['Genotype'].value_counts().reset_index()
        genotype_counts.columns = ['Genotype', 'Count']
        genotype_counts = genotype_counts[genotype_counts['Genotype'].notna()]
        
        if len(genotype_counts) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create pie chart
                fig_pie = px.pie(
                    genotype_counts,
                    values='Count',
                    names='Genotype',
                    title=f"Genotype Distribution{' - ' + selected_state_pie if selected_state_pie != 'All States' else ' - All States'}",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )
                fig_pie.update_layout(
                    showlegend=True,
                    height=500,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.write("**Genotype Counts**")
                st.dataframe(
                    genotype_counts.sort_values('Count', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary metrics
                st.markdown("---")
                st.metric("Total Records", f"{genotype_counts['Count'].sum():,}")
                st.metric("Unique Genotypes", len(genotype_counts))
                if selected_state_pie != 'All States':
                    top_genotype = genotype_counts.iloc[0]
                    st.metric("Top Genotype", f"{top_genotype['Genotype']} ({top_genotype['Count']:,})")
        else:
            st.warning(f"No genotype data available for {selected_state_pie}.")
        
        # Show detailed breakdown
        st.markdown("---")
        st.subheader("State Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Records by State**")
            st.dataframe(
                state_data[['state', 'count']].sort_values('count', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            # Genotype breakdown by state
            if selected_genotype == 'All':
                st.write("**Top Genotypes by State**")
                genotype_by_state = map_df.groupby(['state', 'Genotype']).size().reset_index(name='count')
                genotype_by_state = genotype_by_state.sort_values('count', ascending=False).head(20)
                st.dataframe(genotype_by_state, use_container_width=True, hide_index=True)
            else:
                st.write(f"**Distribution of {selected_genotype}**")
                st.metric("Total Records", f"{state_data['count'].sum():,}")
                st.metric("States with this Genotype", len(state_data))
        
        # Additional statistics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{state_data['count'].sum():,}")
        
        with col2:
            st.metric("States Represented", len(state_data))
        
        with col3:
            avg_per_state = state_data['count'].mean()
            st.metric("Avg Records per State", f"{avg_per_state:,.0f}")
    
    else:
        st.warning("No data available for the selected filters.")

# Data Table page
elif page == "Flu Data Table":
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

# Dengue Overview page
elif page == "Dengue Overview":
    if len(df_dengue) == 0:
        st.error("Dengue dataset could not be loaded. Please check if 'border_data.csv' exists.")
    else:
        st.title("🦟 Dengue Fever Dataset Analysis")
        st.markdown("Analysis of dengue virus data from border states")
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_records = len(df_dengue)
            st.metric("Total Records", f"{total_records:,}")
        
        with col2:
            unique_genotypes = df_dengue['Genotype'].nunique() if 'Genotype' in df_dengue.columns else 0
            st.metric("Dengue Types", unique_genotypes)
        
        with col3:
            unique_countries = df_dengue['Country'].nunique() if 'Country' in df_dengue.columns else 0
            st.metric("Countries", unique_countries)
        
        with col4:
            if 'Year' in df_dengue.columns and df_dengue['Year'].notna().any():
                date_range = f"{int(df_dengue['Year'].min())} - {int(df_dengue['Year'].max())}"
            else:
                date_range = "N/A"
            st.metric("Year Range", date_range)
        
        st.markdown("---")
        
        # Genotype breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🦠 Dengue Type Distribution")
            if 'Genotype' in df_dengue.columns:
                genotype_counts = df_dengue['Genotype'].value_counts()
                genotype_counts = genotype_counts[genotype_counts.index.notna()]
                if len(genotype_counts) > 0:
                    pie_df = pd.DataFrame({
                        'Genotype': [f"Type {int(g)}" if pd.notna(g) and not pd.isna(g) else "Unknown" for g in genotype_counts.index],
                        'Count': genotype_counts.values
                    })
                    fig_pie = px.pie(
                        pie_df,
                        values='Count',
                        names='Genotype',
                        title="Distribution of Dengue Types",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No genotype data available")
            else:
                st.info("Genotype column not found")
        
        with col2:
            st.subheader("🌎 Records by Country")
            if 'Country' in df_dengue.columns:
                country_counts = df_dengue['Country'].value_counts()
                st.bar_chart(country_counts)
            else:
                st.info("Country column not found")
            
            st.markdown("---")
            st.write("**Top Locations**")
            if 'Geo_Location' in df_dengue.columns:
                location_counts = df_dengue['Geo_Location'].value_counts().head(10)
                st.dataframe(location_counts.reset_index().rename(columns={'index': 'Location', 'Geo_Location': 'Count'}), 
                            use_container_width=True, hide_index=True)
            else:
                st.info("Location data not available")
        
        st.markdown("---")
        
        # Yearly trends
        if 'Year' in df_dengue.columns and df_dengue['Year'].notna().any():
            st.subheader("📈 Yearly Trends")
            yearly_counts = df_dengue.groupby('Year').size().reset_index(name='Count')
            yearly_counts = yearly_counts[yearly_counts['Year'].notna()]
            if len(yearly_counts) > 0:
                st.line_chart(yearly_counts.set_index('Year'))
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📋 Dataset Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dengue Types in Dataset:**")
            if 'Genotype' in df_dengue.columns:
                genotypes_list = df_dengue['Genotype'].value_counts()
                for genotype, count in genotypes_list.items():
                    if pd.notna(genotype):
                        st.write(f"- Type {int(genotype)}: {count:,} records")
            else:
                st.info("Genotype data not available")
        
        with col2:
            st.write("**Countries in Dataset:**")
            if 'Country' in df_dengue.columns:
                countries_list = df_dengue['Country'].value_counts()
                for country, count in countries_list.items():
                    if pd.notna(country):
                        st.write(f"- {country}: {count:,} records")
            else:
                st.info("Country data not available")

# Dengue Analysis page
elif page == "Dengue Analysis":
    if len(df_dengue) == 0:
        st.error("Dengue dataset could not be loaded. Please check if 'border_data.csv' exists.")
    else:
        st.title("🔬 Dengue Fever Detailed Analysis")
        st.markdown("Comprehensive analysis and exploration of dengue virus data")
        st.markdown("---")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Country' in df_dengue.columns:
                countries_filter = st.multiselect(
                    "Select Countries",
                    options=sorted(df_dengue['Country'].dropna().unique()),
                    default=None
                )
            else:
                countries_filter = None
                st.info("Country filter not available")
        
        with col2:
            if 'Genotype' in df_dengue.columns:
                genotypes_filter = st.multiselect(
                    "Select Dengue Types",
                    options=sorted([g for g in df_dengue['Genotype'].dropna().unique() if pd.notna(g)]),
                    default=None
                )
            else:
                genotypes_filter = None
                st.info("Genotype filter not available")
        
        with col3:
            if 'Year' in df_dengue.columns and df_dengue['Year'].notna().any():
                year_range = st.slider(
                    "Year Range",
                    min_value=int(df_dengue['Year'].min()),
                    max_value=int(df_dengue['Year'].max()),
                    value=(int(df_dengue['Year'].min()), int(df_dengue['Year'].max()))
                )
            else:
                year_range = None
                st.info("Year filter not available")
        
        # Apply filters
        filtered_dengue = df_dengue.copy()
        
        if countries_filter and 'Country' in filtered_dengue.columns:
            filtered_dengue = filtered_dengue[filtered_dengue['Country'].isin(countries_filter)]
        
        if genotypes_filter and 'Genotype' in filtered_dengue.columns:
            filtered_dengue = filtered_dengue[filtered_dengue['Genotype'].isin(genotypes_filter)]
        
        if year_range and 'Year' in filtered_dengue.columns:
            filtered_dengue = filtered_dengue[
                (filtered_dengue['Year'] >= year_range[0]) & 
                (filtered_dengue['Year'] <= year_range[1])
            ]
        
        # Display filtered results
        st.markdown("---")
        st.subheader(f"Filtered Results: {len(filtered_dengue):,} records")
        
        if len(filtered_dengue) > 0:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Filtered Records", f"{len(filtered_dengue):,}")
            
            with col2:
                if 'Country' in filtered_dengue.columns:
                    countries_in_filter = filtered_dengue['Country'].nunique()
                    st.metric("Countries", countries_in_filter)
                else:
                    st.metric("Countries", "N/A")
            
            with col3:
                if 'Genotype' in filtered_dengue.columns:
                    genotypes_in_filter = filtered_dengue['Genotype'].nunique()
                    st.metric("Dengue Types", genotypes_in_filter)
                else:
                    st.metric("Dengue Types", "N/A")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Records by Country**")
                if 'Country' in filtered_dengue.columns:
                    country_filtered_counts = filtered_dengue['Country'].value_counts()
                    st.bar_chart(country_filtered_counts)
                else:
                    st.info("Country data not available")
            
            with col2:
                st.write("**Records by Dengue Type**")
                if 'Genotype' in filtered_dengue.columns:
                    genotype_filtered_counts = filtered_dengue['Genotype'].value_counts()
                    genotype_filtered_counts = genotype_filtered_counts[genotype_filtered_counts.index.notna()]
                    if len(genotype_filtered_counts) > 0:
                        st.bar_chart(genotype_filtered_counts)
                    else:
                        st.info("No genotype data available")
                else:
                    st.info("Genotype data not available")
            
            # Yearly trend by genotype
            if ('Year' in filtered_dengue.columns and filtered_dengue['Year'].notna().any() and 
                'Genotype' in filtered_dengue.columns and filtered_dengue['Genotype'].notna().any()):
                st.markdown("---")
                st.subheader("📈 Yearly Trend by Dengue Type")
                
                genotype_trends = filtered_dengue.groupby(['Year', 'Genotype']).size().reset_index(name='Count')
                genotype_trends = genotype_trends[genotype_trends['Year'].notna()]
                genotype_trends = genotype_trends[genotype_trends['Genotype'].notna()]
                
                if len(genotype_trends) > 0:
                    # Convert Genotype to string and format
                    genotype_trends['Genotype'] = genotype_trends['Genotype'].astype(str)
                    genotype_trends['Genotype'] = genotype_trends['Genotype'].apply(lambda x: f"Type {x}" if x.replace('.','').replace('-','').isdigit() else x)
                    
                    # Use long format directly with Plotly
                    fig_dengue_trends = px.line(
                        genotype_trends,
                        x='Year',
                        y='Count',
                        color='Genotype',
                        title='Yearly Trend by Dengue Type',
                        labels={'Count': 'Number of Records', 'Year': 'Year', 'Genotype': 'Dengue Type'},
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    
                    fig_dengue_trends.update_layout(
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            x=1.02
                        ),
                        xaxis_title="Year",
                        yaxis_title="Number of Records"
                    )
                    
                    st.plotly_chart(fig_dengue_trends, use_container_width=True)
            
            # Location breakdown
            st.markdown("---")
            st.subheader("📍 Geographic Distribution")
            
            if 'Geo_Location' in filtered_dengue.columns:
                location_counts = filtered_dengue['Geo_Location'].value_counts().head(15)
                st.bar_chart(location_counts)
            else:
                st.info("Location data not available")
            
            # Data table
            st.markdown("---")
            st.subheader("📋 Detailed Data")
            
            display_cols = ['Accession', 'Genotype', 'Country', 'Geo_Location', 'Collection_Date', 'Year', 'Nuc_Completeness']
            available_display_cols = [col for col in display_cols if col in filtered_dengue.columns]
            
            if len(available_display_cols) > 0:
                st.dataframe(
                    filtered_dengue[available_display_cols].sort_values('Collection_Date', ascending=False, na_position='last'),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv_dengue = filtered_dengue.to_csv(index=False)
                st.download_button(
                    label="Download filtered dengue data as CSV",
                    data=csv_dengue,
                    file_name=f"dengue_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No displayable columns available")
        else:
            st.warning("No records match the selected filters.")

# Dengue Map page
elif page == "Dengue Map":
    if len(df_dengue) == 0:
        st.error("Dengue dataset could not be loaded. Please check if 'border_data.csv' exists.")
    else:
        st.title("🗺️ Dengue Type Distribution Map")
        st.markdown("Visualize dengue virus types by geographic location")
        st.markdown("---")
        
        # Location coordinates (approximate centroids for dengue locations)
        location_coords = {
            'California': {'lat': 36.7783, 'lon': -119.4179},
            'Texas': {'lat': 31.9686, 'lon': -99.9018},
            'Arizona': {'lat': 34.0489, 'lon': -111.0937},
            'Nuevo Leon': {'lat': 25.6866, 'lon': -100.3161},
            'Monterrey': {'lat': 25.6866, 'lon': -100.3161},
            'Sonora': {'lat': 29.2972, 'lon': -110.3309},
            'Baja California Sur': {'lat': 26.0444, 'lon': -111.6661},
            'Coahuila': {'lat': 27.0586, 'lon': -101.7061},
            'Tamaulipas': {'lat': 23.7363, 'lon': -99.1411},
            'La Colorada': {'lat': 28.6, 'lon': -110.7},
            'Cd Victoria': {'lat': 23.7363, 'lon': -99.1411},
            'Apodaca': {'lat': 25.7833, 'lon': -100.1833},
            'Huatabampo': {'lat': 26.8333, 'lon': -109.6333},
            'La Paz': {'lat': 24.1422, 'lon': -110.3108}
        }
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            selected_dengue_type = st.selectbox(
                "Select Dengue Type to Visualize",
                options=['All'] + sorted([str(int(g)) if pd.notna(g) and not pd.isna(g) else 'Unknown' 
                                         for g in df_dengue['Genotype'].dropna().unique() if pd.notna(g)]),
                index=0
            )
        
        with col2:
            map_type = st.selectbox(
                "Map Type",
                options=["Scatter Map (by Location)", "Counts by Type"],
                index=0
            )
        
        # Prepare data for map
        map_dengue_df = df_dengue.copy()
        
        if selected_dengue_type != 'All':
            if selected_dengue_type != 'Unknown':
                map_dengue_df = map_dengue_df[map_dengue_df['Genotype'].astype(str) == selected_dengue_type]
            else:
                map_dengue_df = map_dengue_df[map_dengue_df['Genotype'].isna()]
        
        # Extract location from Geo_Location
        if 'Geo_Location' in map_dengue_df.columns:
            # Extract state/location name
            map_dengue_df['location'] = map_dengue_df['Geo_Location'].str.extract(r'(?:USA|Mexico):\s*([^,]+)', expand=False)
            # Clean up location names
            map_dengue_df['location'] = map_dengue_df['location'].str.strip()
        
        # Aggregate by location
        if 'location' in map_dengue_df.columns:
            location_data = map_dengue_df.groupby('location').agg({
                'Genotype': 'count',
                'Accession': 'count'
            }).reset_index()
            location_data.columns = ['location', 'count', 'total']
            
            # Try to match locations to coordinates
            def get_coords(loc):
                if pd.isna(loc):
                    return None, None
                loc_str = str(loc).strip()
                # Try exact match first
                if loc_str in location_coords:
                    return location_coords[loc_str]['lat'], location_coords[loc_str]['lon']
                # Try partial matches
                for key, coords in location_coords.items():
                    if key.lower() in loc_str.lower() or loc_str.lower() in key.lower():
                        return coords['lat'], coords['lon']
                return None, None
            
            location_data['lat'] = location_data['location'].apply(lambda x: get_coords(x)[0])
            location_data['lon'] = location_data['location'].apply(lambda x: get_coords(x)[1])
            
            # Remove locations without coordinates
            location_data = location_data.dropna(subset=['lat', 'lon'])
        else:
            location_data = pd.DataFrame(columns=['location', 'count', 'total', 'lat', 'lon'])
        
        st.markdown("---")
        
        if len(location_data) > 0:
            if map_type == "Scatter Map (by Location)":
                # Create scatter map
                fig = px.scatter_mapbox(
                    location_data,
                    lat='lat',
                    lon='lon',
                    size='count',
                    hover_name='location',
                    hover_data={'count': True, 'lat': False, 'lon': False},
                    color='count',
                    color_continuous_scale='Viridis',
                    size_max=50,
                    zoom=4,
                    height=600,
                    mapbox_style="open-street-map",
                    title=f"Dengue Distribution by Location{' - Type ' + selected_dengue_type if selected_dengue_type != 'All' else ''}"
                )
                
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    mapbox=dict(
                        center=dict(lat=32.0, lon=-110.0),
                        zoom=4
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Counts by Type
                # Show breakdown by dengue type for each location
                if 'Genotype' in map_dengue_df.columns and 'location' in map_dengue_df.columns:
                    type_by_location = map_dengue_df.groupby(['location', 'Genotype']).size().reset_index(name='count')
                    type_by_location = type_by_location[type_by_location['location'].notna()]
                    type_by_location = type_by_location[type_by_location['Genotype'].notna()]
                    
                    # Add coordinates
                    type_by_location['lat'] = type_by_location['location'].apply(lambda x: get_coords(x)[0])
                    type_by_location['lon'] = type_by_location['location'].apply(lambda x: get_coords(x)[1])
                    type_by_location = type_by_location.dropna(subset=['lat', 'lon'])
                    
                    if len(type_by_location) > 0:
                        # Format genotype labels
                        type_by_location['Genotype_Label'] = type_by_location['Genotype'].apply(
                            lambda x: f"Type {int(x)}" if pd.notna(x) and not pd.isna(x) else "Unknown"
                        )
                        
                        # Create map with different colors for each type
                        fig = px.scatter_mapbox(
                            type_by_location,
                            lat='lat',
                            lon='lon',
                            size='count',
                            color='Genotype_Label',
                            hover_name='location',
                            hover_data={'count': True, 'Genotype_Label': True, 'lat': False, 'lon': False},
                            size_max=50,
                            zoom=4,
                            height=600,
                            mapbox_style="open-street-map",
                            title="Dengue Counts by Type and Location",
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=30, b=0),
                            mapbox=dict(
                                center=dict(lat=32.0, lon=-110.0),
                                zoom=4
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # Breakdown by type
            st.markdown("---")
            st.subheader("📊 Dengue Type Breakdown by Location")
            
            if 'Genotype' in map_dengue_df.columns and 'location' in map_dengue_df.columns:
                type_location_crosstab = pd.crosstab(map_dengue_df['location'], map_dengue_df['Genotype'])
                type_location_crosstab = type_location_crosstab.loc[:, type_location_crosstab.columns.notna()]
                
                if len(type_location_crosstab) > 0:
                    # Format column names
                    type_location_crosstab.columns = [f"Type {int(c)}" if pd.notna(c) and not pd.isna(c) else "Unknown" 
                                                      for c in type_location_crosstab.columns]
                    
                    st.bar_chart(type_location_crosstab)
                    
                    st.write("**Cross-tabulation Table**")
                    st.dataframe(type_location_crosstab, use_container_width=True)
            
            # Location summary
            st.markdown("---")
            st.subheader("📍 Location Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Records by Location**")
                st.dataframe(
                    location_data[['location', 'count']].sort_values('count', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.write("**Summary Metrics**")
                st.metric("Total Records", f"{location_data['count'].sum():,}")
                st.metric("Locations", len(location_data))
                avg_per_location = location_data['count'].mean()
                st.metric("Avg Records per Location", f"{avg_per_location:,.1f}")
        else:
            st.warning("No location data available for mapping. Please check if Geo_Location data is available in the dataset.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Disease Dataset Analysis (Flu & Dengue)")
