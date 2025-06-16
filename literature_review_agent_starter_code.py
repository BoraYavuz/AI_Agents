!pip install langchain.agents, langchain_openai, langchain.tools, langchain_community.utilities, langchain.prompts

import streamlit as st
import pandas as pd
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import ArxivQueryRun, Tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain.prompts import ChatPromptTemplate
import arxiv
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Research Literature Agent",
    page_icon="🔬",
    layout="wide"
)

# Sidebar for configuration
st.sidebar.title("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
max_papers = st.sidebar.slider("Max papers to analyze", 5, 50, 10)
years_back = st.sidebar.slider("Years to look back", 1, 10, 3)

def search_and_analyze_papers(query, max_results=10):
    """Search and analyze papers using arXiv"""
    try:
        # Search arXiv
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for paper in search.results():
            papers.append({
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'published': paper.published.strftime('%Y-%m-%d'),
                'summary': paper.summary,
                'url': paper.entry_id,
                'categories': paper.categories
            })
        
        return papers
    except Exception as e:
        st.error(f"Error searching papers: {e}")
        return []

def create_research_tools():
    """Create tools for the research agent"""
    
    def paper_search_tool(query: str) -> str:
        """Search for academic papers on a given topic"""
        papers = search_and_analyze_papers(query, max_papers)
        if not papers:
            return "No papers found for this query."
        
        result = f"Found {len(papers)} papers on '{query}':\n\n"
        for i, paper in enumerate(papers, 1):
            result += f"{i}. **{paper['title']}**\n"
            result += f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n"
            result += f"   Published: {paper['published']}\n"
            result += f"   Summary: {paper['summary'][:200]}...\n\n"
        
        return result
    
    def trend_analysis_tool(query: str) -> str:
        """Analyze research trends over time"""
        papers = search_and_analyze_papers(query, max_papers * 2)
        if not papers:
            return "No papers found for trend analysis."
        
        # Group by year
        df = pd.DataFrame(papers)
        df['year'] = pd.to_datetime(df['published']).dt.year
        yearly_counts = df.groupby('year').size().reset_index(name='count')
        
        # Create visualization data
        trend_info = f"Research trend analysis for '{query}':\n"
        trend_info += f"Total papers analyzed: {len(papers)}\n"
        trend_info += f"Year range: {yearly_counts['year'].min()} - {yearly_counts['year'].max()}\n"
        trend_info += f"Peak year: {yearly_counts.loc[yearly_counts['count'].idxmax(), 'year']} ({yearly_counts['count'].max()} papers)\n"
        
        return trend_info
    
    return [
        Tool(
            name="paper_search",
            func=paper_search_tool,
            description="Search for academic papers on a specific topic"
        ),
        Tool(
            name="trend_analysis", 
            func=trend_analysis_tool,
            description="Analyze research publication trends over time"
        )
    ]

def create_research_agent(api_key):
    """Create the research agent"""
    if not api_key:
        return None
    
    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=api_key,
        model="gpt-4o-mini"  # Cost-effective for research tasks
    )
    
    tools = create_research_tools()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research literature analysis agent. Your role is to:
        1. Search for relevant academic papers
        2. Analyze research trends and patterns
        3. Summarize key findings and insights
        4. Identify research gaps and opportunities
        5. Provide structured, comprehensive reports
        
        Always be thorough, accurate, and cite your sources appropriately."""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Main interface
st.title("🔬 Research Literature Analysis Agent")
st.write("An AI agent that helps you analyze academic literature, identify trends, and discover research opportunities.")

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    research_query = st.text_input(
        "Enter your research question or topic:",
        placeholder="e.g., 'machine learning in healthcare', 'climate change impacts on agriculture'"
    )

with col2:
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Literature Review", "Trend Analysis", "Gap Analysis", "Comparative Study"]
    )

if st.button("🚀 Start Research", type="primary"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not research_query:
        st.error("Please enter a research query.")
    else:
        # Create agent
        agent = create_research_agent(openai_api_key)
        
        if agent:
            with st.spinner("🔍 Researching... This may take a few minutes."):
                try:
                    # Customize prompt based on analysis type
                    if analysis_type == "Literature Review":
                        prompt = f"Conduct a comprehensive literature review on '{research_query}'. Search for papers, summarize key findings, and identify major themes."
                    elif analysis_type == "Trend Analysis":
                        prompt = f"Analyze research trends for '{research_query}'. Use both paper search and trend analysis tools to identify patterns over time."
                    elif analysis_type == "Gap Analysis":
                        prompt = f"Identify research gaps in '{research_query}'. Find existing papers and analyze what areas need more investigation."
                    else:  # Comparative Study
                        prompt = f"Compare different approaches or methodologies in '{research_query}' research. Analyze various papers and highlight differences."
                    
                    # Run the agent
                    result = agent.invoke({"input": prompt})
                    
                    # Display results
                    st.success("✅ Research Complete!")
                    st.markdown("## 📊 Analysis Results")
                    st.markdown(result["output"])
                    
                    # Add download option
                    st.download_button(
                        label="📄 Download Research Report",
                        data=result["output"],
                        file_name=f"research_report_{research_query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during research: {e}")
                    st.write("Please check your API key and try again.")

# Instructions section
with st.expander("ℹ️ How to Use This Tool"):
    st.markdown("""
    ### Getting Started:
    1. **Add your OpenAI API Key** in the sidebar (get one at openai.com)
    2. **Enter your research topic** - be specific for better results
    3. **Choose analysis type** - different approaches for different needs
    4. **Click Start Research** and wait for the analysis
    
    ### Example Queries:
    - "Deep learning applications in medical imaging"
    - "Sustainable energy storage solutions"
    - "Natural language processing for financial analysis"
    - "Climate change adaptation strategies"
    
    ### Analysis Types:
    - **Literature Review**: Comprehensive overview of existing research
    - **Trend Analysis**: Publication patterns and temporal trends
    - **Gap Analysis**: Identifies under-researched areas
    - **Comparative Study**: Compares different approaches/methods
    """)

# Footer
st.markdown("---")
st.markdown("*Built with LangChain, OpenAI, and Streamlit. Deploy on Streamlit Cloud for free!*")
