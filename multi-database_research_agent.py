import streamlit as st
import pandas as pd
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
import arxiv
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from scholarly import scholarly
import time
import json

# Page config
st.set_page_config(
    page_title="Research Literature Agent",
    page_icon="🔬",
    layout="wide"
)

# Sidebar for configuration
st.sidebar.title("🔧 Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Database selection
st.sidebar.markdown("### 📚 Research Database")
database_options = {
    "ArXiv": "Academic preprints (Physics, Math, CS, etc.)",
    "Google Scholar": "Broad academic search across disciplines", 
    "PubMed": "Medical and life sciences literature",
    "Semantic Scholar": "AI-powered academic search",
    "CrossRef": "Academic publications with DOIs"
}

selected_db = st.sidebar.selectbox(
    "Choose Research Database:",
    options=list(database_options.keys()),
    help="Different databases specialize in different fields"
)

st.sidebar.info(f"**{selected_db}**: {database_options[selected_db]}")

max_papers = st.sidebar.slider("Max papers to analyze", 5, 50, 10)
years_back = st.sidebar.slider("Years to look back", 1, 10, 3)

def search_arxiv_papers(query, max_results=10):
    """Search arXiv for papers"""
    try:
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
                'categories': paper.categories,
                'source': 'ArXiv'
            })
        
        return papers
    except Exception as e:
        st.error(f"ArXiv search error: {e}")
        return []

def search_google_scholar(query, max_results=10):
    """Search Google Scholar for papers"""
    try:
        papers = []
        search_query = scholarly.search_pubs(query)
        
        count = 0
        for paper in search_query:
            if count >= max_results:
                break
                
            # Get detailed info
            try:
                filled_paper = scholarly.fill(paper)
                papers.append({
                    'title': filled_paper.get('title', 'No title'),
                    'authors': [author['name'] for author in filled_paper.get('author', [])],
                    'published': str(filled_paper.get('pub_year', 'Unknown')),
                    'summary': filled_paper.get('abstract', 'No abstract available'),
                    'url': filled_paper.get('pub_url', ''),
                    'categories': [filled_paper.get('venue', 'Unknown venue')],
                    'source': 'Google Scholar',
                    'citations': filled_paper.get('num_citations', 0)
                })
                count += 1
                time.sleep(1)  # Rate limiting
            except Exception as e:
                continue
                
        return papers
    except Exception as e:
        st.error(f"Google Scholar search error: {e}")
        return []

def search_pubmed_papers(query, max_results=10):
    """Search PubMed via NCBI E-utilities API"""
    try:
        # Search for paper IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json'
        }
        
        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()
        
        if 'esearchresult' not in search_data or not search_data['esearchresult']['idlist']:
            return []
        
        # Get paper details
        ids = search_data['esearchresult']['idlist']
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'xml'
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        
        # Simple XML parsing (you might want to use xml.etree.ElementTree for production)
        papers = []
        # This is a simplified version - in production you'd parse the XML properly
        for i, paper_id in enumerate(ids[:5]):  # Limit to first 5 for demo
            papers.append({
                'title': f"PubMed Paper {i+1} (ID: {paper_id})",
                'authors': ['Author data requires XML parsing'],
                'published': 'Date requires XML parsing',
                'summary': 'Abstract requires XML parsing - see PubMed for full details',
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
                'categories': ['PubMed'],
                'source': 'PubMed'
            })
        
        return papers
    except Exception as e:
        st.error(f"PubMed search error: {e}")
        return []

def search_semantic_scholar(query, max_results=10):
    """Search Semantic Scholar API"""
    try:
        api_url = "http://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,authors,year,abstract,url,citationCount,venue'
        }
        
        response = requests.get(api_url, params=params)
        data = response.json()
        
        papers = []
        for paper in data.get('data', []):
            papers.append({
                'title': paper.get('title', 'No title'),
                'authors': [author.get('name', 'Unknown') for author in paper.get('authors', [])],
                'published': str(paper.get('year', 'Unknown')),
                'summary': paper.get('abstract', 'No abstract available'),
                'url': paper.get('url', ''),
                'categories': [paper.get('venue', 'Unknown venue')],
                'source': 'Semantic Scholar',
                'citations': paper.get('citationCount', 0)
            })
        
        return papers
    except Exception as e:
        st.error(f"Semantic Scholar search error: {e}")
        return []

def search_crossref_papers(query, max_results=10):
    """Search CrossRef API"""
    try:
        api_url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': max_results,
            'select': 'title,author,published-print,abstract,URL,container-title'
        }
        
        headers = {'User-Agent': 'Research Agent (mailto:your-email@example.com)'}
        response = requests.get(api_url, params=params, headers=headers)
        data = response.json()
        
        papers = []
        for item in data.get('message', {}).get('items', []):
            # Extract publication date
            pub_date = 'Unknown'
            if 'published-print' in item:
                date_parts = item['published-print'].get('date-parts', [[]])[0]
                if date_parts:
                    pub_date = str(date_parts[0])  # Year
            
            papers.append({
                'title': item.get('title', ['No title'])[0],
                'authors': [f"{author.get('given', '')} {author.get('family', '')}" 
                          for author in item.get('author', [])],
                'published': pub_date,
                'summary': item.get('abstract', 'No abstract available'),
                'url': item.get('URL', ''),
                'categories': item.get('container-title', ['Unknown journal']),
                'source': 'CrossRef'
            })
        
        return papers
    except Exception as e:
        st.error(f"CrossRef search error: {e}")
        return []

def search_papers_by_database(database, query, max_results=10):
    """Route search to appropriate database function"""
    if database == "ArXiv":
        return search_arxiv_papers(query, max_results)
    elif database == "Google Scholar":
        return search_google_scholar(query, max_results)
    elif database == "PubMed":
        return search_pubmed_papers(query, max_results)
    elif database == "Semantic Scholar":
        return search_semantic_scholar(query, max_results)
    elif database == "CrossRef":
        return search_crossref_papers(query, max_results)
    else:
        return []

def create_research_tools(selected_database):
    """Create tools for the research agent based on selected database"""
    
    def paper_search_tool(query: str) -> str:
        """Search for academic papers on a given topic using selected database"""
        papers = search_papers_by_database(selected_database, query, max_papers)
        if not papers:
            return f"No papers found for '{query}' in {selected_database}."
        
        result = f"Found {len(papers)} papers on '{query}' from {selected_database}:\n\n"
        for i, paper in enumerate(papers, 1):
            result += f"{i}. **{paper['title']}**\n"
            result += f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n"
            result += f"   Published: {paper['published']}\n"
            result += f"   Source: {paper['source']}\n"
            result += f"   Summary: {paper['summary'][:200]}...\n"
            if paper.get('citations'):
                result += f"   Citations: {paper['citations']}\n"
            result += f"   URL: {paper['url']}\n\n"
        
        return result
    
    def trend_analysis_tool(query: str) -> str:
        """Analyze research trends over time using selected database"""
        papers = search_papers_by_database(selected_database, query, max_papers * 2)
        if not papers:
            return f"No papers found for trend analysis in {selected_database}."
        
        # Group by year
        df = pd.DataFrame(papers)
        
        # Handle different date formats
        years = []
        for date_str in df['published']:
            try:
                if isinstance(date_str, str) and len(date_str) >= 4:
                    year = int(date_str[:4])
                    years.append(year)
                else:
                    years.append(None)
            except:
                years.append(None)
        
        df['year'] = years
        df = df.dropna(subset=['year'])
        
        if df.empty:
            return "Could not extract publication years for trend analysis."
        
        yearly_counts = df.groupby('year').size().reset_index(name='count')
        
        trend_info = f"Research trend analysis for '{query}' in {selected_database}:\n"
        trend_info += f"Total papers analyzed: {len(papers)}\n"
        trend_info += f"Year range: {int(yearly_counts['year'].min())} - {int(yearly_counts['year'].max())}\n"
        trend_info += f"Peak year: {int(yearly_counts.loc[yearly_counts['count'].idxmax(), 'year'])} ({yearly_counts['count'].max()} papers)\n"
        
        return trend_info
    
    def database_comparison_tool(query: str) -> str:
        """Compare results across different databases"""
        results = {}
        databases = ["ArXiv", "Semantic Scholar", "CrossRef"]
        
        for db in databases:
            try:
                papers = search_papers_by_database(db, query, 5)
                results[db] = len(papers)
            except:
                results[db] = 0
        
        comparison = f"Database comparison for '{query}':\n"
        for db, count in results.items():
            comparison += f"- {db}: {count} papers found\n"
        
        return comparison
    
    return [
        Tool(
            name="paper_search",
            func=paper_search_tool,
            description=f"Search for academic papers using {selected_database}"
        ),
        Tool(
            name="trend_analysis", 
            func=trend_analysis_tool,
            description=f"Analyze research publication trends using {selected_database}"
        ),
        Tool(
            name="database_comparison",
            func=database_comparison_tool,
            description="Compare search results across multiple databases"
        )
    ]

def create_research_agent(api_key, selected_database):
    """Create the research agent with database-specific tools"""
    if not api_key:
        return None
    
    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=api_key,
        model="gpt-4o-mini"  # Cost-effective for research tasks
    )
    
    tools = create_research_tools(selected_database)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a research literature analysis agent specialized in {selected_database}. Your role is to:
        1. Search for relevant academic papers using {selected_database}
        2. Analyze research trends and patterns
        3. Summarize key findings and insights
        4. Identify research gaps and opportunities
        5. Compare findings across different databases when requested
        6. Provide structured, comprehensive reports
        
        Always mention the source database ({selected_database}) in your analysis and be thorough, accurate, and cite your sources appropriately."""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Main interface
st.title("🔬 Multi-Database Research Literature Agent")
st.write(f"An AI agent that analyzes academic literature from **{selected_db}** and other research databases.")

# Database info display
st.info(f"🎯 **Currently using:** {selected_db} - {database_options[selected_db]}")

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
        ["Literature Review", "Trend Analysis", "Gap Analysis", "Database Comparison"]
    )

if st.button("🚀 Start Research", type="primary"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not research_query:
        st.error("Please enter a research query.")
    else:
        # Create agent
        agent = create_research_agent(openai_api_key, selected_db)
        
        if agent:
            with st.spinner(f"🔍 Researching using {selected_db}... This may take a few minutes."):
                try:
                    # Customize prompt based on analysis type
                    if analysis_type == "Literature Review":
                        prompt = f"Conduct a comprehensive literature review on '{research_query}' using {selected_db}. Search for papers, summarize key findings, and identify major themes."
                    elif analysis_type == "Trend Analysis":
                        prompt = f"Analyze research trends for '{research_query}' using {selected_db}. Use both paper search and trend analysis tools to identify patterns over time."
                    elif analysis_type == "Gap Analysis":
                        prompt = f"Identify research gaps in '{research_query}' using {selected_db}. Find existing papers and analyze what areas need more investigation."
                    else:  # Database Comparison
                        prompt = f"Compare research findings for '{research_query}' across different databases. Use the database comparison tool and provide insights on coverage differences."
                    
                    # Run the agent
                    result = agent.invoke({"input": prompt})
                    
                    # Display results
                    st.success(f"✅ Research Complete using {selected_db}!")
                    st.markdown("## 📊 Analysis Results")
                    st.markdown(result["output"])
                    
                    # Add download option
                    report_content = f"Research Report: {research_query}\n"
                    report_content += f"Database: {selected_db}\n"
                    report_content += f"Analysis Type: {analysis_type}\n"
                    report_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    report_content += result["output"]
                    
                    st.download_button(
                        label="📄 Download Research Report",
                        data=report_content,
                        file_name=f"research_report_{selected_db.lower()}_{research_query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during research: {e}")
                    st.write("Please check your API key and try again.")

# Database-specific tips
with st.expander(f"💡 Tips for {selected_db}"):
    if selected_db == "ArXiv":
        st.markdown("""
        **ArXiv Tips:**
        - Best for: Physics, Mathematics, Computer Science, Quantitative Biology
        - Search terms: Use specific technical terms
        - Example: "deep learning computer vision", "quantum computing algorithms"
        """)
    elif selected_db == "Google Scholar":
        st.markdown("""
        **Google Scholar Tips:**
        - Best for: Broad academic coverage across all disciplines
        - Search terms: Use natural language or specific phrases
        - Example: "machine learning healthcare applications", "climate change agriculture"
        """)
    elif selected_db == "PubMed":
        st.markdown("""
        **PubMed Tips:**
        - Best for: Medical and life sciences research
        - Search terms: Use MeSH terms or medical terminology
        - Example: "diabetes treatment", "cancer immunotherapy"
        """)
    elif selected_db == "Semantic Scholar":
        st.markdown("""
        **Semantic Scholar Tips:**
        - Best for: AI-powered search with citation analysis
        - Search terms: Technical terms work well
        - Example: "natural language processing", "computer vision"
        """)
    elif selected_db == "CrossRef":
        st.markdown("""
        **CrossRef Tips:**
        - Best for: Peer-reviewed academic publications with DOIs
        - Search terms: Use formal academic language
        - Example: "machine learning applications", "climate change mitigation"
        """)

# Instructions section
with st.expander("ℹ️ How to Use This Multi-Database Tool"):
    st.markdown(f"""
    ### Getting Started:
    {chr(10)} 1. **Select your database** from the sidebar dropdown"
    {chr(10)} 2. **Add your OpenAI API Key** in the sidebar
    {chr(10)} 3. **Enter your research topic** - tailor it to your chosen database
    {chr(10)} 4. **Choose analysis type** - different approaches for different needs
    {chr(10)} 5. **Click Start Research** and wait for the analysis
    
    ### Available Databases:
    {chr(10)}
    {chr(10).join([f"- **{db}**: {desc}" for db, desc in database_options.items()])}
    
    ### Analysis Types:
    - **Literature Review**: Comprehensive overview using selected database
    - **Trend Analysis**: Publication patterns over time
    - **Gap Analysis**: Identifies under-researched areas
    - **Database Comparison**: Compares coverage across databases
    
    ### Pro Tips:
    - Start with ArXiv for technical/CS topics
    - Use PubMed for medical research
    - Try Google Scholar for broad interdisciplinary topics
    - Use Database Comparison to see coverage differences
    """)

# Footer
st.markdown("---")
st.markdown("*Built with LangChain, OpenAI, and Streamlit. Deploy on Streamlit Cloud for free!*")
