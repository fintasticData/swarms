import streamlit as st
import os
from crewai import Crew, MasterAgent, Agent
from crewai_tools import (
    FileReadTool,
    APITestTool,
    WebsiteSearchTool,
    CSVAnalysisTool,
    PDFExtractionTool
)
from google.generativeai import GenerativeModel, configure  # Gemini API

# Load Gemini API key from environment file
from dotenv import load_dotenv
load_dotenv()  # Load .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
configure(api_key=GEMINI_API_KEY)
gemini_model = GenerativeModel("gemini-pro")  # Use the Gemini Pro model

# Define custom tools package suggestions
TOOL_PACKS = {
    "basic": [FileReadTool(), CSVAnalysisTool()],
    "web": [WebsiteSearchTool(), APITestTool()],
    "data": [PDFExtractionTool(), CSVAnalysisTool()],
    "full": [FileReadTool(), WebsiteSearchTool(), APITestTool(), CSVAnalysisTool(), PDFExtractionTool()]
}

class CommandCenter(MasterAgent):
    def __init__(self, name="CommandCenter", **kwargs):
        super().__init__(name=name, **kwargs)
        self.swarm = []
        self.tool_pack = []

    def setup_swarm(self, tool_pack_name="basic"):
        self.swarm = [
            DataAnalysisAgent(name="DataAgent", tools=TOOL_PACKS.get(tool_pack_name, [])),
            WebScrapingAgent(name="ScraperAgent", tools=TOOL_PACKS.get(tool_pack_name, [])),
            APIIntegrationAgent(name="APIAgent", tools=TOOL_PACKS.get(tool_pack_name, [])),
            CodeGenerationAgent(name="CodeGenAgent")  # Add the Code Generation Agent
        ]

class DataAnalysisAgent(Agent):
    def execute(self, task):
        if self.tools:
            return f"{self.name} used {self.tools[0].name} to analyze: {task}"
        return f"{self.name} completed analysis: {task}"

class WebScrapingAgent(Agent):
    def execute(self, task):
        if self.tools:
            return f"{self.name} used {self.tools[1].name} to scrape: {task}"
        return f"{self.name} scraped: {task}"

class APIIntegrationAgent(Agent):
    def execute(self, task):
        if self.tools:
            return f"{self.name} used {self.tools[2].name} to integrate: {task}"
        return f"{self.name} integrated: {task}"

class CodeGenerationAgent(Agent):
    def execute(self, task):
        try:
            # Use Gemini API to generate code
            response = gemini_model.generate_content(f"Generate Python code for: {task}")
            return f"{self.name} generated code:\n```python\n{response.text}\n```"
        except Exception as e:
            return f"{self.name} failed to generate code: {str(e)}"

# Streamlit App
def main():
    st.title("CrewAI Master-Swarm Orchestration with Code Generation")

    # Initialize session state
    if 'crew' not in st.session_state:
        st.session_state.crew = Crew()
        st.session_state.command_center = CommandCenter()
        st.session_state.results = None

    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        tool_pack = st.selectbox("Select Tool Package", list(TOOL_PACKS.keys()))
        st.session_state.command_center.setup_swarm(tool_pack)

    # Tabs for Agent Setup and Workflow Execution
    tab1, tab2, tab3 = st.tabs(["Agent Setup", "Task Execution", "Results"])

    with tab1:
        st.subheader("Agent Setup")
        st.write("### Current Agents in Swarm")
        for agent in st.session_state.command_center.swarm:
            with st.expander(f"Agent: {agent.name}"):
                st.write(f"**Role:** {agent.__class__.__name__}")
                if hasattr(agent, "tools") and agent.tools:
                    st.write("**Tools:**")
                    for tool in agent.tools:
                        st.write(f"- {tool.name}")
                else:
                    st.write("**Tools:** None")

    with tab2:
        st.subheader("Task Execution")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            data_task = st.text_input("Data Agent Task", "Analyze sales data")

        with col2:
            scrape_task = st.text_input("Scraper Agent Task", "Scrape competitor prices")

        with col3:
            api_task = st.text_input("API Agent Task", "Fetch weather data")

        with col4:
            code_task = st.text_input("CodeGen Agent Task", "Generate a Python function to calculate factorial")

        if st.button("Execute Workflow"):
            tasks = {
                "DataAgent": data_task,
                "ScraperAgent": scrape_task,
                "APIAgent": api_task,
                "CodeGenAgent": code_task
            }

            with st.spinner("Orchestrating swarm agents..."):
                st.session_state.results = st.session_state.command_center.run_workflow(tasks)

            st.success("Workflow Completed!")

    with tab3:
        st.subheader("Results")
        if st.session_state.results:
            for agent, result in st.session_state.results.items():
                with st.expander(f"{agent} Report"):
                    st.write(result)
        else:
            st.write("No results yet. Run the workflow in the 'Task Execution' tab.")

if __name__ == "__main__":
    main()
