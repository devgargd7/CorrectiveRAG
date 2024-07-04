from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

class AgentHelper:

  def __init__(self, model, yaml_config_reader):
     self.yaml_config_reader = yaml_config_reader 
     self.model = model 

  def get_llm(self, model, format="json", temperature=0):
      return ChatOllama(model=model, format=format, temperature=temperature)

  def create_agent(self, prompt_template, output_parser):
      llm = self.get_llm(self.model, format="json" if isinstance(output_parser, JsonOutputParser) else None)
      return prompt_template | llm | output_parser

  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

  def get_agents(self):
    # Retrieval Grader
    retrieval_grader = self.create_agent(
        PromptTemplate(
          template = self.yaml_config_reader.get_prompt_template("retrieval_grader"),
          input_variables=["question", "document"],
        ), JsonOutputParser()
      )

    ### Generate
    # Chain
    rag_chain = self.create_agent(
        PromptTemplate(
          template = self.yaml_config_reader.get_prompt_template("rag_chain"),
          input_variables=["question", "document"],
        ), StrOutputParser()
      )

    ### Hallucination Grader
    hallucination_grader = self.create_agent(
        PromptTemplate(
          template = self.yaml_config_reader.get_prompt_template("hallucination_grader"),
          input_variables=["generation", "documents"],
        ), JsonOutputParser()
      )

    ### Answer Grader
    answer_grader = self.create_agent(
        PromptTemplate(
          template = self.yaml_config_reader.get_prompt_template("answer_grader"),
          input_variables=["generation", "documents"],
        ), JsonOutputParser()
      )

    ### Router
    question_router = self.create_agent(
        PromptTemplate(
          template = self.yaml_config_reader.get_prompt_template("question_router"),
          input_variables=["question"],
        ), JsonOutputParser()
      )

    ### Search
    web_search_tool = TavilySearchResults(k=3)

    return retrieval_grader, rag_chain, hallucination_grader, answer_grader, question_router, web_search_tool