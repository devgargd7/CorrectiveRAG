import os

class EnvironmentConfig:
  def __init__(self, api_key):
      self.api_key = api_key
      self.set_environment_variables()

  def set_environment_variables(self):
      os.environ["LANGCHAIN_TRACING_V2"] = "true"
      os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
      os.environ["LANGCHAIN_API_KEY"] = self.api_key