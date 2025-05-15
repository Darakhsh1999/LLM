from datetime import datetime
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun


# DuckDuckGo
web_search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=web_search.run,
    description="Search the web for information"
)


# Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
