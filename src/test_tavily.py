import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
response = client.search(query="Bitcoin market news", search_depth="basic", max_results=1)

print("Headline:", response["results"][0]["title"])
print("URL:", response["results"][0]["url"])
