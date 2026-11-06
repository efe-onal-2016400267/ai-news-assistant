import json
import sys

from ai_news_assistant.agent.tools import (
    create_get_article_tool,
    create_get_latest_articles_tool,
    create_search_tool,
)
from ai_news_assistant.rag.config import DEFAULT_DB

tools = [
    create_search_tool(retriever=None, store_path=DEFAULT_DB),
    create_get_latest_articles_tool(store_path=DEFAULT_DB),
    create_get_article_tool(store_path=DEFAULT_DB),
]

snapshot = {t.name: {"descrition": t.description, "args": t.args} for t in tools}

with open(sys.argv[1], "w") as f:
    json.dump(snapshot, f, indent=2, sort_keys=True)
