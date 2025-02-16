from duckduckgo_search import DDGS
from .config import Config
from urllib.parse import urlparse

class KenyanWebSearch:
    def __init__(self):
        self.ddgs = DDGS()

    def is_kenyan_domain(self, url):
        domain = urlparse(url).netloc.lower()
        return any(domain.endswith(tld) for tld in Config.KENYA_DOMAINS)

    def search(self, query, max_results=5):
        results = []
        for r in self.ddgs.text(query, max_results=20):
            if self.is_kenyan_domain(r['link']) and len(results) < max_results:
                results.append({
                    'title': r['title'],
                    'link': r['link'],
                    'snippet': r['body']
                })
        return results 