import gzip
import html
import io
import json
import re
import urllib.error
import urllib.request
from urllib.parse import quote

from build_kb import splitter

API = "https://api.stackexchange.com/2.3"


def api_get(url):
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    if response.headers.get("Content-Encoding") == "gzip":
        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    return json.loads(raw)


def strip_html(text):
    # drop the tags but keep the newlines so code blocks stay readable
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def search_items(query, pages):
    search_url = (
        f"{API}/search/advanced?order=desc&sort=votes&site=stackoverflow"
        f"&pagesize={pages}&q={quote(query)}&filter=withbody"
    )
    return api_get(search_url)["items"]


def fetch_answers(items):
    # one call for all question ids; the API returns them sorted by score
    ids = ";".join(str(item["question_id"]) for item in items)
    url = (
        f"{API}/questions/{ids}/answers?order=desc&sort=votes"
        f"&site=stackoverflow&pagesize=30&filter=withbody"
    )
    top = {}
    for answer in api_get(url)["items"]:
        top.setdefault(answer["question_id"], answer["body"])
    return top


def fetch_pages(query, pages=5):
    # Stack Overflow needs every word to match, so one long sentence finds
    # nothing; try several keyword windows and merge what comes back.
    words = re.sub(r"[^A-Za-z0-9_ ]", " ", query.lower()).split()
    seen, items = set(), []
    for start, end in [(0, 10), (0, 6), (0, 4), (2, 8), (4, 10),
                       (len(words) - 8, len(words))]:
        chunk_words = words[max(start, 0):end]
        if len(chunk_words) < 3:
            continue
        try:
            found = search_items(" ".join(chunk_words), pages)
        except urllib.error.HTTPError:
            continue
        for item in found:
            if item["question_id"] not in seen:
                seen.add(item["question_id"])
                items.append(item)
        if len(items) >= pages:
            break
    items = items[:pages]
    answers = fetch_answers(items)
    results = []
    for item in items:
        body = answers.get(item["question_id"]) or item["body"]
        results.append({
            "title": html.unescape(item["title"]),
            "url": item["link"],
            "text": strip_html(body),
        })
    return results


def search_stack_overflow(query, k=3):
    documents, sources = [], []
    for page in fetch_pages(query, pages=max(k, 5)):
        chunks = splitter.split_text(f"{page['title']}\n\n{page['text']}")
        if len(chunks[0]) < 150:
            continue  # answer too short to be of any use
        documents.append(chunks[0])
        sources.append({
            "source": "stackoverflow",
            "problem": page["title"],
            "url": page["url"],
            "chunk": 0,
        })
        if len(documents) == k:
            break
    return documents, sources
