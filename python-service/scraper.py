from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/153.0 Safari/537.36"
    )
}

TIMEOUT = 15

ABOUT_KEYWORDS = (
    "about",
    "about-us",
    "company",
    "who-we-are",
    "our-story",
)


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
        "form",
    ]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    return text[:30000]


def fetch_page(url: str) -> tuple[str, str]:
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT,
        allow_redirects=True,
    )

    response.raise_for_status()

    return response.url, response.text


def find_about_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    base_domain = urlparse(base_url).netloc

    links: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()

        if not href:
            continue

        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        if parsed.netloc != base_domain:
            continue

        combined = (
            f"{anchor.get_text(' ', strip=True)} "
            f"{parsed.path}"
        ).lower()

        if any(keyword in combined for keyword in ABOUT_KEYWORDS):
            links.append(absolute)

    return list(dict.fromkeys(links))


def scrape_company(domain: str) -> dict:
    if not domain.startswith("http"):
        domain = f"https://{domain}"

    candidates = [
        domain,
        domain.rstrip("/") + "/about",
        domain.rstrip("/") + "/about-us",
        domain.rstrip("/") + "/company",
        domain.rstrip("/") + "/who-we-are",
    ]

    visited = set()
    pages = []

    for url in candidates:
        try:
            final_url, html = fetch_page(url)

            if final_url in visited:
                continue

            visited.add(final_url)

            text = clean_text(html)

            if len(text) >= 300:
                pages.append({
                    "url": final_url,
                    "text": text,
                })

            discovered = find_about_links(final_url, html)

            for link in discovered[:3]:
                try:
                    final_link, link_html = fetch_page(link)

                    if final_link in visited:
                        continue

                    visited.add(final_link)

                    link_text = clean_text(link_html)

                    if len(link_text) >= 300:
                        pages.append({
                            "url": final_link,
                            "text": link_text,
                        })

                except requests.RequestException:
                    continue

        except requests.RequestException:
            continue

    if not pages:
        return {
            "status": "FAILED",
            "website": domain,
            "pages": [],
            "text": "",
        }

    combined = "\n\n".join(
        f"SOURCE: {page['url']}\n{page['text']}"
        for page in pages
    )

    return {
        "status": "SUCCESS",
        "website": domain,
        "pages": [page["url"] for page in pages],
        "text": combined[:60000],
    }