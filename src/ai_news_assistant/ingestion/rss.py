import httpx

USER_AGENT = "ai-news-assistant/0.1"

def download_feed(url: str, client: httpx.Client | None = None, *, timeout: float = 15.0) -> bytes:
    """Download a feed and return the raw XML bytes."""

    # To be able to do offline tests with httpx.MockTransport.
    # Tests will call with mock transport, production will call with None as client.
    if client is None:
        with httpx.Client() as own_client:
            return download_feed(url, own_client, timeout=timeout)
        
    response = client.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )

    response.raise_for_status()
    return response.content