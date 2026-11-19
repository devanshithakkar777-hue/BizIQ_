"""Small HTTP client (standard library only): timeouts, retries with backoff, no redirects, secrets never in errors."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class Response:
    status: int
    headers: dict = field(default_factory=dict)
    body: bytes = b""

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self):
        try:
            return json.loads(self.body.decode("utf-8", errors="replace"))
        except ValueError as e:
            raise HttpError("The service returned something that is not JSON.", self.status) from e


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):          # never follow: a redirect could carry credentials to another host
        return None


def _urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float):
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, dict(r.headers.items()), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items()), e.read()


class Http:
    """transport(method, url, headers, body, timeout) -> (status, headers, body); replace it in tests."""

    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, transport=None, retries: int = 3, timeout: float = 30, sleep=time.sleep):
        self.transport, self.retries, self.timeout, self.sleep = transport or _urllib_transport, retries, timeout, sleep

    def request(self, method: str, url: str, *, headers: dict | None = None, params: dict | None = None,
                json_body=None, form: dict | None = None, data: bytes | None = None) -> Response:
        if not url.lower().startswith(("https://", "http://")):
            raise HttpError("Only http and https addresses are allowed.")
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        h = {"Accept": "application/json", **(headers or {})}
        body = data
        if json_body is not None:
            body, h["Content-Type"] = json.dumps(json_body).encode(), "application/json"
        elif form is not None:
            body, h["Content-Type"] = urllib.parse.urlencode(form).encode(), "application/x-www-form-urlencoded"
        last = ""
        for attempt in range(self.retries + 1):
            try:
                status, rh, rb = self.transport(method, url, h, body, self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last, status, rh, rb = f"network error ({type(e).__name__})", None, {}, b""
            if status is not None and status not in self.RETRY_STATUS:
                break
            if attempt < self.retries:
                wait = 2 ** attempt
                ra = {k.lower(): v for k, v in (rh or {}).items()}.get("retry-after")
                if ra and str(ra).isdigit():
                    wait = min(int(ra), 60)
                self.sleep(wait)
        else:
            raise HttpError(f"Gave up after {self.retries + 1} tries: {last or f'HTTP {status}'}", status)
        if status is None:
            raise HttpError(last)
        if 300 <= status < 400:
            raise HttpError(f"The service redirected the request (HTTP {status}); check the address you entered.", status)
        if status >= 400:
            snippet = rb.decode("utf-8", errors="replace")[:300].replace("\n", " ")
            hint = {401: " Credentials were rejected or have expired.", 403: " This account lacks permission for that data."}.get(status, "")
            raise HttpError(f"HTTP {status}.{hint} {snippet}".strip(), status)
        return Response(status, rh, rb)
