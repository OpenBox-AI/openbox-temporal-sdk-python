"""
Quick test to verify whether OTel instrumentor request_hook can pause HTTP requests.

Tests httpx, requests, and urllib3 with a sleep in the request_hook.
If total time >= SLEEP_SECONDS, the hook CAN pause execution.

Run:
    uv run python tests/test_otel_hook_pause.py
"""

import time

import httpx
import requests as requests_lib
import urllib3
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter


SLEEP_SECONDS = 5
TEST_URL = "https://httpbin.org/get"


def _make_hook(lib_name):
    def hook(span, *args, **kwargs):
        print(f"  [{lib_name}] request_hook fired, sleeping {SLEEP_SECONDS}s...")
        time.sleep(SLEEP_SECONDS)
        print(f"  [{lib_name}] sleep done")
    return hook


def _check_result(lib_name, elapsed):
    if elapsed >= SLEEP_SECONDS:
        print(f"  >>> YES - {lib_name} OTel request_hook CAN pause the request ({elapsed:.2f}s)\n")
    else:
        print(f"  >>> NO - {lib_name} OTel request_hook CANNOT pause the request ({elapsed:.2f}s)\n")


def test_httpx(provider):
    print(f"{'='*60}")
    print(f"[1/3] httpx")
    print(f"{'='*60}")

    HTTPXClientInstrumentor().instrument(request_hook=_make_hook("httpx"))
    start = time.time()
    try:
        response = httpx.get(TEST_URL)
        elapsed = time.time() - start
        print(f"  Status: {response.status_code} | Time: {elapsed:.2f}s")
        _check_result("httpx", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e} | Time: {elapsed:.2f}s")
        _check_result("httpx", elapsed)
    finally:
        HTTPXClientInstrumentor().uninstrument()


def test_requests(provider):
    print(f"{'='*60}")
    print(f"[2/3] requests")
    print(f"{'='*60}")

    RequestsInstrumentor().instrument(request_hook=_make_hook("requests"))
    start = time.time()
    try:
        response = requests_lib.get(TEST_URL)
        elapsed = time.time() - start
        print(f"  Status: {response.status_code} | Time: {elapsed:.2f}s")
        _check_result("requests", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e} | Time: {elapsed:.2f}s")
        _check_result("requests", elapsed)
    finally:
        RequestsInstrumentor().uninstrument()


def test_urllib3(provider):
    print(f"{'='*60}")
    print(f"[3/3] urllib3")
    print(f"{'='*60}")

    URLLib3Instrumentor().instrument(request_hook=_make_hook("urllib3"))
    start = time.time()
    try:
        http = urllib3.PoolManager()
        response = http.request("GET", TEST_URL)
        elapsed = time.time() - start
        print(f"  Status: {response.status} | Time: {elapsed:.2f}s")
        _check_result("urllib3", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e} | Time: {elapsed:.2f}s")
        _check_result("urllib3", elapsed)
    finally:
        URLLib3Instrumentor().uninstrument()


def main():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    print(f"\nTesting OTel request_hook pause with {SLEEP_SECONDS}s sleep\n")

    test_httpx(provider)
    test_requests(provider)
    test_urllib3(provider)

    provider.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
