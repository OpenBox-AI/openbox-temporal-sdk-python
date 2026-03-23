# openbox/context_propagation.py
"""Ensure OTel context propagates across thread boundaries.

NOT sandbox-safe — do NOT import from workflow_interceptor.py.

When async activities call loop.run_in_executor(), Python < 3.12 does NOT
copy ContextVars into the executor thread. This breaks OTel trace propagation
because the httpx/requests instrumentor creates spans with trace_id=0 in the
new thread, and hook_governance can't find the activity context.

This module patches the asyncio event loop's default executor to automatically
copy ContextVars (including OTel trace context) into executor threads — the
same approach used by the DeepAgent SDK's _run_async().
"""

import concurrent.futures
import contextvars
import functools
import logging

logger = logging.getLogger(__name__)


class ContextPropagatingExecutor(concurrent.futures.ThreadPoolExecutor):
    """ThreadPoolExecutor that copies ContextVars into spawned threads.

    Wraps submitted callables with contextvars.copy_context().run() so that
    OTel trace context, Temporal activity context, and other ContextVars
    propagate correctly across thread boundaries.
    """

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, functools.partial(fn, *args, **kwargs))


def install_context_propagating_executor(max_workers: int = 32) -> None:
    """Install a ContextVar-propagating executor as the asyncio default.

    After this call, loop.run_in_executor(None, fn) will automatically
    propagate OTel trace context into the executor thread.

    Only patches the default executor (None). Users passing an explicit
    executor to run_in_executor are unaffected.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("[OpenBox] No running event loop — skipping executor patch")
        return
    loop.set_default_executor(ContextPropagatingExecutor(max_workers=max_workers))
    logger.info("[OpenBox] Installed context-propagating executor for OTel trace propagation")
