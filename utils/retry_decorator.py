# utils/retry_decorator.py
import asyncio
import functools
from typing import Any, Callable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog

logger = structlog.get_logger()

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    multiplier: float = 2.0
):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(
                multiplier=multiplier,
                min=initial_wait,
                max=max_wait
            ),
            retry=retry_if_exception_type((Exception,)),
            reraise=True
        )
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning("Function retry needed", 
                             function=func.__name__, 
                             error=str(e))
                raise
        
        return wrapper
    return decorator
