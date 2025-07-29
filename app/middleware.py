# app/middleware.py
# app/middleware.py

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("uvicorn.access")  # Use uvicorn's access logger (or configure your own)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # You can log headers, path, method, client info here
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Request from {client_host}: {request.method} {request.url.path}")

        # Process the request
        response: Response = await call_next(request)

        # Measure processing duration
        duration = (time.time() - start_time) * 1000
        logger.info(f"Response status: {response.status_code} completed in {duration:.2f}ms")

        return response

