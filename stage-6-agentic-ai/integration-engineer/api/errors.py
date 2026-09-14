"""Centralized structured error handlers and exceptions."""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse


class AADAError(Exception):
    def __init__(self, code: str, message: str, run_id: str = None, status_code: int = 400):
        self.code = code
        self.message = message
        self.run_id = run_id
        self.status_code = status_code
        super().__init__(message)


def aada_error_handler(request: Request, exc: AADAError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "run_id": exc.run_id,
            }
        },
    )


def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "run_id": None,
            }
        },
    )


def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Safe response: never leak internal stack traces to users
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during execution.",
                "run_id": None,
            }
        },
    )
