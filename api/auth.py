"""Password auth for dashboard — day-scoped token, no DB required."""

import os
import json
import hashlib
from datetime import datetime, timedelta


def verify_password(provided: str) -> bool:
    expected = os.getenv("DASHBOARD_PASSWORD", "")
    return bool(expected) and (
        hashlib.sha256(provided.encode()).hexdigest() ==
        hashlib.sha256(expected.encode()).hexdigest()
    )


def make_token(password: str) -> str:
    base = f"{password}{datetime.utcnow().strftime('%Y-%m-%d')}"
    return hashlib.sha256(base.encode()).hexdigest()


def handler(request):
    try:
        body = request.body if hasattr(request, "body") else b""
        data = json.loads(body) if body else {}
    except Exception:
        data = {}

    password = data.get("password", "")
    if verify_password(password):
        token = make_token(password)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "token":   token,
                "expires": str(datetime.utcnow() + timedelta(hours=24)),
            }),
        }
    return {
        "statusCode": 401,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": "Invalid password"}),
    }
