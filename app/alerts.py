import httpx
from app.config import settings


async def dispatch_webhook_alert(title: str, details: str) -> None:
    if not settings.alert_webhook_url:
        return

    payload = {
        "content": f"🚨 **PulseGuard AI — {title}**\n{details}",
        "text": f"🚨 PulseGuard AI — {title}: {details}",
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            await client.post(settings.alert_webhook_url, json=payload)
    except Exception:
        # Never block telemetry ingestion if an external webhook fails
        pass