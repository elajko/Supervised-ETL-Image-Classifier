from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from app import db
from app.models import SourceCredentialsRequest, SourceStatus
from app.sources.registry import ADAPTERS, get_adapter

router = APIRouter(prefix="/api/sources")


async def _status_for(adapter) -> SourceStatus:
    configured = await adapter.is_configured()
    authenticated = await adapter.is_authenticated() if configured else False
    return SourceStatus(
        name=adapter.name,
        domains=list(adapter.domains),
        needs_client_secret=adapter.needs_client_secret,
        supports_interactive_auth=adapter.supports_interactive_auth,
        configured=configured,
        authenticated=authenticated,
    )


@router.get("", response_model=list[SourceStatus])
async def list_sources() -> list[SourceStatus]:
    return [await _status_for(adapter) for adapter in ADAPTERS]


@router.get("/{site}/status", response_model=SourceStatus)
async def source_status(site: str) -> SourceStatus:
    adapter = get_adapter(site)
    if adapter is None:
        raise HTTPException(status_code=404, detail="unknown source")
    return await _status_for(adapter)


@router.post("/{site}/credentials")
async def set_credentials(site: str, req: SourceCredentialsRequest) -> dict:
    adapter = get_adapter(site)
    if adapter is None:
        raise HTTPException(status_code=404, detail="unknown source")
    client_id = req.client_id.strip()
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id must not be blank")
    if adapter.needs_client_secret and not (req.client_secret or "").strip():
        raise HTTPException(status_code=400, detail=f"{site} requires a client secret")
    await db.set_source_credentials(site, client_id, (req.client_secret or "").strip() or None)
    return {"status": "ok"}


@router.get("/{site}/auth-url")
async def auth_url(site: str, request: Request) -> dict:
    adapter = get_adapter(site)
    if adapter is None:
        raise HTTPException(status_code=404, detail="unknown source")
    if not adapter.supports_interactive_auth:
        raise HTTPException(status_code=400, detail=f"{site} does not use interactive login")
    creds = await adapter.get_credentials()
    if not creds:
        raise HTTPException(status_code=400, detail=f"{site} credentials are not configured yet")
    redirect_uri = str(request.base_url).rstrip("/") + f"/api/sources/{site}/callback"
    return {"url": adapter.get_auth_url(redirect_uri, creds)}


@router.get("/{site}/callback", response_class=HTMLResponse)
async def callback(site: str, request: Request) -> HTMLResponse:
    adapter = get_adapter(site)
    if adapter is None:
        return HTMLResponse("<p>Unknown source.</p>", status_code=404)

    code = request.query_params.get("code")
    error = request.query_params.get("error")
    if error or not code:
        return HTMLResponse(f"<p>Authentication failed: {error or 'no code returned'}. You can close this tab.</p>")

    redirect_uri = str(request.base_url).rstrip("/") + f"/api/sources/{site}/callback"
    try:
        await adapter.handle_callback(code, redirect_uri)
    except Exception as e:
        return HTMLResponse(f"<p>Authentication failed: {e}. You can close this tab.</p>")
    return HTMLResponse("<p>Authenticated! You can close this tab and return to the app.</p>")
