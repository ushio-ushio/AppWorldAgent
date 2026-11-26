"""Official Sign in with ChatGPT, independent of Codex CLI credentials."""
from __future__ import annotations

import base64
from ctypes import byref, create_string_buffer, windll
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import time
from contextlib import contextmanager
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid
import webbrowser

import jwt
from jwt import PyJWKClient
import requests

AUTH = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
DEFAULT_PATH = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AgentLab" / "siwc.dat"


class SIWCError(RuntimeError):
    pass


class _BLOB(__import__("ctypes").Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", __import__("ctypes").POINTER(__import__("ctypes").c_byte))]


def _protect(data: bytes, *, decrypt: bool = False) -> bytes:
    if os.name != "nt":
        raise SIWCError("SIWC credential storage requires Windows DPAPI")
    import ctypes
    source = create_string_buffer(data)
    input_blob = _BLOB(len(data), ctypes.cast(source, ctypes.POINTER(ctypes.c_byte)))
    output_blob = _BLOB()
    function = windll.crypt32.CryptUnprotectData if decrypt else windll.crypt32.CryptProtectData
    if decrypt:
        success = function(byref(input_blob), None, None, None, None, 0, byref(output_blob))
    else:
        success = function(byref(input_blob), None, None, None, None, 0, byref(output_blob))
    if not success:
        raise SIWCError("Windows DPAPI credential operation failed")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        windll.kernel32.LocalFree(output_blob.pbData)


class CredentialStore:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(_protect(self.path.read_bytes(), decrypt=True))

    def save(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            temporary.write_bytes(_protect(json.dumps(record).encode("utf-8")))
            os.replace(temporary, self.path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def _credential_refresh_lock(path: Path, timeout_seconds: float = 60):
    """Serialize refresh-token rotation across Windows worker processes."""
    import msvcrt
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "r+b", buffering=0) as lock_file:
        if os.fstat(fd).st_size == 0:
            lock_file.write(b"\0")
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise SIWCError("Timed out waiting for SIWC credential refresh lock") from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _validate_id_token(token: str, client_id: str, nonce: str | None = None) -> dict:
    jwks = PyJWKClient(AUTH + "/.well-known/jwks.json")
    key = jwks.get_signing_key_from_jwt(token)
    claims = jwt.decode(token, key.key, algorithms=["RS256"], audience=client_id, issuer=AUTH)
    if nonce is not None and not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        raise SIWCError("ID Token nonce mismatch")
    if not claims.get("sub"):
        raise SIWCError("ID Token subject missing")
    return claims


def _token_request(data: dict) -> dict:
    try:
        response = requests.post(AUTH + "/api/accounts/oauth/token", data=data, timeout=(5, 20))
    except requests.RequestException as exc:
        raise SIWCError("OAuth token endpoint connection failed") from exc
    if response.status_code != 200:
        raise SIWCError(f"OAuth token exchange failed (HTTP {response.status_code}); sign in again")
    return response.json()


def login(
    store: CredentialStore | None = None,
    *,
    timeout_seconds: int = 240,
    open_browser: bool = True,
) -> dict:
    """Run SIWC sign-in, optionally printing a URL for manual browser navigation.

    ``open_browser=False`` is intended for Windows host-shell use when the
    caller must paste the authorization URL into an already-open browser. In
    that mode, the optional ``id_token_hint`` is omitted so it is never printed
    as part of the URL.
    """
    store = store or CredentialStore()
    saved = store.load()
    host_id = saved.get("ext_agent_host_id") or "urn:uuid:" + str(uuid.uuid4())
    client_id = saved.get("client_id") or "dynamic_agent_client"
    first = client_id == "dynamic_agent_client"
    state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    received: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path != "/auth/callback":
                self.send_error(404)
                return
            query = parse_qs(parsed.query)
            received.update({key: values[0] for key, values in query.items() if values})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Authorization received. You may close this tab.")

        def log_message(self, *_args):
            pass  # Never log the authorization code or callback URL.

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = timeout_seconds
    redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
    params = dict(client_id=client_id, ext_agent_host_id=host_id, response_type="code",
                  redirect_uri=redirect, scope=SCOPES, resource=RESOURCE, state=state,
                  nonce=nonce, code_challenge_method="S256", code_challenge=challenge)
    if first:
        params["agent_name_hint"] = "AppWorld AgentLab"
    elif open_browser and saved.get("id_token"):
        params["id_token_hint"] = saved["id_token"]
    try:
        authorization_url = AUTH + "/api/accounts/authorize?" + urlencode(params)
        if open_browser:
            webbrowser.open(authorization_url)
            print("Browser opened for ChatGPT authorization; waiting for 127.0.0.1 callback.", flush=True)
        else:
            print("Paste this authorization URL into your existing browser tab:", flush=True)
            print(authorization_url, flush=True)
            print("Waiting for the 127.0.0.1 callback; do not share the URL.", flush=True)
        server.handle_request()
    finally:
        server.server_close()
    if not secrets.compare_digest(received.get("state", ""), state):
        raise SIWCError("OAuth callback missing or state mismatch")
    if received.get("error"):
        raise SIWCError("ChatGPT authorization denied or unavailable")
    issued = received.get("client_id")
    if first and (not issued or issued == "dynamic_agent_client"):
        raise SIWCError("Dynamic client registration did not return an issued client ID")
    if not first and issued and issued != client_id:
        raise SIWCError("Callback client ID differs from saved registration")
    client_id = issued or client_id
    if not received.get("code"):
        raise SIWCError("OAuth callback did not contain an authorization code")
    token = _token_request(dict(grant_type="authorization_code", client_id=client_id,
                                code=received["code"], code_verifier=verifier,
                                redirect_uri=redirect, resource=RESOURCE))
    claims = _validate_id_token(token["id_token"], client_id, nonce)
    if saved.get("subject") and saved["subject"] != claims["sub"]:
        raise SIWCError("Signed-in account does not match stored account")
    scopes = set(token.get("scope", "").split())
    if "chatgpt.tokens.use.direct" not in scopes:
        raise SIWCError("ChatGPT plan inference permission was not granted")
    record = dict(client_id=client_id, ext_agent_host_id=host_id, subject=claims["sub"],
                  email=claims.get("email"), id_token=token["id_token"],
                  access_token=token["access_token"], refresh_token=token.get("refresh_token"),
                  scopes=sorted(scopes), expires_at=(datetime.now(timezone.utc) +
                  timedelta(seconds=int(token.get("expires_in", 3600)))).isoformat())
    store.save(record)
    return {"authenticated": True, "granted_scopes": sorted(scopes), "client_id_issued": True}


def access_token(store: CredentialStore | None = None) -> str:
    store = store or CredentialStore()
    record = store.load()
    if not record:
        raise SIWCError("No SIWC account: run the browser sign-in first")
    if "chatgpt.tokens.use.direct" not in record.get("scopes", []):
        raise SIWCError("ChatGPT plan inference permission unavailable")
    expiry = datetime.fromisoformat(record["expires_at"])
    if expiry > datetime.now(timezone.utc) + timedelta(minutes=2):
        return record["access_token"]
    if not record.get("refresh_token"):
        raise SIWCError("Refresh token unavailable: sign in again")
    with _credential_refresh_lock(store.path):
        record = store.load()
        if "chatgpt.tokens.use.direct" not in record.get("scopes", []):
            raise SIWCError("ChatGPT plan inference permission unavailable")
        expiry = datetime.fromisoformat(record["expires_at"])
        if expiry > datetime.now(timezone.utc) + timedelta(minutes=2):
            return record["access_token"]
        if not record.get("refresh_token"):
            raise SIWCError("Refresh token unavailable: sign in again")
        token = _token_request(dict(grant_type="refresh_token", client_id=record["client_id"],
                                    refresh_token=record["refresh_token"], resource=RESOURCE))
        claims = _validate_id_token(token["id_token"], record["client_id"]) if token.get("id_token") else None
        if claims and claims["sub"] != record["subject"]:
            raise SIWCError("Refreshed account identity changed")
        scopes = set(token.get("scope", " ".join(record["scopes"])).split())
        if "chatgpt.tokens.use.direct" not in scopes:
            raise SIWCError("Refresh removed ChatGPT plan inference permission")
        record.update(access_token=token["access_token"], refresh_token=token.get("refresh_token", record["refresh_token"]),
                      scopes=sorted(scopes), expires_at=(datetime.now(timezone.utc) +
                      timedelta(seconds=int(token.get("expires_in", 3600)))).isoformat())
        if token.get("id_token"):
            record["id_token"] = token["id_token"]
        store.save(record)
        return record["access_token"]
