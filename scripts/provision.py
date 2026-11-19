"""One-time provisioning, run explicitly with ADMIN_DATABASE_URL and --apply.

Never run this script from Streamlit. The administrator URL is not used by the app.
"""
from __future__ import annotations
import argparse
import os
import secrets
from urllib.parse import quote, urlsplit, urlunsplit
import psycopg
from psycopg import sql
from dotenv import load_dotenv


def role_url(admin_url: str, role: str, password: str) -> str:
    p = urlsplit(admin_url)
    if not p.hostname or not p.path:
        raise ValueError("Invalid admin database URL")
    host = p.hostname
    if ":" in host:
        host = f"[{host}]"
    if p.port:
        host += f":{p.port}"
    return urlunsplit((p.scheme, f"{quote(role)}:{quote(password, safe='')}@{host}",p.path,p.query,p.fragment))


def provision(admin_url: str):
    reader_pw = secrets.token_urlsafe(36)
    uploader_pw = secrets.token_urlsafe(36)
    with psycopg.connect(admin_url) as conn:
        for role, password in [("optimizer_reader",reader_pw),("optimizer_uploader",uploader_pw)]:
            exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s",(role,)).fetchone()
            if exists:
                # Avoid altering pre-existing roles that may be used by another application.
                raise RuntimeError(f"Role {role} already exists; aborting to protect existing configuration.")
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT").format(sql.Identifier(role), sql.Literal(password)))
        conn.execute("CREATE SCHEMA IF NOT EXISTS optimizer_data")
        conn.execute("REVOKE ALL ON SCHEMA optimizer_data FROM PUBLIC")
        conn.execute("GRANT USAGE ON SCHEMA optimizer_data TO optimizer_reader")
        conn.execute("GRANT USAGE, CREATE ON SCHEMA optimizer_data TO optimizer_uploader")
        conn.execute("GRANT CONNECT ON DATABASE " + sql.Identifier(conn.info.dbname).as_string(conn) + " TO optimizer_reader, optimizer_uploader")
        conn.execute("ALTER ROLE optimizer_reader SET default_transaction_read_only = on")
        conn.execute("ALTER ROLE optimizer_reader SET statement_timeout = '5000ms'")
        conn.execute("ALTER ROLE optimizer_uploader SET statement_timeout = '15000ms'")
        # Do not grant public table permissions to reader; uploaded tables get explicit SELECT grants.
        conn.execute("REVOKE CREATE ON SCHEMA public FROM optimizer_reader, optimizer_uploader")
    return role_url(admin_url,"optimizer_reader",reader_pw),role_url(admin_url,"optimizer_uploader",uploader_pw)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    load_dotenv()
    if not args.apply:
        print("Dry run: would create optimizer_reader, optimizer_uploader, schema optimizer_data and limited grants. Use --apply to continue.")
    else:
        url=os.getenv("ADMIN_DATABASE_URL")
        if not url:
            raise SystemExit("ADMIN_DATABASE_URL is required for one-time provisioning.")
        reader,uploader=provision(url)
        env_path=".env"
        # Replace placeholders, rather than append duplicate keys that dotenv might ignore.
        from pathlib import Path
        path = Path(env_path)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        lines = [line for line in existing.splitlines() if not line.startswith(("READONLY_DATABASE_URL=", "UPLOAD_DATABASE_URL="))]
        lines += ["READONLY_DATABASE_URL=" + reader, "UPLOAD_DATABASE_URL=" + uploader]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try: os.chmod(env_path,0o600)
        except OSError: pass
        print("Provisioned restricted roles. Connection URLs appended to .env (not printed).")
