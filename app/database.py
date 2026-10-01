"""Database engine/session bootstrap.

Works with both SQLite (local dev) and PostgreSQL (Neon on Vercel).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import get_settings
from .models import Base, User


def get_engine():
    url = get_settings().database_url
    kwargs = {}
    if url.startswith("postgres"):
        kwargs = {"pool_pre_ping": True, "pool_size": 10, "max_overflow": 20}
    elif url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
    return create_engine(url, **kwargs)


engine = None
SessionLocal = None


def init_db():
    global engine, SessionLocal
    url = get_settings().database_url
    if url.startswith("sqlite"):
        # Vercel serverless filesystems are read-only: SQLite cannot persist.
        # Only use it for LOCAL development.
        pass
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    _auto_add_missing_columns()
    _seed_admin()
    return SessionLocal


def _auto_add_missing_columns():
    """`create_all` only creates whole tables — it never adds new columns to a
    table that already exists. That silently breaks the API with
    "no such column" after any model change. This adds any missing columns
    (ADD COLUMN IF NOT EXISTS) so schema changes deploy safely."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # table was just created with the right shape
        have = {c["name"] for c in inspector.get_columns(table.name)}
        for col in table.columns:
            if col.name in have:
                continue
            ddl = _column_ddl(col)
            if not ddl:
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE %s ADD COLUMN %s" % (table.name, ddl)))
                print("[db] added column %s.%s" % (table.name, col.name))
            except Exception as e:  # never block boot on a migration hiccup
                print("[db] could not add %s.%s: %s" % (table.name, col.name, e))


def _column_ddl(col):
    """Render a portable `ALTER TABLE ... ADD COLUMN` fragment."""
    try:
        coltype = col.type.compile(engine.dialect)
    except Exception:
        return None
    ddl = "%s %s" % (col.name, coltype)
    default = getattr(col, "server_default", None)
    if default is not None and default.arg is not None:
        ddl += " DEFAULT %s" % (default.arg.text if hasattr(default.arg, "text") else default.arg)
    return ddl


def _seed_admin():
    """Create the first admin from env config if it doesn't exist."""
    import bcrypt
    from .config import get_settings

    s = get_settings()
    with SessionLocal() as db:
        existing = db.query(User).filter(User.is_admin == True).first()  # noqa: E712
        if existing:
            return
        admin = db.query(User).filter(User.email == s.admin_email).first()
        if not admin:
            admin = User(
                email=s.admin_email,
                username=s.admin_email.split("@")[0],
                password_hash=bcrypt.hashpw(
                    s.admin_password.encode("utf-8"), bcrypt.gensalt()
                ).decode("utf-8"),
                is_admin=True,
                is_active=True,
            )
            db.add(admin)
            db.commit()


def get_db():
    if SessionLocal is None:
        init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def active_driver() -> str:
    url = get_settings().database_url
    if url.startswith("postgres"):
        return "postgresql (Neon)"
    return "sqlite"


def get_ollama_url(db) -> str:
    """Resolve the Ollama base URL. Prefers the runtime-updatable value stored
    in ApiSetting (set by the Colab/Kaggle GPU notebooks), falls back to the
    OLLAMA_URL env var."""
    from .models import ApiSetting

    row = db.query(ApiSetting).filter(ApiSetting.key == "ollama_url").first()
    if row and row.value:
        return row.value
    return get_settings().ollama_url
