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
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    _seed_admin()
    return SessionLocal


def _seed_admin():
    """Create the first admin from env config if it doesn't exist."""
    from passlib.hash import bcrypt
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
                password_hash=bcrypt.hash(s.admin_password),
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
