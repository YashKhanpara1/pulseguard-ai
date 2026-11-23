from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, Integer, String, desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from app.config import settings


engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class IncidentLog(Base):
    __tablename__ = "incident_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    service_name: Mapped[str] = mapped_column(String(80), index=True)
    endpoint: Mapped[str] = mapped_column(String(120))
    incident_type: Mapped[str] = mapped_column(String(40))  # "ANOMALY" or "DRIFT"
    latency_ms: Mapped[float] = mapped_column(Float)
    status_code: Mapped[int] = mapped_column(Integer)
    risk_score: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_recent_incidents(limit: int = 15) -> list[dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(IncidentLog).order_by(desc(IncidentLog.id)).limit(limit)
        )
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "service_name": r.service_name,
                "endpoint": r.endpoint,
                "incident_type": r.incident_type,
                "latency_ms": r.latency_ms,
                "status_code": r.status_code,
                "risk_score": r.risk_score,
                "reason": r.reason,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]