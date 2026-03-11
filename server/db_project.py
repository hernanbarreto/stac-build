"""
STAC Build — Project Database Models
======================================

SQLAlchemy models for the hierarchical project structure:
    Project → IFCModel[]
    Project → ScanDay[] → Source[]

Authors: Hernán Barreto — Ingerop IN3
"""

from datetime import datetime, date

from sqlalchemy import (
    Column, Integer, String, Date, DateTime, Text, Float,
    ForeignKey, Boolean,
)
from sqlalchemy.orm import relationship

from db import Base


# ═══════════════════════════════════════════════════════════════════
#  PROJECT
# ═══════════════════════════════════════════════════════════════════

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    location = Column(String(300), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    ifc_models = relationship("IFCModel", back_populates="project", cascade="all, delete-orphan")
    scan_days = relationship("ScanDay", back_populates="project", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "slug": self.slug,
            "name": self.name,
            "location": self.location,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "ifc_count": len(self.ifc_models) if self.ifc_models else 0,
            "scan_day_count": len(self.scan_days) if self.scan_days else 0,
        }


# ═══════════════════════════════════════════════════════════════════
#  IFC MODEL (discipline)
# ═══════════════════════════════════════════════════════════════════

class IFCModel(Base):
    __tablename__ = "ifc_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    discipline = Column(String(50), nullable=False)  # HVAC, Structure, Architecture, MEP, etc.
    filename = Column(String(200), nullable=False)
    version = Column(String(50), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="ifc_models")

    # Valid disciplines (extensible)
    DISCIPLINES = (
        "Architecture", "Structure", "HVAC", "MEP",
        "Electrical", "Plumbing", "Fire", "Landscape",
        "Foundation", "Interior", "Other",
    )

    def to_dict(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "discipline": self.discipline,
            "filename": self.filename,
            "version": self.version,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
        }


# ═══════════════════════════════════════════════════════════════════
#  SCAN DAY
# ═══════════════════════════════════════════════════════════════════

class ScanDay(Base):
    __tablename__ = "scan_days"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    scan_date = Column(Date, nullable=False)
    operator = Column(String(100), nullable=True)
    zone = Column(String(200), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="scan_days")
    sources = relationship("Source", back_populates="scan_day", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "project_id": self.project_id,
            "scan_date": self.scan_date.isoformat() if self.scan_date else None,
            "operator": self.operator,
            "zone": self.zone,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source_count": len(self.sources) if self.sources else 0,
        }


# ═══════════════════════════════════════════════════════════════════
#  SOURCE (one video = one reconstruction run)
# ═══════════════════════════════════════════════════════════════════

class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_day_id = Column(Integer, ForeignKey("scan_days.id"), nullable=False)
    source_name = Column(String(100), nullable=False)  # slug used in dir name
    video_path = Column(String(500), nullable=True)     # original video file
    operator = Column(String(100), nullable=True)
    duration_s = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    status = Column(String(30), default="pending")      # pending, processing, done, error
    created_at = Column(DateTime, default=datetime.utcnow)

    scan_day = relationship("ScanDay", back_populates="sources")

    # Valid statuses
    STATUSES = ("pending", "extracting", "processing", "merging", "done", "error")

    def to_dict(self):
        return {
            "id": self.id,
            "scan_day_id": self.scan_day_id,
            "source_name": self.source_name,
            "video_path": self.video_path,
            "operator": self.operator,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
