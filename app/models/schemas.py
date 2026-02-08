from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int

class AskRequest(BaseModel):
    question: str
    document_id: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    confidence_reason: str
    is_reliable: bool

class ExtractRequest(BaseModel):
    document_id: str

class ShipmentData(BaseModel):
    shipment_id: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    pickup_datetime: Optional[datetime] = None
    delivery_datetime: Optional[datetime] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[float] = None
    currency: Optional[str] = None
    weight: Optional[float] = None
    carrier_name: Optional[str] = None
