from fastapi import APIRouter
from .v1 import lp_ocr

router = APIRouter()

@router.get("/health")
def healthCheck():
    return {"Hello": "World"}


router.include_router(
    lp_ocr.router,
    prefix='/lp-ocr',
    tags=["OCR"],
)