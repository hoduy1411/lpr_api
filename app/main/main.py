from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..core.constants import constants
from ..api import route


app = FastAPI(
    title = constants.PROJECT_TITLE,
    version = constants.PROJECT_VERSION,
    docs_url='/api/docs',
    redoc_url='/api/redoc',
    openapi_url='/api/openapi.json',
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(route.router, prefix=constants.API_V1_STR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)