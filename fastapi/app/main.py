from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.connection import close_mongo_connection, connect_to_mongo, get_database
from app.routers.admin import router as admin_router
from app.routers.auth import router as auth_router
from app.routers.summarization import router as summarization_router
from app.routers.evaluation import router as evaluation_router


@asynccontextmanager
async def lifespan(app: FastAPI):

    await connect_to_mongo()
    try:
        yield
    finally:
        await close_mongo_connection()


app = FastAPI(title="FastAPI Auth with MongoDB", lifespan=lifespan)

# CORS middleware - cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # Vite dev server
        "http://localhost:3000",    # Alternative port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],            # GET, POST, PUT, DELETE, OPTIONS...
    allow_headers=["*"],            # Authorization, Content-Type...
)


app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(summarization_router)
app.include_router(evaluation_router)


@app.get("/")
async def root():

    db = get_database()
    collections = await db.list_collection_names()
    return {"message": "Connected to MongoDB!", "collections": collections}
