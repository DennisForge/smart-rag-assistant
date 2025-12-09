from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Osnovne postavke aplikacije
    app_name: str = "Smart RAG Assistant"
    environment: str = "dev"
    debug: bool = True

    # RAG paths
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    chroma_dir: str = "storage/chroma"

    # Model / embedding postavke (za kasnije)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
