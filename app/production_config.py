from pydantic_settings import BaseSettings
from typing import List, ClassVar

class ProductionSettings(BaseSettings):
    """Production configuration settings"""
    
    # Domain configuration - using ClassVar since this isn't a configurable field
    DOMAIN: ClassVar[str] = "legal.adityaberry.me"
    
    # CORS settings - allow the domain and all origins for tunnel usage
    CORS_ORIGINS: List[str] = [
        f"https://{DOMAIN}",
        f"http://{DOMAIN}",
        "http://localhost:5173",  # Frontend dev server
        "http://localhost:3000",  # Alternative frontend dev server
        "*"  # Allow all origins for tunnel usage
    ]
    
    # Environment
    ENVIRONMENT: str = "production"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment"""
        return self.CORS_ORIGINS
    
    class Config:
        env_file = ".env"

# Production settings instance
prod_settings = ProductionSettings()
