"""Configuration management for AI Research Assistant.

This module handles environment configuration and settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Configuration settings for the research assistant."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # GCP Configuration
        self.project_id = os.getenv("PROJECT_ID", "")
        self.location = os.getenv("LOCATION", "us-central1")
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")

        # Research Settings
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "3"))
        self.quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.8"))

        self._resolve_credentials_path()

        # Validate required settings
        if not self.project_id:
            print("⚠️  Warning: PROJECT_ID not set in environment")

    def _resolve_credentials_path(self):
        """Make a relative service-account path work from any working directory."""
        creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds and not os.path.isabs(creds):
            resolved = (PROJECT_ROOT / creds).resolve()
            if resolved.exists():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved)

    def configure_adk_environment(self) -> str:
        """Export the environment variables the ADK runtime uses to reach Gemini.

        ADK constructs its own model client, so it reads credentials from the
        environment rather than from a caller-supplied genai.Client.
        """
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
            return "api_key"

        if self.project_id:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
            os.environ["GOOGLE_CLOUD_LOCATION"] = self.location
            return "vertex_ai"

        return "none"

    def __repr__(self):
        """String representation of config."""
        return (f"Config(project_id='{self.project_id}', "
                f"location='{self.location}', "
                f"model='{self.model_name}')")


# Global config instance
config = Config()
