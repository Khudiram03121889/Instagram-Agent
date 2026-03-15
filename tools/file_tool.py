import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FileArchiverInput(BaseModel):
    content: str = Field(
        ...,
        description="The full text content to save (Script, Prompts, Captions combined).",
    )
    topic: str = Field(..., description="The topic name to be used as the filename.")


class FileArchiverTool(BaseTool):
    name: str = "File Archiver Tool"
    description: str = (
        "Saves generated content to a text file in the archive directory. "
        "Uses ARCHIVE_DIR env var if set, otherwise uses ./archive."
    )
    args_schema: Type[BaseModel] = FileArchiverInput

    def _run(self, content: str, topic: str) -> str:
        # Use configurable archive directory; default to project-local path.
        base_dir = os.getenv("ARCHIVE_DIR", "archive")
        if not os.path.isabs(base_dir):
            base_dir = os.path.abspath(base_dir)

        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as e:
            return f"Error creating directory '{base_dir}': {e}"

        safe_filename = "".join(c for c in topic if c.isalnum() or c in (" ", "-", "_")).strip()
        if not safe_filename:
            safe_filename = "untitled_project"
        safe_filename = safe_filename[:120]

        file_path = os.path.join(base_dir, f"{safe_filename}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully saved archive to: {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"
