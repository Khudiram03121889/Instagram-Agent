from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import os

class FileArchiverInput(BaseModel):
    content: str = Field(..., description="The full text content to save (Script, Prompts, Captions combined).")
    topic: str = Field(..., description="The topic name to be used as the filename.")

class FileArchiverTool(BaseTool):
    name: str = "File Archiver Tool"
    description: str = "Saves the generated content to a text file in the specific D: drive folder."
    args_schema: Type[BaseModel] = FileArchiverInput

    def _run(self, content: str, topic: str) -> str:
        # Define the base directory as requested
        base_dir = r"D:\AI  Details\Instagram AI videos\Scripts, Captions and prompts of videos"
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as e:
            return f"Error creating directory '{base_dir}': {str(e)}"

        # Clean filename to be safe
        safe_filename = "".join([c for c in topic if c.isalnum() or c in (' ', '-', '_')]).strip()
        if not safe_filename:
            safe_filename = "untitled_project"
            
        file_path = os.path.join(base_dir, f"{safe_filename}.txt")
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ Successfully saved archive to: {file_path}"
        except Exception as e:
            return f"❌ Error writing file: {str(e)}"
