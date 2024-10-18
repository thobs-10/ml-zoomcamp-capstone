import os
from pathlib import Path
from loguru import logger

project_name = "src"

list_of_files =[
    ".github/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    "data/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/predictions/.gitkeep",
    f"{project_name}/models/.gitkeep",
    "setup.py",
    "notebooks/.gitkeep",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/structure.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/feature_engineering.py",
    f"{project_name}/components/model_training.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/entities/__init__.py",
    f"{project_name}/entities/config.py",
    f"{project_name}/entities/artifact.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    f"{project_name}/pipelines/__init__.py",
    f"{project_name}/pipelines/data_ingestion_pipeline.py",
    f"{project_name}/pipelines/feature_engineering_pipeline.py",
    f"{project_name}/pipelines/model_training_pipeline.py",
    f"{project_name}/app.py",
    f"{project_name}/Dockerfile",
    f"{project_name}/Makefile",
    f"{project_name}/tests/__init__.py",
    f"{project_name}/exceptions/__init__.py",
    f"{project_name}/streamlit_app.py",
    "README.md",
    "requirements.txt",
    "LICENSE",
    ".gitignore",
    ".env",
    
]

for filepath in list_of_files:
    file_path = Path(filepath)
    filedir, filename = os.path.split(file_path)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logger.debug(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filename)) or (os.path.getsize(filename) == 0):
        with open(file_path, "w") as f:
            pass
        logger.debug(f"Creating empty file: {filename}")
    else:
        logger.debug(f"{filename} already exists")
