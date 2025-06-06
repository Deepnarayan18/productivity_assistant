import os 
from pathlib import Path 
import logging 

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s:') 

list_of_files = [
    ".github/workflows/.gitkeep", 
    "backend/ingest_data.py", 
    "backend/model.py", 
    "app.py", 
     
    "templates/index.html", 
    ".env", 
    "Dockerfile",
    "test.py"
] 

for filepath in list_of_files: 
    filepath = Path(filepath)  
    filedir,filename = os.path.split(filepath) 
    
    if filedir !="": 
        os.makedirs(filedir,exist_ok=True) 
        logging.info(f"creating directory;{filedir} for the file: {filename}") 
        
    if(not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): 
        with open(filepath,"w")as f:
             pass 
             logging.info(f"creating file: {filename}") 
    
    else: 
        logging.info(f"file {filename} already exists")