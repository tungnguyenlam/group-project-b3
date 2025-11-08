from dotenv import load_dotenv
import os 

# Simple validation
def get_hf_token_base_dir():
    BASE_DIR = os.getcwd()
    if not BASE_DIR.endswith('/group-project-b3'):
        raise ValueError(f"Expected to be in .../group-project-b3 directory, but got: {BASE_DIR}")
    else:
        load_dotenv(os.path.abspath(os.path.join(BASE_DIR,'..','.env')))
        HF_TOKEN = os.getenv('HF_TOKEN')
    return HF_TOKEN, BASE_DIR

BASE_DIR = get_hf_token_base_dir()[1]

if __name__ == "__main__":
    print(get_hf_token_base_dir()[0])