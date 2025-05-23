# streamlit_app.py

import os
import sys

os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"


# Add src/ to sys.path so Streamlit can find your modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(PROJECT_ROOT)

# Import main function from the correct folder name 'web_dashboard'
from web_dashboard.dashboard import main

if __name__ == "__main__":
    main()

