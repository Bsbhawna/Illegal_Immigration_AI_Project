import os
import sys
import traceback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print("✅ [streamlit_app] sys.path updated. Attempting import...")

try:
    from web_dashboard.dashboard import main
    print("✅ Successfully imported `main` from web_dashboard.dashboard")
    main()
except Exception as e:
    print(" Streamlit app crashed during execution:")
    print(f"Exception Type: {type(e).__name__}")
    print(f"Exception Message: {e}")
    traceback.print_exc()
