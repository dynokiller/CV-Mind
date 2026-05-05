import os
import sys

# Add the app directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'app'))

try:
    from app.app import app
    print("Import successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
