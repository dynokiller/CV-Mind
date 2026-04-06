from app.main import app
import sys

print("Registered Routes:")
for route in app.routes:
    print(f"{route.methods} {route.path}")

if not any(r.path == "/analyze" and "POST" in r.methods for r in app.routes):
    print("\nERROR: POST /analyze route not found!")
    sys.exit(1)

print("\nSUCCESS: Routes verified.")
