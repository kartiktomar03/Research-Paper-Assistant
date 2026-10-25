"""Convenience entry point: python app.py"""
from backend.app import app

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
