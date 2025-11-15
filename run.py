# run.py
import os
from app import create_app

# We create the app instance by calling the factory
app = create_app()

if __name__ == "__main__":
    # Get config from the app's config object
    debug_mode = app.config.get('FLASK_DEBUG', False)
    port = int(app.config.get('FLASK_PORT', 5000))
    
    app.logger.info(f"--- Starting Warmth Server at http://127.0.0.1:{port} ---")
    app.logger.info(f"Debug mode: {debug_mode}")
    
    app.run(debug=debug_mode, port=port, host='127.0.0.1')