import os
from flask import Flask
from endpoints import endpoints
from recommendations import recommendations_bp
from database import load_database
from flask_cors import CORS


app = Flask(__name__)

CORS(app)

# Register the Blueprint with the app
app.register_blueprint(endpoints)
app.register_blueprint(recommendations_bp)

def str_to_bool(value: str) -> bool:
    """Converts a string to a boolean if it's a valid representation, otherwise returns None."""
    true_values = {"t", "true"}
    false_values = {"f", "false"}

    if isinstance(value, str):
        value = value.strip().lower()  # Normalize case and remove spaces
        if value in true_values:
            return True
        elif value in false_values:
            return False

    return None  # Return None if not a valid boolean string
    
def main():
    force_load = os.getenv("FORCE_LOAD", "False")
    force_load = str_to_bool(force_load)
    load_database(force_load)

if __name__ == "__main__":
    main()
    app.run(debug=True)
