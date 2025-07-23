import os
import replicate
import subprocess
import uuid
import shutil

class Predictor:
    def setup(self):
        os.makedirs("/tmp/colette", exist_ok=True)

    def predict(self, file: replicate.File, question: str) -> str:
        # Save file locally
        filename = f"/tmp/colette/input_{uuid.uuid4()}"
        with open(filename, "wb") as f:
            f.write(file.read())

        # Run colette CLI on the file
        # Set environment vars Colette expects
        os.environ["COLETTE_DEVICE"] = "cpu"
        os.environ["COLETTE_CACHE_DIR"] = "/tmp/colette/cache"

        # Run Colette command
        try:
            result = subprocess.run(
                ["colette", "ask", filename, question],
                capture_output=True,
                check=True,
                text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"
