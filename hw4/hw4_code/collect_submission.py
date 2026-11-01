import os
import zipfile
import sys

import os
import subprocess
import zipfile
import fnmatch
import sys

# --- Configuration ---
NOTEBOOK_FILE = "main.ipynb"
PDF_OUTPUT_NAME = "hw4_notebook_submission"
ZIP_OUTPUT_NAME = "hw4_code_submission"

# Files and directories to be included in the zip file
# Add any other files or directories you need to include here
INCLUDED_PATHS = [
    "diffusion_model.py",
    "noise_prediction_net.py",
    "noise_scheduler.py",
    "randomizer.py",
    "robotics/dataset.py",
    "utils.py",
]

# --- Script Start ---
print("=== HW4 Submission Collection Script ===")

# 1. Convert notebook to PDF
# ===========================
print("\nConverting notebook to PDF...")
# This command calls jupyter's nbconvert tool to create a PDF.
# It requires nbconvert and its webpdf dependencies (like chromium) to be installed.
# To install: pip install "nbconvert[webpdf]"
try:
    # Use sys.executable to ensure we use the jupyter from the current python env
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "webpdf",
        NOTEBOOK_FILE,
        "--output",
        PDF_OUTPUT_NAME,
        "--allow-chromium-download",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    print(f"Successfully converted {NOTEBOOK_FILE} to {PDF_OUTPUT_NAME}.pdf")
except FileNotFoundError:
    print("Error: 'jupyter' command not found.")
    print(
        "Please ensure you are running this script in an environment where Jupyter is installed."
    )
except subprocess.CalledProcessError as e:
    print(f"Error converting notebook to PDF.")
    print(
        "Please ensure 'nbconvert' and its dependencies are installed (`pip install \"nbconvert[webpdf]\"`)."
    )
    print("\n--- nbconvert Error Output ---")
    print(e.stderr)
    print("------------------------------")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# 2. Create zip archive
# ======================
print("\nCreating zip archive...")

# Patterns to exclude from the zip file (directories and file types)
# Note: These are checked against parts of the path
EXCLUDE_DIRS = {"__pycache__", ".venv", ".ipynb_checkpoints"}
EXCLUDE_PATTERNS = ["*.csv", "*.bin", "*.pt", "*.pth"]
# Exact filenames to exclude
EXCLUDE_FILES = {f"{PDF_OUTPUT_NAME}.pdf", f"{ZIP_OUTPUT_NAME}.zip"}

# Remove old zip file if it exists
if os.path.exists(f"{ZIP_OUTPUT_NAME}.zip"):
    os.remove(f"{ZIP_OUTPUT_NAME}.zip")

try:
    with zipfile.ZipFile(f"{ZIP_OUTPUT_NAME}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        # Iterate over each path specified for inclusion
        for path in INCLUDED_PATHS:
            if not os.path.exists(path):
                print(f"Warning: Path '{path}' not found, skipping.")
                continue

            # If it's a file, write it directly
            if os.path.isfile(path):
                zipf.write(path)
                continue

            # If it's a directory, walk through its contents
            for root, dirs, files in os.walk(path):
                # Exclude specified directories from being traversed further
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

                for file in files:
                    file_path = os.path.join(root, file)

                    # Check if the file should be excluded
                    is_excluded_file = file in EXCLUDE_FILES
                    is_excluded_pattern = any(
                        fnmatch.fnmatch(file, pat) for pat in EXCLUDE_PATTERNS
                    )

                    if not is_excluded_file and not is_excluded_pattern:
                        zipf.write(file_path)

    print(f"Created {ZIP_OUTPUT_NAME}.zip")
except Exception as e:
    print(f"Error creating zip file: {e}")


# 3. Final summary
# =================
print("\n" + "=" * 36)
print("=== Submission Collection Complete ===")
print(f"Generated: {PDF_OUTPUT_NAME}.pdf")
print(f"Generated: {ZIP_OUTPUT_NAME}.zip")
print("=" * 36)
print("\nPlease submit these files to Gradescope:")
print(f"  - {PDF_OUTPUT_NAME}.pdf")
print(f"  - {ZIP_OUTPUT_NAME}.zip")
print("\nCongratulations on completing the assignment!")
