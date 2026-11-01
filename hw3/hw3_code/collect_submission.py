import os
import subprocess
import zipfile
import fnmatch
import sys

# --- Configuration ---
NOTEBOOK_FILE = "main.ipynb"
PDF_OUTPUT_NAME = "hw3_notebook_submission"
ZIP_OUTPUT_NAME = "hw3_code_submission"

# Files and directories to be included in the zip file
# Add any other files or directories you need to include here
INCLUDED_PATHS = ["src/transformer.py"]

# --- Script Start ---
print("=== HW3 Submission Collection Script ===")

# 1. Convert notebook to PDF
# ===========================
print("\nConverting notebook to PDF...")

# Check if notebook file exists first
if not os.path.exists(NOTEBOOK_FILE):
    print(f"FATAL ERROR: Notebook file '{NOTEBOOK_FILE}' not found!")
    sys.exit(1)

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
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(f"Successfully converted {NOTEBOOK_FILE} to {PDF_OUTPUT_NAME}.pdf")
except FileNotFoundError:
    print("FATAL ERROR: 'jupyter' command not found.")
    print(
        "Please ensure you are running this script in an environment where Jupyter is installed."
    )
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"FATAL ERROR: Failed to convert notebook to PDF.")
    print(
        "Please ensure 'nbconvert' and its dependencies are installed (`pip install \"nbconvert[webpdf]\"`)."
    )
    print("\n--- nbconvert Error Output ---")
    print(e.stderr)
    print("------------------------------")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during PDF conversion: {e}")
    sys.exit(1)


# 2. Create zip archive
# ======================
print("\nCreating zip archive...")

# Check that all required paths exist before proceeding
print("Checking required paths...")
missing_paths = []
for path in INCLUDED_PATHS:
    if not os.path.exists(path):
        missing_paths.append(path)

if missing_paths:
    print("FATAL ERROR: Required paths not found:")
    for path in missing_paths:
        print(f"  - {path}")
    print("All required files must exist before creating submission.")
    sys.exit(1)

print("All required paths found")

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
            # If it's a file, write it directly
            if os.path.isfile(path):
                zipf.write(path)
                print(f"  Added file: {path}")
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
                        print(f"  Added file: {file_path}")

    print(f"Created {ZIP_OUTPUT_NAME}.zip")

    # Verify the zip file was created and is not empty
    if not os.path.exists(f"{ZIP_OUTPUT_NAME}.zip"):
        print("FATAL ERROR: Zip file was not created successfully!")
        sys.exit(1)

    zip_size = os.path.getsize(f"{ZIP_OUTPUT_NAME}.zip")
    if zip_size == 0:
        print("FATAL ERROR: Created zip file is empty!")
        sys.exit(1)

    print(f"Zip file verified (size: {zip_size} bytes)")

except Exception as e:
    print(f"FATAL ERROR: Failed to create zip file: {e}")
    sys.exit(1)


# 3. Final verification and summary
# ==================================
print("\nPerforming final verification...")

# Verify PDF was created
pdf_path = f"{PDF_OUTPUT_NAME}.pdf"
if not os.path.exists(pdf_path):
    print(f"FATAL ERROR: PDF file '{pdf_path}' was not created!")
    sys.exit(1)

pdf_size = os.path.getsize(pdf_path)
if pdf_size == 0:
    print(f"FATAL ERROR: PDF file '{pdf_path}' is empty!")
    sys.exit(1)

print(f"PDF verified: {pdf_path} ({pdf_size} bytes)")

# Verify ZIP was created (already checked above, but double-check)
zip_path = f"{ZIP_OUTPUT_NAME}.zip"
if not os.path.exists(zip_path):
    print(f"FATAL ERROR: ZIP file '{zip_path}' was not created!")
    sys.exit(1)

zip_size = os.path.getsize(zip_path)
if zip_size == 0:
    print(f"FATAL ERROR: ZIP file '{zip_path}' is empty!")
    sys.exit(1)

print(f"ZIP verified: {zip_path} ({zip_size} bytes)")

print("\n" + "=" * 36)
print("=== Submission Collection Complete ===")
print(f"Generated: {PDF_OUTPUT_NAME}.pdf")
print(f"Generated: {ZIP_OUTPUT_NAME}.zip")
print("=" * 36)
print("\nPlease submit these files to Gradescope:")
print(f"  - {PDF_OUTPUT_NAME}.pdf")
print(f"  - {ZIP_OUTPUT_NAME}.zip")
print("\nCongratulations on completing the assignment!")
