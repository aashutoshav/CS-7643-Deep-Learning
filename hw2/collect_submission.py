import os
import subprocess
import zipfile
import fnmatch
import sys

# --- Configuration ---
NOTEBOOK_FILE = "main.ipynb"
PDF_OUTPUT_NAME = "hw2_notebook_submission"
PART_ONE_ZIP_OUTPUT_NAME = "hw2_part1_code_submission"
PART_TWO_ZIP_OUTPUT_NAME = "hw2_part2_code_submission"

# Files and directories to be included in the zip file
# Add any other files or directories you need to include here
PART_ONE_INCLUDED_PATHS = ["part1-convnet/modules/", "part1-convnet/optimizer/"]
PART_TWO_INCLUDED_PATHS = [
    "part2-pytorch/configs/",
    "part2-pytorch/losses/",
    "part2-pytorch/models/",
    "part2-pytorch/checkpoints/",
]

# --- Script Start ---
print("=== HW2 Submission Collection Script ===")

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
EXCLUDE_PATTERNS = ["*.csv", "*.bin", "*.pt"]
# Exact filenames to exclude
EXCLUDE_FILES = {
    f"{PDF_OUTPUT_NAME}.pdf",
    f"{PART_ONE_ZIP_OUTPUT_NAME}.zip",
    f"{PART_TWO_ZIP_OUTPUT_NAME}.zip",
}


def create_zip_archive(zip_output_name, zip_included_paths):
    # Remove old zip file if it exists
    if os.path.exists(f"{zip_output_name}.zip"):
        os.remove(f"{zip_output_name}.zip")
    try:
        with zipfile.ZipFile(
            f"{zip_output_name}.zip", "w", zipfile.ZIP_DEFLATED
        ) as zipf:
            # Iterate over each path specified for inclusion
            for path in zip_included_paths:
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

        print(f"Created {zip_output_name}.zip")
    except Exception as e:
        print(f"Error creating zip file: {e}")


create_zip_archive(PART_ONE_ZIP_OUTPUT_NAME, PART_ONE_INCLUDED_PATHS)
create_zip_archive(PART_TWO_ZIP_OUTPUT_NAME, PART_TWO_INCLUDED_PATHS)

# 3. Final summary
# =================
print("\n" + "=" * 36)
print("=== Submission Collection Complete ===")
print(f"Generated: {PDF_OUTPUT_NAME}.pdf")
print(f"Generated: {PART_ONE_ZIP_OUTPUT_NAME}.zip")
print(f"Generated: {PART_TWO_ZIP_OUTPUT_NAME}.zip")
print("=" * 36)
print("\nPlease submit these files to Gradescope:")
print(f"  - {PDF_OUTPUT_NAME}.pdf")
print(f"  - {PART_ONE_ZIP_OUTPUT_NAME}.zip")
print(f"  - {PART_TWO_ZIP_OUTPUT_NAME}.zip")
print("\nCongratulations on completing the assignment!")
