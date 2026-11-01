import requests
import tarfile
import os
from tqdm.auto import tqdm

# Define the file URL and the local path where it will be saved
file_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
download_dir = "computer_vision"
file_path = os.path.join(download_dir, "cifar-10-python.tar.gz")

# --- 1. Create the target directory if it doesn't exist ---
os.makedirs(download_dir, exist_ok=True)

# --- 2. Download the file with a progress bar ---
print(f"Downloading dataset from {file_url}...")
try:
    response = requests.get(file_url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with open(file_path, "wb") as file, tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, desc="CIFAR-10"
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong during download.")
    else:
        print("Download complete.")

    # --- 3. Extract the .tar.gz file ---
    print(f"Extracting {file_path}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=download_dir)
    print("Extraction complete.")

finally:
    # --- 4. Clean up by removing the downloaded .tar.gz file ---
    if os.path.exists(file_path):
        print(f"Removing {file_path}.")
        os.remove(file_path)
        print("Cleanup complete.")
