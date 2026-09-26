import os
import gdown
import tarfile


def download_and_extract(url, filename):
    print(f"Downloading {url}...")
    gdown.download(url, filename, quiet=False, fuzzy=True)

    print(f"Extracting {filename}...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")  # extract into current directory

    print(f"Removing {filename}...")
    os.remove(filename)


# CIFAR-10 dataset URL (new file)
cifar_url = (
    "https://drive.google.com/file/d/1m5bfPQi43rPipVpqmU_ffyVtz4VK3Sc9/view?usp=sharing"
)

# Download and extract
download_and_extract(cifar_url, "cifar-10-python.tar.gz")

print("Done! Files are in the 'data' folder.")
