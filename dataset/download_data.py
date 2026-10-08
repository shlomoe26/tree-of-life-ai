import os
import urllib.request

def download_tiny_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs("dataset", exist_ok=True)
    filepath = os.path.join("dataset", "input.txt")
    
    if not os.path.exists(filepath):
        print("--- Downloading Tiny Shakespeare ---")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")
    else:
        print("Dataset already present.")

if __name__ == "__main__":
    download_tiny_shakespeare()
