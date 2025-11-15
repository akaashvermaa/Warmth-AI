import nltk
import sys
import time

print("--- Starting NLTK Model Downloader ---")

# Define all the resources your project needs
resources_to_download = [
    'punkt',    # Required by mood_analyzer.py (word_tokenize)
    'wordnet',  # Required by mood_analyzer.py (WordNetLemmatizer)
]

# Map a resource name to one or more candidate nltk.data paths to check
_RESOURCE_PATH_CANDIDATES = {
    'punkt': [
        'tokenizers/punkt',      # usual
        'tokenizers/punkt_tab',  # sometimes referenced in errors
    ],
    'wordnet': [
        'corpora/wordnet',
    ],
}

def _is_resource_available(resource):
    """Check common candidate paths for the given resource."""
    candidates = _RESOURCE_PATH_CANDIDATES.get(resource, [f"corpora/{resource}", f"tokenizers/{resource}"])
    for path in candidates:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            continue
    return False

def download_resource(resource, retries=1, wait=1.0):
    """
    Downloads a single NLTK resource with a robust existence check.
    Returns True on success.
    """
    print(f"Checking for: {resource}...")

    if _is_resource_available(resource):
        print(f"Resource '{resource}' is already available.")
        return True

    print(f"Resource '{resource}' not found. Attempting download...")
    try:
        nltk.download(resource)
    except Exception as e:
        print(f"Error downloading '{resource}': {e}", file=sys.stderr)
        return False

    # Wait a moment then re-check (sometimes filesystem propagation is slow)
    for _ in range(retries + 1):
        if _is_resource_available(resource):
            print(f"Successfully downloaded and verified '{resource}'.")
            return True
        time.sleep(wait)

    print(f"Failed to verify '{resource}' after download.", file=sys.stderr)
    return False


# --- Main Execution ---
all_successful = True
for res in resources_to_download:
    ok = download_resource(res, retries=2, wait=0.5)
    if not ok:
        all_successful = False

if all_successful:
    print("\n--- All NLTK models are ready. ---")
    sys.exit(0)
else:
    print("\n--- One or more NLTK models failed to download. ---", file=sys.stderr)
    sys.exit(1) # Exit with an error code