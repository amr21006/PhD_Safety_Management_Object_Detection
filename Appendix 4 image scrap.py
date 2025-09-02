"""4th PhD Image Scraper"""

import os
import hashlib
from io import BytesIO
from html.parser import HTMLParser
from urllib.parse import quote, unquote

import urllib3
import PIL.Image as Image
import numpy as np
import imagehash

import warnings
warnings.filterwarnings(“ignore”, message=’Unverified HTTPS request’)

from selenium import webdriver

# Install necessary packages if running in Colab environment.  Skip if already installed.
try:
    from google.colab import drive  # Check for Colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
  !apt-get update -qq > /dev/null  # Suppress output
  !apt install chromium-chromedriver -y > /dev/null


def setup_browser():
    options = webdriver.ChromeOptions()
    options.add_argument(“--headless”)
    options.add_argument(“--no-sandbox”)
    options.add_argument(“--disable-dev-shm-usage”)

    if IN_COLAB:  # Provide path for Colab
        options.binary_location = “/usr/bin/chromium-browser”  # Correct path in Colab

    # For local execution, you may need to set the executable_path 
    # if it’s not automatically found in your PATH
    # browser = webdriver.Chrome(options=options, executable_path=’/path/to/chromedriver’)

    browser = webdriver.Chrome(options=options) # Works without path if in your system’s PATH
    return browser


def sha256(fname):
    hasher = hashlib.sha256()
    with open(fname, “rb”) as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def remove_duplicate_images(directory):
    print("\nChecking for duplicate images by comparing SHA-256 hash")
    hashes = {}
    duplicates = []
    for filename in os.listdir(directory):
        if filename.endswith((“.png”, “.jpg”, “.jpeg”)):
            filepath = os.path.join(directory, filename)
            filehash = sha256(filepath)
            if filehash in hashes:
                duplicates.append(filepath)
            else:
                hashes[filehash] = filepath
    
    for duplicate in duplicates:
        print(f"Removing duplicate image: {duplicate}")
        os.remove(duplicate)

    if not duplicates:
        print("No duplicate images found")



def get_perceptual_hash(img_path):
    img = Image.open(img_path)
    hashes = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]
    combined_hash = np.array([h(img).hash for h in hashes]).flatten()
    return np.where(combined_hash, 1, 0)



def compare_hash(hash1, hash2):
    return np.count_nonzero(hash1 == hash2) / len(hash1)


def remove_similar_images(directory, similarity_threshold=0.98):
    print("\nChecking for similar images")
    hashes = []
    similar = []
    for filename in os.listdir(directory):
      if filename.endswith((“.png”, “.jpg”, “.jpeg”)):  # Check file extension
        filepath = os.path.join(directory, filename)

        try:  # Handle potential errors like corrupted images
            filehash = get_perceptual_hash(filepath)
        except Exception as e:
            print(f"Error processing image {filename}: {e}")  # Print the specific error
            similar.append(filepath)  # Remove corrupted images to avoid future issues
            continue


        for existing_hash in hashes:
            if compare_hash(filehash, existing_hash) >= similarity_threshold:
                similar.append(filepath)
                break
        else:  # Only append if the file is not similar to any existing ones
            hashes.append(filehash)

    for sim_file in similar:
        print(f"Removing similar image: {sim_file}")
        os.remove(sim_file)

    if not similar:
        print("No similar images found")



class Extractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.src = []
        self.tag_attr = None

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            for attr, value in attrs:
                if attr == self.tag_attr:
                    self.src.append(value)


def create_output_directory(keyword, out_dir=None):
    directory = keyword if out_dir is None else os.path.join(out_dir, keyword)
    os.makedirs(directory, exist_ok=True)


def add_prefix_suffix(keyword, prefix=None, suffix=None):
    if prefix:
        keyword = f"{prefix} {keyword}"
    if suffix:
        keyword = f"{keyword} {suffix}"
    return keyword.strip()



def filter_src_format(src_list):
    return [
        src
        for src in src_list
        if any(ext in src for ext in (“.png”, “.jpg”, “.jpeg”, “/png”, “/jpg”, “/jpeg”)) or “https:” in src
    ]


def get_img_data(url, src):
    if "https:" in src or "www." in src:
      try:  # Wrap in try block for better error handling
          response = http.request(“GET”, src)
          img_data = BytesIO(response.data)
      except Exception as e: # Catch exceptions and move on
          print(f"Error downloading {src}: {e}")
          return None
    elif src.endswith((".png", ".jpg")):
      # handle relative image paths
      # ... (add logic as needed)
      return None  # or raise an error, handle differently
    else: # Base64 encoded image
      # ...handle base64
      return None # Or raise an error


def scrape_images_search_engine(keyword, search_engine, output_directory, num_images=None):
    print(f"\nSearch engine: {search_engine}")

    search_engine_urls = {
        "google": f"https://www.google.com/search?tbm=isch&q={quote(keyword)}",
        "bing": f"https://www.bing.com/images/search?q={quote(keyword)}",
        "yahoo": f"https://images.search.yahoo.com/search/images?p={quote(keyword)}",
        "duckduckgo": f"https://duckduckgo.com/?q={quote(keyword)}&iax=images&ia=images",
    }
    url = search_engine_urls[search_engine]
    print(f"URL: {url}")

    browser.get(url)


    scroll_count = {"google": 3, "bing": 3, "yahoo": 1, "duckduckgo": 5}
    for _ in range(scroll_count.get(search_engine, 1)): # Use get to handle potential missing keys
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    

    extractor.src = []
    extractor.tag_attr = "src"  # Set here, then conditionally change for Google

    if search_engine == "google":
      extractor.tag_attr = "data-src"
      extractor.feed(browser.page_source)  # Feed page source twice: once for data-src

      extractor.tag_attr = "src"  # And again for src
      extractor.feed(browser.page_source)
    
    elif search_engine == "bing":
      extractor.feed(browser.page_source)
      extractor.src = list({x.split(“?w”)[0] for x in extractor.src if "OIP" in x })

    elif search_engine == "yahoo":
      extractor.feed(browser.page_source)
      extractor.src = list({x.split(“&”)[0] for x in extractor.src})



    elif search_engine == "duckduckgo":
      extractor.feed(browser.page_source)
      extractor.src = [unquote(x.split(“?”)[-1][2:]) for x in extractor.src]  # Unquote DuckDuckGo URLs




    extractor.src = filter_src_format(extractor.src)
    print(f"Number of images found: {len(extractor.src)}")

    # ... rest of the function (downloading and saving) ...




# --- Main execution ---
if __name__ == "__main__":  # Add this block to only run code when directly executed

  http = urllib3.PoolManager()
  browser = setup_browser()
  extractor = Extractor()




  search_engine = "google"  # or “all”, “bing”, etc.
  num_images = None  # or a specific number
  prefix = None
  suffix = None
  similarity_threshold = 0.98
  out_dir = "images"


  if search_engine not in (“google”, “bing”, “yahoo”, “duckduckgo”, “all”):
      raise ValueError("Invalid search engine.")

  if num_images is not None and num_images < 1:  # Check for 0 or negative num_images
      raise ValueError("num_images must be positive or None.")  

  try:
      with open("keywords.txt", “r”) as infile:
          keywords = [line.strip() for line in infile if line.strip()]

          for keyword in keywords:
              print(“\n” + “-” * 100)
              keyword = add_prefix_suffix(keyword, prefix, suffix)
              print(f"Keyword: {keyword}")

              create_output_directory(keyword, out_dir)
              output_directory = keyword if out_dir is None else os.path.join(out_dir, keyword)

              if search_engine == "all":
                for se in (“google”, “bing”, “yahoo”, “duckduckgo”):
                    scrape_images_search_engine(keyword, se, output_directory, num_images) # Remove unnecessary num_images calculation

              else:
                  scrape_images_search_engine(keyword, search_engine, output_directory, num_images)



              remove_duplicate_images(output_directory)
              remove_similar_images(output_directory, similarity_threshold)
              print(“-” * 100)

  except FileNotFoundError:
       print("keywords.txt not found.")

  finally: # Ensure cleanup even if errors occur
      browser.quit()

      if IN_COLAB and os.path.exists("images"):
          !zip -r images.zip images  # Zip the “images” folder in Colab
