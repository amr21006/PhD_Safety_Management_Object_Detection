Algorithm 5-1
# Setup
Initialize WebDriver, Extractor, PoolManager
If search_engine is not in the list of allowed search engines, throw Exception
If num_images is defined, calculate per search engine images

# Read the keywords
For each keyword in the keywords file:
    If keyword is not empty:
        Add prefix and suffix if they exist
        Create output directory for the keyword
        Determine output_directory
        
        If search_engine is not "all":
            Call scrape_images_search_engine with keyword, search_engine, output_directory, num_images
        Else:
            For each search engine in ['google', 'bing', 'yahoo', 'duckduckgo']:
                Call scrape_images_search_engine with keyword, each search engine, output_directory, num_images
        
        # Post-Processing
        Call remove_duplicate_images with output_directory
        Call remove_similar_images with output_directory, similarity_threshold

        Wait for 2 seconds

Close the WebDriver

# Function scrape_images_search_engine
Function scrape_images_search_engine(keyword, search_engine, output_directory, num_images):
    Construct URL based on search engine and keyword
    Navigate to URL using WebDriver and wait for the page to load
    Scroll down page to load more images
    Parse page source with HTML parser and extract img tag attribute values
    Filter the extracted URLs
    For each URL in filtered URLs:
        Retrieve image data using get_img_data function
        Save image data as image file
    Print number of images downloaded

# Function get_img_data
Function get_img_data(url, src):
    If src is direct image URL:
        Make HTTP request and get image data
    ElseIf src ends with image file extension:
        Construct absolute URL and make HTTP request to get image data
    Else:
        Decode base64-encoded image and get image data
    Return image data


