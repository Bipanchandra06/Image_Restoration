import os
import time
import requests
import yaml
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import hashlib

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageScraper:
    """
    Selenium-based web scraper for collecting high-quality images
    """
    
    def __init__(self, config_path="/users/bipan/onedrive/desktop/clear vision/config/config.yaml"):
        """
        Initialize the image scraper with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self.config = config_data['scraper']
                logger.info(f"Loaded configuration from {config_path}")
                logger.info(f"Search terms: {self.config['search_terms']}")
                logger.info(f"Download path: {self.config['download_path']}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
        
        # Create absolute path for download directory
        if self.config['download_path'].startswith('./'):
            # Convert relative path to absolute
            base_dir = os.path.dirname(os.path.abspath(config_path))
            self.config['download_path'] = os.path.normpath(
                os.path.join(base_dir, self.config['download_path'][2:])
            )
        
        logger.info(f"Using download path: {self.config['download_path']}")
        
        # Create download directory if it doesn't exist
        try:
            os.makedirs(self.config['download_path'], exist_ok=True)
            logger.info(f"Created download directory: {self.config['download_path']}")
        except Exception as e:
            logger.error(f"Failed to create download directory: {e}")
            raise
        
        # Initialize browser options
        self.setup_browser()
        
    def setup_browser(self):
        """Setup Chrome browser with appropriate options"""
        try:
            chrome_options = Options()
            # Comment out headless for debugging if needed
            # chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920x1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Initialize the Chrome driver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Chrome browser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome browser: {e}")
            raise
        
    def scrape_images(self, custom_search_terms=None):
        """
        Scrape images from the web
        
        Args:
            custom_search_terms (list, optional): Custom search terms to override config
        
        Returns:
            list: Paths to downloaded images
        """
        search_terms = custom_search_terms or self.config['search_terms']
        logger.info(f"Starting image scraping with terms: {search_terms}")
        
        image_count = 0
        downloaded_images = []
        
        for term in search_terms:
            logger.info(f"Processing search term: '{term}'")
            if image_count >= self.config['max_total_images']:
                logger.info("Reached maximum total images limit")
                break
                
            term_images = self._download_images_for_term(
                term, 
                self.config['num_images_per_term'],
                image_count
            )
            
            downloaded_images.extend(term_images)
            image_count += len(term_images)
            logger.info(f"Downloaded {len(term_images)} images for term '{term}'")
            
        logger.info(f"Total images downloaded: {len(downloaded_images)}")
        return downloaded_images

    def _download_images_for_term(self, term, num_images, current_count):
        """
        Download images for a specific search term
        
        Args:
            term (str): Search term
            num_images (int): Number of images to download
            current_count (int): Current image count
            
        Returns:
            list: Paths to downloaded images
        """
        downloaded_paths = []
        max_images = min(num_images, self.config['max_total_images'] - current_count)
        
        if max_images <= 0:
            return downloaded_paths
        
        # Navigate to Google Images
        try:
            logger.info(f"Navigating to Google Images for term: '{term}'")
            self.driver.get("https://images.google.com")
            
            # Accept cookies if prompt appears
            try:
                WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept all')]"))
                ).click()
                logger.info("Accepted cookies popup")
            except (TimeoutException, WebDriverException) as e:
                logger.info(f"No cookie prompt found or couldn't interact with it: {e}")
                
            # Find search box and enter term
            try:
                search_box = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "q"))
                )
                search_box.clear()
                search_box.send_keys(term)
                search_box.send_keys(Keys.RETURN)
                logger.info(f"Entered search term: '{term}'")
            except Exception as e:
                logger.error(f"Failed to find or interact with search box: {e}")
                return downloaded_paths
            
            # Wait for images to load
            logger.info("Waiting for search results to load...")
            time.sleep(10)
            logger.info("Page loaded, starting to process images")
            
            # Debug screenshot for troubleshooting
            screenshot_path = os.path.join(self.config['download_path'], f"debug_screenshot_{term}.png")
            self.driver.save_screenshot(screenshot_path)
            logger.info(f"Saved debug screenshot to {screenshot_path}")
            
            # Updated CSS selectors for Google Images as of 2025
            selectors_to_try = [
                ".rg_i.Q4LuWd",                # Traditional selector
                "img.rg_i",                     # Alternative selector
                "img.Q4LuWd",                   # Another possibility
                "div[data-id] img",             # More generic approach
                "div.isv-r img",                # Another common structure
                "img[data-src]",                # Images with data-src attribute
                "img[src*='http']"              # All images with http in src
            ]
            
            # Track scroll attempts
            scroll_attempts = 0
            max_scroll_attempts = 10
            
            # Scroll to load more images
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            logger.info(f"Initial page height: {last_height}")
            
            while len(downloaded_paths) < max_images and scroll_attempts < max_scroll_attempts:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                logger.info(f"Scrolled down, attempt {scroll_attempts+1}/{max_scroll_attempts}")
                time.sleep(5)
                
                # Calculate new scroll height and compare with last scroll height
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                logger.info(f"New page height: {new_height}")
                
                if new_height == last_height:
                    # Try to click on "Show more results" if available
                    try:
                        # Try different "Show more" button selectors
                        show_more_selectors = [
                            ".mye4qd",                               # Traditional selector
                            "input[value='Show more results']",      # Input button
                            "button.r0zKGf",                         # Another possible button
                            "div.YstHxe input"                       # Yet another possibility
                        ]
                        
                        for selector in show_more_selectors:
                            try:
                                more_buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                if more_buttons:
                                    for btn in more_buttons:
                                        if btn.is_displayed():
                                            logger.info(f"Found 'Show more' button with selector: {selector}")
                                            btn.click()
                                            logger.info("Clicked 'Show more' button")
                                            time.sleep(5)
                                            break
                            except Exception as e:
                                logger.debug(f"Failed with selector {selector}: {e}")
                                continue
                    except Exception as e:
                        logger.info(f"Couldn't find or click 'Show more' button: {e}")
                        scroll_attempts += 1
                else:
                    # Reset scroll attempts if page height changed
                    scroll_attempts = 0
                    
                last_height = new_height
                
                # Extract image URLs
                all_images = []
                for selector in selectors_to_try:
                    try:
                        images = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if images:
                            logger.info(f"Found {len(images)} images with selector: {selector}")
                            all_images.extend(images)
                            break
                    except Exception as e:
                        logger.debug(f"Failed with selector {selector}: {e}")
                
                if not all_images:
                    logger.warning("No images found with any selector")
                    scroll_attempts += 1
                    continue
                
                logger.info(f"Processing {len(all_images)} images")
                
                for i, img in enumerate(all_images):
                    if len(downloaded_paths) >= max_images:
                        logger.info(f"Reached target of {max_images} images for term '{term}'")
                        break
                        
                    try:
                        # Try to get image URL directly first
                        src = img.get_attribute('src')
                        data_src = img.get_attribute('data-src')
                        
                        # If we have a direct usable image URL
                        if src and src.startswith('http') and not src.startswith('data:'):
                            logger.info(f"Found direct image URL: {src[:50]}...")
                            img_path = self._download_image(src, term)
                            if img_path:
                                downloaded_paths.append(img_path)
                                continue
                        elif data_src and data_src.startswith('http'):
                            logger.info(f"Found data-src image URL: {data_src[:50]}...")
                            img_path = self._download_image(data_src, term)
                            if img_path:
                                downloaded_paths.append(img_path)
                                continue
                            
                        # If direct URL didn't work, try clicking
                        logger.info(f"Trying to click image {i+1}")
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", img)
                        time.sleep(1)
                        img.click()
                        logger.info("Clicked on thumbnail")
                        time.sleep(5)
                        
                        # Try different selectors for the full-sized image
                        full_img_selectors = [
                            ".n3VNCb",                       # Traditional
                            ".KAlRDb",                       # Another possibility
                            "img.sFlh5c",                    # Another possibility
                            "img.iPVvYb",                    # Another possibility
                            "img[jsname='HiaYvf']",          # By jsname attribute
                            "img[jsaction*='load']",         # Images with load action
                            "div.v4dQwb img"                 # Container with image
                        ]
                        
                        found_image = False
                        for selector in full_img_selectors:
                            try:
                                # Wait for selector with short timeout
                                WebDriverWait(self.driver, 3).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                                )
                                
                                # Get all images matching selector
                                actual_images = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                logger.info(f"Found {len(actual_images)} full images with selector: {selector}")
                                
                                for actual_image in actual_images:
                                    src = actual_image.get_attribute('src')
                                    if src and src.startswith('http') and not src.startswith('data:'):
                                        logger.info(f"Found full image URL: {src[:50]}...")
                                        img_path = self._download_image(src, term)
                                        if img_path:
                                            downloaded_paths.append(img_path)
                                            found_image = True
                                            break
                                
                                if found_image:
                                    break
                            except Exception as e:
                                logger.debug(f"Failed with full image selector {selector}: {e}")
                        
                        # Try to go back to results
                        try:
                            self.driver.execute_script("document.querySelector('div.dFMRD').click()")
                        except:
                            try:
                                # Alternative: use browser back button
                                self.driver.back()
                                time.sleep(3)
                            except Exception as e:
                                logger.error(f"Failed to go back to results: {e}")
                                # Reload the page as last resort
                                self.driver.get(self.driver.current_url)
                                time.sleep(5)
                                
                    except Exception as e:
                        logger.error(f"Error processing image {i+1}: {e}")
                        continue
                
                # Break if we've downloaded enough images
                if len(downloaded_paths) >= max_images:
                    break
                    
                # Increment scroll attempts
                scroll_attempts += 1
                
        except Exception as e:
            logger.error(f"Error during scraping for term '{term}': {e}")
            
        logger.info(f"Completed search for '{term}', downloaded {len(downloaded_paths)} images")
        return downloaded_paths
        
    def _download_image(self, url, term):
        """
        Download an image from URL
        
        Args:
            url (str): Image URL
            term (str): Search term for filename
            
        Returns:
            str: Path to downloaded image or None if failed
        """
        try:
            # Create a unique filename using URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()
            safe_term = term.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = f"{safe_term}_{url_hash}.jpg"
            download_path = os.path.join(self.config['download_path'], filename)
            
            # Check if file already exists
            if os.path.exists(download_path):
                logger.info(f"Image already exists: {download_path}")
                return download_path
                
            # Download the image
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/'
            }
            
            logger.info(f"Downloading image from: {url[:50]}...")
            response = requests.get(url, stream=True, timeout=10, headers=headers)
            response.raise_for_status()
            
            # Save the image
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Verify file was created and has content
            if os.path.exists(download_path) and os.path.getsize(download_path) > 0:
                logger.info(f"Downloaded: {download_path} ({os.path.getsize(download_path)} bytes)")
                return download_path
            else:
                logger.error(f"Download failed or empty file: {download_path}")
                if os.path.exists(download_path):
                    os.remove(download_path)
                return None
            
        except Exception as e:
            logger.error(f"Error downloading image {url[:50]}...: {e}")
            return None
            
    def close(self):
        """Close the browser"""
        if hasattr(self, 'driver'):
            logger.info("Closing browser")
            self.driver.quit()
            
    def __del__(self):
        """Destructor to ensure browser is closed"""
        self.close()

# Script to run the scraper independently
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape images from the web')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--terms', nargs='+', help='Custom search terms')
    
    args = parser.parse_args()
    
    try:
        scraper = ImageScraper(config_path=args.config)
        downloaded_images = scraper.scrape_images(custom_search_terms=args.terms)
        scraper.close()
        
        print(f"Successfully downloaded {len(downloaded_images)} images")
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        print(f"Script failed with error: {e}")