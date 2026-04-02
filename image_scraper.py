import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
url = input("Please enter the target URL: ").strip()
try:
    count_input = input("Please enter the number of images to download (default 50): ").strip()
    TARGET_IMAGE_COUNT = int(count_input) if count_input else 50
except ValueError:
    print("Invalid number. Defaulting to 50.")
    TARGET_IMAGE_COUNT = 50

os.makedirs("downloaded_images", exist_ok=True)

print("Initializing Selenium browser...")
options = Options()
# Removed headless to bypass Cloudflare bot protection natively in foreground
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

try:
    driver = webdriver.Chrome(options=options)
    
    print(f"Fetching page: {url}")
    driver.get(url)
    
    print("Waiting for page load and Cloudflare/Bot validation to pass...")
    time.sleep(5)
    
    print(f"Scrolling to load at least {TARGET_IMAGE_COUNT} images... Please wait.")
    
    valid_image_urls = []
    seen_urls = set()
    no_new_images_count = 0
    
    while len(valid_image_urls) < TARGET_IMAGE_COUNT and no_new_images_count < 10:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Try to find and click "Show more results" if it exists (very common on Google Images)
        try:
            more_button = driver.find_element(By.CSS_SELECTOR, "input.mye4qd, .mye4qd, [value='Show more results']")
            if more_button.is_displayed():
                more_button.click()
                time.sleep(2)
        except:
            pass
            
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        images = soup.find_all('img')
        
        current_valid_count = len(valid_image_urls)
        
        for img in images:
            src = img.get('src') or img.get('data-src') or img.get('srcset')
            if not src:
                continue
                
            if ',' in src and ' ' in src:
                src = src.split(' ')[0]
                
            if src.startswith('//'):
                src = 'https:' + src
            elif not src.startswith('http'):
                src = urljoin(url, src)
                
            # Filter generic logos or small tracking pixels
            if 'logo' in src.lower() or 'icon' in src.lower() or 'base64' in src:
                pass
            else:
                if src not in seen_urls:
                    seen_urls.add(src)
                    valid_image_urls.append(src)
                    if len(valid_image_urls) >= TARGET_IMAGE_COUNT:
                        break
                        
        if len(valid_image_urls) == current_valid_count:
            no_new_images_count += 1
        else:
            no_new_images_count = 0
            
    driver.quit()
    
    print(f"Found {len(valid_image_urls)} unique raw images. Starting download...")
    
    downloaded = 0
    for i, img_url in enumerate(valid_image_urls):
        if downloaded >= TARGET_IMAGE_COUNT:
            break
            
        try:
            # Add timeout to skip dead links quickly
            img_resp = requests.get(img_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if img_resp.status_code == 200:
                with open(f"downloaded_images/image_{downloaded+1}.jpg", 'wb') as handler:
                    handler.write(img_resp.content)
                print(f"Downloaded image {downloaded+1}/{TARGET_IMAGE_COUNT}")
                downloaded += 1
        except Exception as e:
            pass
            
    print(f"Successfully finished downloading {downloaded} images into the 'downloaded_images' folder.")
except Exception as e:
    print(f"Failed to fetch using Selenium: {e}")
