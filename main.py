# Enhanced Backend for SellFromAnywhere.com Audit Tool (No OpenAI Dependency for Demo/Test Mode)
# Requirements: Flask, requests, beautifulsoup4, nltk, pillow
# pip install flask requests beautifulsoup4 nltk pillow

from flask import Flask, request, jsonify
import requests
import re
import nltk
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import base64
import logging
import time
import random

nltk.download('punkt')

app = Flask(__name__)

# --- CONFIGURATION ---
HEADERS = {"User-Agent": "Mozilla/5.0"}

# --- HELPERS ---
def extract_amazon_listing_details(asin):
    url = f"https://www.amazon.com/dp/{asin}"
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, 'html.parser')

    title = soup.select_one('#productTitle')
    bullets = [li.text.strip() for li in soup.select('#feature-bullets li') if li.text.strip()]
    description = soup.select_one('#productDescription')
    images = []

    # Scrape images
    img_data_script = soup.find("script", text=re.compile("ImageBlockATF"))
    if img_data_script:
        matches = re.findall(r'"hiRes":"(https:[^"\\]+\.jpg)"', img_data_script.string)
        images.extend(matches[:5])  # Limit to top 5 images

    return {
        "title": title.text.strip() if title else "",
        "bullets": bullets,
        "description": description.text.strip() if description else "",
        "images": images
    }

def extract_reviews_from_amazon(asin, max_reviews=100):
    reviews = []
    page = 1
    while len(reviews) < max_reviews:
        url = f"https://www.amazon.com/product-reviews/{asin}/?pageNumber={page}"
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, 'html.parser')
        page_reviews = [rev.text.strip() for rev in soup.select('[data-hook="review-body"]')]
        if not page_reviews:
            break
        reviews.extend(page_reviews)
        page += 1
        time.sleep(1)  # Polite crawling

    return reviews[:max_reviews]

def audit_all_content(title, bullets, description, reviews, images):
    # Placeholder fallback for environments without openai module
    score = random.randint(60, 90)
    summary = "Most reviews are positive, with some users complaining about delivery or packaging."
    suggestions = [
        "Improve title with relevant high-converting keywords.",
        "Enhance bullet points to include clear features and benefits.",
        "Rewrite product description for readability and keyword inclusion.",
        "Optimize images for quality and infographic overlay.",
        "Add backend keywords that capture related long-tail queries."
    ]
    services = [
        "Listing Copywriting", "Image Redesign", "Review Management", "Search Term Optimization", "A+ Content Design"
    ]

    return f"""
    1. Listing Score: {score}/100
    2. Sentiment Summary: {summary}
    3. Actionable Fixes:
    - {'\n    - '.join(suggestions)}

    4. Suggested Services:
    - {'\n    - '.join(services)}

    5. Competitor Benchmarking: Consider evaluating keyword gaps, pricing differences, and review count against similar top-selling listings.
    """

def is_asin(value):
    return bool(re.match(r'^B0[A-Z0-9]{8}$', value.strip(), re.IGNORECASE))

# --- ROUTE ---
@app.route('/audit', methods=['POST'])
def audit():
    data = request.json
    input_value = data.get('input', '').strip()

    if not input_value:
        return jsonify(success=False, message="No input provided."), 400

    try:
        if is_asin(input_value):
            details = extract_amazon_listing_details(input_value)
            reviews = extract_reviews_from_amazon(input_value, max_reviews=100)
            audit_report = audit_all_content(
                details['title'], details['bullets'], details['description'], reviews, details['images']
            )

            return jsonify(success=True,
                           listing_score="Included in report",
                           sentiment_summary="Included in report",
                           recommendations=audit_report)

        elif "shopify.com" in input_value:
            res = requests.get(input_value, headers=HEADERS)
            soup = BeautifulSoup(res.text, 'html.parser')
            title = soup.title.string if soup.title else "No title"
            bullets = [li.text.strip() for li in soup.select('ul li')][:5]
            description = soup.get_text()[:500]
            audit_report = audit_all_content(title, bullets, description, [], [])

            return jsonify(success=True,
                           listing_score="Included in report",
                           sentiment_summary="N/A for Shopify",
                           recommendations=audit_report)

        else:
            return jsonify(success=False, message="Invalid input format. Use a valid ASIN or Shopify store URL."), 400

    except Exception as e:
        logging.exception("Error during audit")
        return jsonify(success=False, message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
