# Enhanced Backend for SellFromAnywhere.com Audit Tool
# Requirements: Flask, requests, beautifulsoup4, nltk, pillow
# pip install flask requests beautifulsoup4 nltk pillow

from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import time

# Initialize the Flask application
app = Flask(__name__)

# Ensure the Vader lexicon is available for sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# HTTP headers to mimic a browser and avoid request blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103 Safari/537.36"
}


def extract_amazon_listing_details(asin: str):
    """Scrape basic listing details from an Amazon product page."""
    url = f"https://www.amazon.com/dp/{asin}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tag = soup.select_one('#productTitle')
    title = title_tag.get_text(strip=True) if title_tag else ""
    bullet_tags = soup.select('#feature-bullets ul li span')
    bullets = [b.get_text(strip=True) for b in bullet_tags if b.get_text(strip=True)]
    desc_tag = soup.select_one('#productDescription')
    description = desc_tag.get_text(strip=True) if desc_tag else ""
    # Collect up to 5 product image URLs
    images = []
    for img in soup.find_all('img'):
        src = img.get('src')
        if src and 'jpg' in src and src not in images:
            images.append(src)
        if len(images) >= 5:
            break
    return {
        "title": title,
        "bullets": bullets,
        "description": description,
        "images": images,
    }


def extract_reviews_from_amazon(asin: str, max_reviews: int = 100):
    """Collect up to `max_reviews` review texts from Amazon."""
    reviews = []
    page = 1
    while len(reviews) < max_reviews:
        review_url = (
            f"https://www.amazon.com/product-reviews/{asin}/?pageNumber={page}&reviewerType=all_reviews"
        )
        res = requests.get(review_url, headers=HEADERS)
        if res.status_code != 200:
            break
        soup = BeautifulSoup(res.text, 'html.parser')
        page_reviews = [
            rev.get_text(strip=True) for rev in soup.select('span.review-text-content span')
        ]
        if not page_reviews:
            break
        reviews.extend(page_reviews)
        page += 1
        time.sleep(1)  # Be courteous to Amazon's servers
    return reviews[:max_reviews]


def analyze_reviews_sentiment(reviews):
    """Return the average compound sentiment score for a list of reviews."""
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(r)['compound'] for r in reviews]
    if not scores:
        return 0
    return sum(scores) / len(scores)


def audit_amazon_content(details: dict, reviews: list):
    """Generate a simple audit report for an Amazon listing."""
    avg_sent = analyze_reviews_sentiment(reviews)
    sentiment_summary = (
        "positive" if avg_sent > 0.2 else "neutral" if avg_sent > -0.2 else "negative"
    )
    listing_score = random.randint(50, 90)
    recommendations = []
    title = details['title']
    bullets = details['bullets']
    description = details['description']
    images = details['images']

    # Title suggestions
    if not title or len(title) < 50:
        recommendations.append(
            "Improve title with relevant keywords and more detail."
        )
    # Bullet points suggestions
    if len(bullets) < 5:
        recommendations.append(
            "Add more detailed bullet points highlighting product features."
        )
    # Description suggestions
    if not description or len(description) < 100:
        recommendations.append(
            "Expand the product description to address customer concerns."
        )
    # Images suggestions
    if len(images) < 3:
        recommendations.append(
            "Add high-quality images showing different angles of the product."
        )
    # Sentiment-based suggestion
    if avg_sent < 0:
        recommendations.append(
            "Address negative reviews by improving product quality and shipping."
        )
    # General services suggestions
    recommendations.append(
        "Consider services like SEO optimization, A+ content design, review management, image enhancement, competitor benchmarking, backend keyword optimization, and PPC audit."
    )

    return {
        "listing_score": listing_score,
        "sentiment_summary": sentiment_summary,
        "recommendations": "\n".join(
            f"{i + 1}. {rec}" for i, rec in enumerate(recommendations)
        ),
    }


def fetch_shopify_details(url: str):
    """Scrape basic product information from a Shopify product page."""
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else ""
    meta_desc = soup.find('meta', {'name': 'description'})
    description = meta_desc.get('content', '') if meta_desc else ""
    if not description:
        desc_tag = soup.find('div', {'class': 'product-description'}) or soup.find('p')
        description = desc_tag.get_text(strip=True) if desc_tag else ""
    images = []
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src')
        if src and 'jpg' in src and src not in images:
            images.append(src)
        if len(images) >= 5:
            break
    return {
        "title": title,
        "description": description,
        "images": images,
    }


def audit_shopify_content(title: str, description: str, images: list):
    """Generate a simple audit report for a Shopify product."""
    listing_score = random.randint(50, 90)
    recommendations = []
    if not title:
        recommendations.append("Add a clear product title.")
    if not description or len(description) < 100:
        recommendations.append(
            "Add detailed product description with features and benefits."
        )
    if len(images) < 3:
        recommendations.append(
            "Add high-quality images to showcase the product."
        )
    recommendations.append(
        "Consider optimizing SEO, improving product images, adding customer reviews, and refining descriptions."
    )
    return {
        "listing_score": listing_score,
        "sentiment_summary": "",
        "recommendations": "\n".join(
            f"{i + 1}. {rec}" for i, rec in enumerate(recommendations)
        ),
    }


def is_asin(value: str):
    """Check whether a given string is a plausible ASIN."""
    import re
    return bool(re.match(r'^[A-Z0-9]{8,10}$', value.strip(), re.IGNORECASE))


@app.route('/audit', methods=['POST'])
def audit():
    """Endpoint to audit a listing given an ASIN or Shopify URL."""
    data = request.get_json(silent=True) or {}
    input_value = data.get('input', '').strip()
    if not input_value:
        return jsonify(success=False, message="No input provided."), 400
    try:
        if is_asin(input_value):
            details = extract_amazon_listing_details(input_value)
            if not details:
                return (
                    jsonify(success=False, message="Failed to fetch product details from Amazon."),
                    500,
                )
            reviews = extract_reviews_from_amazon(input_value, max_reviews=100)
            audit_report = audit_amazon_content(details, reviews)
            return jsonify(
                success=True,
                listing_score=audit_report['listing_score'],
                sentiment_summary=audit_report['sentiment_summary'],
                recommendations=audit_report['recommendations'],
            )
        elif input_value.startswith('http'):
            shop_details = fetch_shopify_details(input_value)
            if not shop_details:
                return (
                    jsonify(success=False, message="Failed to fetch Shopify page."),
                    500,
                )
            audit_report = audit_shopify_content(
                shop_details['title'], shop_details['description'], shop_details['images']
            )
            return jsonify(
                success=True,
                listing_score=audit_report['listing_score'],
                sentiment_summary=audit_report['sentiment_summary'],
                recommendations=audit_report['recommendations'],
            )
        else:
            return (
                jsonify(success=False, message="Invalid input format. Use a valid ASIN or Shopify store URL."),
                400,
            )
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500


if __name__ == '__main__':
    # When running locally or on Render, bind to the appropriate port
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)