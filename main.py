# Enhanced Backend for SellFromAnywhere.com Audit Tool
# Requirements: Flask, requests, beautifulsoup4, nltk, pillow
# pip install flask requests beautifulsoup4 nltk pillow

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import time

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) so that the frontend hosted on
# sellfromanywhere.com can make requests to this backend without being blocked
# by the browser's same-origin policy. In production you may want to restrict
# the allowed origins to the domain(s) that will consume this API.
CORS(app)

# Ensure the Vader lexicon is available for sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# HTTP headers to mimic a browser and avoid request blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103 Safari/537.36"
}

# Simple in-memory rate limiting to prevent abuse of the free audit endpoint.
# This dictionary maps client IP addresses to the number of audits performed.
# When the count exceeds MAX_AUDITS_PER_IP, the API returns a 429 status and a
# message instructing the user to contact the business for additional audits.
audit_counts = {}
MAX_AUDITS_PER_IP = 3

@app.before_request
def limit_audit_requests():
    # Only rate limit the audit endpoint on POST requests
    if request.endpoint == 'audit' and request.method == 'POST':
        ip = request.remote_addr or 'unknown'
        count = audit_counts.get(ip, 0)
        if count >= MAX_AUDITS_PER_IP:
            return (
                jsonify(
                    success=False,
                    message="Free audit limit exceeded. Please contact us for additional audits.",
                ),
                429,
            )
        # Increment the count for this IP
        audit_counts[ip] = count + 1


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
    """Generate a detailed audit report for an Amazon listing.

    This function analyzes the scraped details and reviews to generate
    meaningful, prioritized suggestions. The report highlights the most
    impactful improvements a seller can make to boost conversion and
    customer satisfaction.
    """
    avg_sent = analyze_reviews_sentiment(reviews)
    sentiment_summary = (
        "positive" if avg_sent > 0.2 else "neutral" if avg_sent > -0.2 else "negative"
    )
    # Start with a perfect score and deduct points for each issue.
    listing_score = 100
    recommendations = []
    title = details['title']
    bullets = details['bullets']
    description = details['description']
    images = details['images']

    # Title analysis
    if not title:
        recommendations.append("Title is missing. Add a descriptive title including key features and benefits.")
        listing_score -= 25
    else:
        title_length = len(title)
        if title_length < 50:
            recommendations.append(
                f"Your title is only {title_length} characters. Aim for 60–80 characters and include relevant keywords."
            )
            listing_score -= 10
        elif title_length > 200:
            recommendations.append(
                "Your title is very long. Shorten it to under 200 characters for better readability."
            )
            listing_score -= 5

    # Bullet point analysis
    num_bullets = len(bullets)
    if num_bullets == 0:
        recommendations.append("No bullet points found. Add 5 concise bullet points highlighting key product benefits.")
        listing_score -= 20
    elif num_bullets < 5:
        recommendations.append(f"Only {num_bullets} bullet points detected. Aim for 5 bullet points with specific features and benefits.")
        listing_score -= 10
    else:
        for idx, bullet in enumerate(bullets[:5], start=1):
            if len(bullet.split()) < 7:
                recommendations.append(
                    f"Bullet {idx} is too short. Expand it to clearly describe the feature or benefit."
                )
                listing_score -= 2

    # Description analysis
    if not description:
        recommendations.append("Description is missing. Add a detailed description covering product benefits, usage, and brand story.")
        listing_score -= 25
    elif len(description) < 150:
        recommendations.append(
            "Your product description is quite short. Expand it to at least 150 words to address customer questions and concerns."
        )
        listing_score -= 10

    # Image analysis
    num_images = len(images)
    if num_images == 0:
        recommendations.append("No images found. Add high-quality product images showing multiple angles and lifestyle context.")
        listing_score -= 20
    elif num_images < 3:
        recommendations.append(
            f"Only {num_images} images detected. Include at least 3 high-resolution images with different angles and use cases."
        )
        listing_score -= 10

    # Review sentiment analysis
    if avg_sent < -0.2:
        recommendations.append("Customer reviews are mostly negative. Investigate common complaints and address them in the listing and product improvements.")
        listing_score -= 15
    elif avg_sent < 0.2:
        recommendations.append("Customer sentiment is mixed. Highlight positive feedback in the description and address common issues in the Q&A section.")
        listing_score -= 5

    # General improvement recommendations
    recommendations.append(
        "Consider professional services: keyword research & SEO, A+ content design, enhanced brand content, review management, image enhancement, competitor benchmarking, backend keyword optimization, and PPC audit."
    )

    # Ensure the score is within 0–100
    listing_score = max(0, min(100, listing_score))

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
    """Generate a detailed audit report for a Shopify product.

    The report highlights important improvements a merchant can make to
    increase conversion and trust on their product page.
    """
    listing_score = 100
    recommendations = []
    if not title:
        recommendations.append("Product title is missing. Add a clear, concise title with keywords.")
        listing_score -= 25
    elif len(title) < 50:
        recommendations.append(
            f"Product title is only {len(title)} characters. Consider adding more detail and relevant keywords."
        )
        listing_score -= 10

    if not description:
        recommendations.append("Description is missing. Provide a thorough description covering features, benefits, and usage.")
        listing_score -= 25
    elif len(description) < 150:
        recommendations.append(
            "Description is brief. Expand it to at least 150 words to improve SEO and answer customer questions."
        )
        listing_score -= 10

    num_images = len(images)
    if num_images == 0:
        recommendations.append("No images found. Add multiple high-quality images showing the product from various angles.")
        listing_score -= 20
    elif num_images < 3:
        recommendations.append(f"Only {num_images} image(s) found. Include at least 3 images, including lifestyle shots and close-ups.")
        listing_score -= 10

    recommendations.append(
        "Enhance your listing further by optimizing SEO keywords, using professional product photography, adding customer reviews or testimonials, and refining descriptions with storytelling."
    )
    listing_score = max(0, min(100, listing_score))
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