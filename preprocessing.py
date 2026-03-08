import re


def clean_text(text):

    # Remove common email headers
    text = re.sub(r"From:.*\n", "", text)
    text = re.sub(r"Subject:.*\n", "", text)

    # Remove emails and URLs
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()