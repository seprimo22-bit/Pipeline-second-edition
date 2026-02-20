import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)


def analyze_article(question, article):

    prompt = f"""
You are a research fact extraction assistant.

IMPORTANT:
- Do NOT repeat article text.
- Provide facts ABOUT the article:
  methodology
  assumptions
  limitations
  scientific context
  implications
  credibility indicators.

Question:
{question}

Article:
{article}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


@app.route("/", methods=["GET", "POST"])
def index():

    result = ""

    if request.method == "POST":
        question = request.form.get("question", "")
        article = request.form.get("article", "")

        if article.strip():
            result = analyze_article(question, article)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
