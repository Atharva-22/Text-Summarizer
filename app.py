from __future__ import unicode_literals
from flask import Flask, render_template,url_for,request,flash

from nltk_summarization import nltk_summarizer

from bs4 import BeautifulSoup
from urllib.request import urlopen

app = Flask(__name__)



def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize',methods=['GET','POST'])
def summarize():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        raw_url = request.form['raw_url']

        if (raw_url !="") and (rawtext ==""):
            print("Summarizing URL")
            rawtext = get_text(raw_url)
            final_summary = nltk_summarizer(rawtext)
            return render_template('index.html',final_summary=final_summary)
        elif (rawtext !="") and (raw_url ==""):
            print("summarizing text")
            final_summary = nltk_summarizer(rawtext)
            return render_template('index.html',final_summary=final_summary)
        elif (rawtext =="") and (raw_url ==""):
            print("error")
            return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)