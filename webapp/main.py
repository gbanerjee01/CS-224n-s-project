#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask,request, render_template, url_for, redirect
import pandas as pd

from wit import Wit
from flask_bootstrap import Bootstrap

import os
import uuid

CSV = 'audio_files.csv'
ACCESS_TOKEN = 158097242823014
df = None

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_id = uuid.uuid4()
        client = Wit(access_token=user_id)
        return redirect(url_for('response', user_id=user_id, client=client, num_response=0))
    else:
        return render_template("index.html", user_id=user_id, url=url_for('response', user_id=user_id))

@app.route("/response/", methods=["POST"])
def response(user_id, client, num_response):
    if request.method == "POST":
        f = request.files['audio_data']
        resp = None
        
        filename = '{}_{}.wav'.format(user_id, num_response)
        with open(filename, 'wb') as audio:
            resp = client.speech(audio, {'Content-Type': 'audio/wav'})
            f.save(audio)
        df['user'] = user_id
        df['file'] = filename
        df.to_csv(CSV)
        print('file uploaded successfully')
        return render_template("response.html", response=resp, num_response=num_response, user_id=user_id, request="POST")
    # how to determine whether the conversation is over?

@app.route("/success/")
def success(user_id=None):
    return render_template("success.html", user_id=user_id)
        

if __name__ == "__main__":
    if not os.path.isfile(CSV):
        df = pd.DataFrame(columns=['user', 'file'])
    else:
        df = pd.read_csv(CSV)
    app.run(debug=True)
