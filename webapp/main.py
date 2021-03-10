#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask,request, render_template, url_for, redirect
import pandas as pd

from wit import Wit
from flask_bootstrap import Bootstrap

import os
import uuid

CSV = 'audio_files.csv'
ACCESS_TOKEN = "QCT25REOIGYORDXTUB4N4PUO3LA6T63H"
MAX_TASKS = 5
df = None

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_id = uuid.uuid4()
        client = Wit(access_token=ACCESS_TOKEN)
        df = df.append({'files': [], 'num_tasks': 0}).set_value(user_id, files, []).set_value(user_id, num_tasks, 0)
        df.to_csv(CSV)
        return redirect(url_for('start', user_id=user_id, client=client))
    else:
        return render_template("index.html", user_id=user_id, url=url_for('response', user_id=user_id))

@app.route("/start", methods=["GET", "POST"])
def start(user_id, client):
    if request.method == "POST":
        return redirect(url_for('response', user_id=user_id, client=client, num_response=0, request="POST"))
    else:
        # check if the user has already done all the tasks asked of them
        if df.iloc[user_id]['num_tasks'] == MAX_TASKS:
            return render_template("success.html", user_id=user_id)
        else:
            # get a new task for the client to conduct
            # imagine that there's a list of icons here, pick one
            return render_template("start.html", image=image)

@app.route("/response/", methods=["POST"])
def response(user_id, client, num_response):
    f = request.files['audio_data']
    resp = None
    
    filename = '{}_{}_{}.wav'.format(user_id, task_type, num_response)
    with open(filename, 'wb') as audio:
        resp = client.speech(audio, {'Content-Type': 'audio/wav'})
        f.save(audio)
    df = df.set_value(user_id, files, df.iloc[user_id]['files'].append(filename))
    df.to_csv(CSV)
    print('file uploaded successfully')

    # if 'next task' button is hit
    if request.form['Next Task'] == 'next task':
        df[df['user'] == user_id]['num_tasks'] += 1
        return render_template("start.html")

    return render_template("response.html", response=resp, task_type=task_type, num_response=num_response, user_id=user_id, request="POST")

@app.route("/success/")
def success(user_id):
    return render_template("success.html", user_id=user_id)
        

if __name__ == "__main__":
    if not os.path.isfile(CSV):
        df = pd.DataFrame(columns=['files', 'num_tasks'])
    else:
        df = pd.read_csv(CSV)
    app.run(debug=True)
