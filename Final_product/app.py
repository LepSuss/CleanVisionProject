from flask import Flask, request, jsonify, render_template,redirect
from datetime import datetime
import re
import pickle
from test import runmodel
import importlib.util
spec = importlib.util.spec_from_file_location("scale", "/home/pi/arduinocode/arduinocode.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import time


app = Flask(__name__)


@app.route("/")
def home():
    # Quick error handling
    try:
        return render_template('index.html')
    except:
        time.sleep(5)
        return render_template('index.html')

@app.route('/results/', methods=['POST'])
def predict():
    # Quick error handling
    try:
        # Call weight file: weight = weight_file()
        # weight_result = 'Usable'
        weight = foo.scale()
        if weight < 0:
            return render_template('index.html',result='Unusable')
        result = runmodel()
        return render_template('index.html', result=result)
    except:
        time.sleep(5)
        return render_template("index.html")

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    