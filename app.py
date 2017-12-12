import flask
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():


    return flask.send_file('/home/tomnomnom12/depthDemo1/my.png',  mimetype='image/gif')
