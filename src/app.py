#!/usr/bin/env python3

"""
This file is part of the flask+d3 Hello World project.
"""
import json

import flask
import numpy as np

import olib as base


app = flask.Flask(__name__)

STATE = None

@app.route('/reset')
def read():
    ifname = '/home/probinso/git/cluster-rank-demo/src/demo-data/data_100_2_5.csv'

    with open(ifname, 'rb') as fd:
        data = np.loadtxt(fd, delimiter=',', skiprows=1).astype('float')

    io = data[:, 0:2].view(base.IOrganizer)

    global STATE
    STATE = io.selector(2)

    return base.tojson(next(STATE))


import sys


def increment(clstr_id):
    _ = next(STATE)
    return base.tojson(STATE.send(clstr_id))

@app.route('/select/<int:clstr_id>')
def select(clstr_id):
    return increment(clstr_id)

@app.route('/choose')
def choose():
    return increment(-1)


@app.route("/")
def index():
    """
    When you request the root path, you'll get the index.html template.
    """
    return flask.render_template("index.html")


@app.route("/data")
@app.route("/data/<int:ndata>")
@app.route("/data/<int:ndata>/<int:mdata>")
def data(ndata=100, mdata=1):
    """
    On request, this returns a list of ``ndata`` randomly made data points.

    :param ndata: (optional)
        The number of data points to return.

    :returns data:
        A JSON string of ``ndata`` data points.

    """
    tops = ndata * mdata
    
    x = 10 * np.random.rand(tops) - 5
    y = 0.5 * x + 0.5 * np.random.randn(tops)
    A = 10. ** np.random.rand(tops)
    c = np.random.rand(tops * mdata)
    return json.dumps([{"_id": i, "x": x[i], "y": y[i], "area": A[i],
        "color": c[i]}
        for i in range(tops)])


if __name__ == "__main__":
    import os

    port = 8000

    # Open a web browser pointing at the app.
    os.system("open http://localhost:{0}".format(port))

    # Set up the development server on port 8000.
    app.debug = True
    app.run(port=port)
