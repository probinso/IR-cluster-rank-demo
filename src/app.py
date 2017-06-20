#!/usr/bin/env python3

"""
This file is part of the flask+d3 Hello World project.
"""
import json

import flask
import numpy as np

import olib as base
import preproc


app = flask.Flask(__name__)

data, names, terms = None, None, None
STATE = None

@app.route('/load')
def read():
    global data, names, terms
    data, names, terms = preproc.process('/home/probinso/git/'
                                         'cluster-rank-demo/thousand.json')
    return 'Ready'



@app.route('/reset')
def reset():
    global terms

    preproc.terms = terms
    io = base.Handler(data, names, preproc.CosKMOrganizer)

    global STATE
    STATE = io.selector(5, 4)

    return preproc.tojson(next(STATE))


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
