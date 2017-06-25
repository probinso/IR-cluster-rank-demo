#!/usr/bin/env python3

"""
This file is part of the flask+d3 Hello World project.
"""
import json

import flask
import numpy as np

import olib as base
import kmidf as preproc


app = flask.Flask(__name__)

data_path = '/home/probinso/git/cluster-rank-demo/tenthousand.json'

data, names, terms, query = None, None, None, None
STATE = None

@app.route("/")
def read():
    global data, names, terms, query, data_path
    data, names, terms, query = preproc.process(data_path)
    return 'Ready <a href="http://127.0.0.1:8000/reset">NEXT</a>'



@app.route('/reset')
def reset():
    global terms

    preproc.terms = terms
    io = base.Handler(data, names, terms,
                      preproc.CosKMIDFROrganizer, query)

    global STATE
    STATE = io.selector(3, 4)

    return preproc.tojson(next(STATE))


import sys


def increment(clstr_id):
    _ = next(STATE)
    return preproc.tojson(STATE.send(clstr_id))

@app.route('/select/<int:clstr_id>')
def select(clstr_id):
    return increment(clstr_id)

@app.route('/choose')
def choose():
    return increment(-1)

'''
@app.route("/")
def index():
    """
    When you request the root path, you'll get the index.html template.
    """
    return flask.render_template("index.html")
'''

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
    #global data_path
    import os

    '''
    try:
        data_path = os.path.realpath(sys.argv[1])
    except:
        print("usage: {}  <corpus_path>".format(sys.argv[0]))
    '''
    port = 8000


    # Open a web browser pointing at the app.
    os.system("open http://localhost:{0}".format(port))

    # Set up the development server on port 8000.
    app.debug = True
    app.run(port=port)
