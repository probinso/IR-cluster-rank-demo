# clurak

Information Retrieval - Cluster Rank Demo Harness

### Author

* __[Philip Robinson](http://github.com/probinso)__
* Special thanks to __[Brandon Rose](https://github.com/brandomr)__ [document clustering](http://brandonrose.org/clustering)
* Derived from [flask-d3-template](https://github.com/dfm/flask-d3-hello-world) by __[Dan Foreman-Mackey](http://danfm.ca/)__

### Install

* requires `pip` and `virtualenv`

```
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python src/app.py
```

### Description

I hope to show that there is value in the separation of `relevance`, `clustering`, and `ranking` in information retrieval systems. I expect that injecting clustering algorithms into the listing and user selection process will increase results diversity in a beneficial way.

`clurak` hopes to provide a web interface for displaying results, and a simple system to author your own `Observers` that provide `relevance`, `clustering`, and `ranking` models. As a user of `clurak`, you will be responsible for converting your document set into a csv of features.

The project follows the pattern bellow...

```
identify relevant documents
rank documents
select top N documents

while True:
    cluster documents
    rank clusters
    report highest ranked values for each cluster
    select a cluster
```

### Note

This may also be a study in abuse of python coroutines...
