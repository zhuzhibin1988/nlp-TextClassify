from flask import Flask
from flask import request

import similar.opinion_similar as ops
from sentiment.opinion_extract import OpinionExtractor

import json

app = Flask(__name__)


@app.route("/api/opinion_cluster", methods=["POST"])
def cluster_opinion():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    opinions = json_data.get("opinions")

    opinion_extractor = OpinionExtractor()
    new_opinions = []
    for comment in opinions:
        new_opinions += opinion_extractor.extract_opinion(comment)
    cluster = ops.opinion_cluster(new_opinions)
    return "opinions: " + str(cluster)


def main():
    app.run(host="localhost", port=5001)


if __name__ == '__main__':
    main()