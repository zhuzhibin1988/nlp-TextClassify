import time

from sentiment.opinion_extraction import OpinionExtraction

if __name__ == "__main__":
    oe = OpinionExtraction(sentenceFile="../data/comment")
    startTime = time.time()
    sortedKeys = oe.extractor()
    print("Take time", time.time() - startTime)
    for sortK in sortedKeys:
        str = []
        for op in sortK[1]:
            str.append(op.opinion)
        print("%s: %s" % (sortK[0], ",".join(str)))