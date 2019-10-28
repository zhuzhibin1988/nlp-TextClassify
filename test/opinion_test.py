import unittest


class OpinionTest(unittest.TestCase):
    def test_smart_split_doc(self):
        import extract.analysis as aly
        with open("../data/comment3", "r", encoding="utf-8") as corpus:
            for line in corpus:
                try:
                    sub_doc_list = aly.smart_split_doc(line.strip())
                    for sub_doc in sub_doc_list:
                        print(sub_doc)
                        segment = aly.segmentor.segment(sub_doc)
                        postag = aly.postagger.postag(segment)
                        segment = [term for term in segment]
                        postag = [pos for pos in postag]

                        dependency = aly.parser.parse(segment, postag)
                        term_pos_list = []
                        for n, term in enumerate(segment):
                            term_pos_list.append(term + "(" + postag[n] + "-" + dependency[n].relation + ")")
                        print("/".join(term_pos_list))
                except Exception:
                    print("exception: ", line.strip())

    def test_extract_opinion(self):
        import extract.analysis as aly
        aly.load_corpus()
        model = aly.train_model()
        with open("../data/comment", "r", encoding="utf-8") as corpus:
            for line in corpus:
                line = line.strip()
                # try:
                # opinion = aly.extract_opinion(line, model)
                opinion_list = aly.extract_opinion3(line)
                print(line, ":", opinion_list)
                # except Exception:
                #     print("exception: ", line)