"""
生成分类语料
"""

import mysql.connector as connector
from sklearn.feature_extraction.text import TfidfVectorizer


def get_connection():
    connection = connector.connect(host="127.0.0.1", user="qdtech", password="hy24lWNUL1zDVQ0Rrxfq",
                                   database="kz_decision", port=3310)
    return connection


def query(connection, sql):
    statement = connection.cursor()
    statement.execute(sql)
    rs = statement.fetchall()
    return rs


def close_connection(connection):
    connection.close()


class DataProcessor(object):

    def create_file(self):
        sql = ""
        root_path="./data"
        open()

    def get_file(self, category_id, tag):

    def data_classify(self):
        sql = "select category_ids, tag, comment from nlp_comment_data limit 1000"
        rs = query(get_connection(), sql)
        for row in rs:
            category_ids = row[0]
            tags = row[1]
            comment = row[2]