from __future__ import division, print_function, absolute_import
from warnings import warn

# reading data
import pandas as pd
import json

# Network Analysis libraries
import networkx as nx
try:
    import snap
except ImportError:
    warn("Could not import snap. Using NetworkX may result in slower performance for certain analyses")

# Parallelization libraries
from multiprocessing import cpu_count
from joblib import Parallel, delayed

# Type hints in comments
from typing import List


COLUMNS = ["id", "author", "controversiality", "ups", "score", "link_id", "parent_id"]  # type: List[str]


def read_comments(filename, maxsize=100000):
    """
    read the data as a stream and returns a list of comments
    :param filename: data filename
    :param maxsize: maxsize of the comments to read
    :return: list of comments dictionaries
    """
    comments = dict()
    with open(filename) as f:
        for i, line in enumerate(f):
            comments[i] = json.loads(line)
            if i+1 == maxsize:
                break
    return list(comments.values())


def comments2df(comments, first_k=-1, columns=COLUMNS):
    """
    :param comments: list of comments
    :param first_k: only first k comments (default all)
    :param columns: selected columns names from the comments
    :return: dataframe of the comments
    """
    return pd.DataFrame.from_dict(comments[:first_k] if first_k > 0 else comments)[columns]


class RedditNetworkUtils(object):
    def __init__(self, ntw):
        """
        :param ntw: network object
        """
        self.ntw = ntw
        self.mapped_comments = None

    def read_comments_into_network(self, filename, node_key, from_key, to_key, link_key="link_id", maxsize=1e3):
        """
        :param filename: data filename
        :param node_key: field of the comment by which nodes should be defined
        :param from_key: field of the comment by which incoming edge should be defined
        :param to_key: field of the comment by which outgoing edge should be defined
        :param link_key: in case from and to keys have different data than node key,
        specify which field links the node to its edge, default "link_id"
        :param maxsize: maximum size of comments (1e3)
        :return:
        """
        if link_key is not None:
            self.mapped_comments = RedditNetworkUtils.map_comments(filename, node_key, link_key, maxsize)

        with open(filename) as f:
            for i, line in enumerate(f):
                self.add_comment_to_network(json.loads(line), from_key, to_key, link_key)
                if i+1 == maxsize:
                    break

    def add_comment_to_network(self, comment, from_key, to_key, link_key):
        f, t = comment[from_key], comment[to_key]
        if link_key is not None:
            f = self.mapped_comments[f]
            t = self.mapped_comments[t]
        f, t = hash(f), hash(t)

        if self.ntw.has_edge(f, t):
            self.ntw.edges[(f, t)]['w'] += 1
        else:
            self.ntw.add_edge(f, t)

    @staticmethod
    def networkx_to_snappy(nxg, directed=False):
        if directed:
            g = snap.TNGraph.New()
        else:
            g = snap.TUNGraph.New()

        for n in nxg.nodes():
            g.AddNode(n)
        for f, t in nxg.edges():
            g.AddEdge(f, t)

        return g

    @staticmethod
    def map_comments(filename, node_key, link_key, maxsize):
        ids = dict()  # {link_key: id_key}
        with open(filename) as f:
            for i, line in enumerate(f):
                comment = json.loads(line)
                ids[comment[link_key]] = comment[node_key]
                if i+1 == maxsize:
                    break
        return ids
