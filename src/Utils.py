from __future__ import division, print_function, absolute_import
from warnings import warn

# reading data
import pandas as pd
import json

# For text cleaning and sentiment analysis
from textblob import TextBlob

# Network Analysis libraries
import networkx as nx


try:
    import snap
except ImportError:
    warn("Could not import snap. Using NetworkX may result in slower performance for certain analyses")

# Parallelization libraries
from multiprocessing import cpu_count
from joblib import Parallel, delayed


COLUMNS = ["id", "author", "controversiality", "ups", "score", "link_id", "parent_id"]
ATTRS = ["score", "body", "ups", "downs", "controversiality", "subreddit"]
METRICS = {"betweenness":nx.betweenness_centrality,
           "closeness_centrality":nx.closeness_centrality,
           "eigenvector_centrality":nx.eigenvector_centrality}


def sentiment(text):
    return TextBlob(text).sentiment


def calculate_similarity_space(user_subreddit_weight, subreddit_weight):
    user_outgoing_links = {u: sum(s.values()) for u, s in user_subreddit_weight.items()}

    # Calculate Maximum Likelihood
    p_e_u = dict()  # {user: {subreddit: p(e|u)}}
    p_u_e = dict()  # {subreddit: {user: p(u|e)}}

    for u, es in user_subreddit_weight.items():
        for e, w in es.items():
            p_e_u.setdefault(u, dict())
            p_e_u[u].setdefault(e, dict())
            p_e_u[u][e] = float(w) / user_outgoing_links[u]

            p_u_e.setdefault(e, dict())
            p_u_e[e].setdefault(u, dict())
            p_u_e[e][u] = float(w) / subreddit_weight[e]

    # calculate users similarity
    del subreddit_weight

    users_jaccard = dict()  # {(u1, u2): e(u1, u2) / num of e}  # baseline similarity
    users_similarity = dict()  # {(u1, u2): Maximum Likelihood similarity}

    for ui, es in p_e_u.items():
        for e, p_ei_ui in es.items():
            for uj, p_uj_ej in p_u_e[e].items():
                if ui == uj:
                    continue
                key = tuple(sorted([ui, uj]))
                users_jaccard.setdefault(key, set())  # {(u1, u2): set of common edges}
                is_new_e = hash(e) not in users_jaccard[key]
                users_jaccard[key].add(hash(e))

                users_similarity.setdefault(key, 1)
                if is_new_e:
                    users_similarity[key] *= (1 - p_ei_ui*p_uj_ej)
    for users, sim in users_similarity.items():
        users_similarity[users] = 1 - sim
        users_jaccard[users] = float(len(users_jaccard[users])) / sum([len(user_subreddit_weight[u]) for u in users])

    return users_similarity, users_jaccard


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


def map_comments(filename, node_key, link_key, maxsize):
    ids = dict()  # {link_key: id_key}
    with open(filename) as f:
        for i, line in enumerate(f):
            comment = json.loads(line)
            ids[comment[link_key]] = comment[node_key]
            if i+1 == maxsize:
                break
    return ids


class RedditNetworkUtils(object):
    def __init__(self, directed=True, subj_key="subjectivity", pol_key="polarity",
                 subreddit_key="subreddit", weight_key='w'):
        """
        :param ntw: network object
        """
        self.ntw = nx.DiGraph() if directed else nx.Graph()
        self.mapped_comments = None
        self.mle_similarity, self.baseline_similarity = None, None

        self.subj_key = subj_key
        self.pol_key = pol_key
        self.subreddit_key = subreddit_key
        self.weight_key = weight_key

    def read_comments_into_network(self, filename, from_key, to_key, node_key="author", link_key="parent_id",
                                   attrs=ATTRS, metrics=METRICS, maxsize=1e3, calculate_users_similarity=False, update_node_key=False):
        """
        :param filename: data filename
        :param node_key: field of the comment by which nodes should be defined
        :param from_key: field of the comment by which incoming edge should be defined
        :param to_key: field of the comment by which outgoing edge should be defined
        :param link_key: in case from and to keys have different data than node key,
        specify which field links the node to its edge, default "link_id"
        :param attrs: list of field names to add as edges attributes
        :param maxsize: maximum size of comments (1e3)
        :param calculate_users_similarity: boolean default False
        :param update_node_key: boolean update dictionary {link_key: id_key}
        :return:
        """
        if link_key is not None and node_key is not None\
                and (self.mapped_comments is None or update_node_key):
            self.mapped_comments = map_comments(filename, node_key, link_key, maxsize)

        if calculate_users_similarity:
            user_subreddit_weight = dict()  # {user: {subreddit: number of posts by that user in that subreddit}}
            subreddit_weight = dict()  # {subreddit: total number of posts in that subreddit}


        subreddits = set()

        with open(filename) as f:
            for i, l in enumerate(f):
                comment = json.loads(l)
                self.add_comment_to_network(comment, from_key, to_key, link_key, attrs)

                subreddit = comment["subreddit"]

                subreddits.add(subreddit)

                if calculate_users_similarity:
                    user = comment["author"]

                    # updating user-subreddit connection
                    user_subreddit_weight.setdefault(user, dict())
                    user_subreddit_weight[user].setdefault(subreddit, 0)
                    user_subreddit_weight[user][subreddit] += 1

                    # updating subreddit size
                    subreddit_weight.setdefault(subreddit, 0)
                    subreddit_weight[subreddit] += 1

                if i + 1 == maxsize:
                    break

        for e in self.ntw.edges:
            for i in ["subjectivity", "polarity"]:
                val = self.ntw.edges[e][i]
                self.ntw.edges[e][i] = val / float(self.ntw.edges[e]['w'])

        print("Users (nodes): {0}".format(len(self.ntw.nodes)))
        print("Subreddits: {0}".format(len(subreddits)))
        print("Edges: {0}".format(len(self.ntw.edges)))
        if calculate_users_similarity:
            self.set_similarities(*calculate_similarity_space(user_subreddit_weight, subreddit_weight))

        self.aggergate_sentiment()
        self.augment_nodes(metrics)

    def set_similarities(self, mle_similarity, baseline_similarity):
        self.mle_similarity, self.baseline_similarity = mle_similarity, baseline_similarity

    def add_comment_to_network(self, comment, from_key, to_key, link_key, attrs):
        f, t = comment[from_key], comment[to_key]

        if link_key is not None:
            try:
                f = self.mapped_comments[f]
                t = self.mapped_comments[t]
            except KeyError:
                return

        f, t = hash(f), hash(t)

        if f == t:
            return

        sent = sentiment(comment["body"])

        if self.ntw.has_edge(f, t):
            self.ntw.edges[(f, t)]['w'] += 1
            self.ntw.edges[(f, t)]["subjectivity"] += sent.subjectivity
            self.ntw.edges[(f, t)]["polarity"] += sent.polarity
        else:
            self.ntw.add_edge(f, t, w=1, subjectivity=0, polarity=0)

        for attr in attrs:
            if attr != "body":
                self.ntw.edges[(f, t)][attr] = comment[attr]

    def aggergate_sentiment(self):
        """
        :param subj_key: name of subjectivity attribute, default ("subjectivity")
        :param pol_key: name of polarity attribute, default ("polarity")
        :return:
        """
        users_subjectivity = dict()  # {user: subjectivity avg}
        users_polarity = dict()  # {user: polarity avg}

        subreddit_subjectivity = dict()  # {subreddit: subjectivity avg}
        subreddit_polarity = dict()  # {subreddit: polarity avg}

        for e in self.ntw.edges:
            f, t = e
            subj = self.ntw.edges[e][self.subj_key]
            pol = self.ntw.edges[e][self.pol_key]
            subreddit = self.ntw.edges[e][self.subreddit_key]

            users_subjectivity.setdefault(f, 0)
            users_subjectivity[f] += subj

            users_polarity.setdefault(f, 0)
            users_polarity[f] += pol

            subreddit_subjectivity.setdefault(subreddit, 0)
            subreddit_subjectivity[subreddit] += subj

            subreddit_polarity.setdefault(subreddit, 0)
            subreddit_polarity[subreddit] += pol

        for (user, d), (_, ind), (_, outd) in zip(self.ntw.degree(weight=self.weight_key),
                                                  self.ntw.in_degree(weight=self.weight_key),
                                                  self.ntw.out_degree(weight=self.weight_key)):

            self.ntw.nodes[user][self.subj_key] = users_subjectivity[user] / outd if user in users_subjectivity else 0
            self.ntw.nodes[user][self.pol_key] = users_polarity[user] / outd if user in users_polarity else 0

            self.ntw.nodes[user]["deg"] = d
            self.ntw.nodes[user]["indeg"] = ind
            self.ntw.nodes[user]["outdeg"] = outd

            self.ntw.nodes[user]["w_deg"] = d
            self.ntw.nodes[user]["w_indeg"] = ind
            self.ntw.nodes[user]["w_outdeg"] = outd

    def augment_nodes(self, metrics):

        for m, fun in metrics.items():
            try:
                for ix, v in fun(self.ntw).items():
                    self.ntw.nodes[ix][m] = v
            except nx.exception.PowerIterationFailedConvergence:
                continue
            try:
                for ix, v in fun(self.ntw, weight=self.weight_key).items():
                    self.ntw.nodes[ix]["w_{0}".format(m)] = v
            except TypeError:
                continue
            except nx.exception.PowerIterationFailedConvergence:
                continue

# rnu = RedditNetworkUtils()
# rnu.read_comments_into_network("../data/RC_2013-02", "link_id", "parent_id", maxsize=5e3,
#                                calculate_users_similarity=True)