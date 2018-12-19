from __future__ import division, print_function, absolute_import
from warnings import warn
import numpy as np

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
                is_new_e = e not in users_jaccard[key]
                users_jaccard[key].add(e)

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


class RedditNetworkUtils(object):
    def __init__(self, directed=True, subj_key="subjectivity", pol_key="polarity",
                 subreddit_key="subreddit", weight_key='w'):
        """
        :param ntw: network object
        """
        self.ntw = nx.DiGraph() if directed else nx.Graph()
        self.mle_similarity, self.baseline_similarity = None, None  # {(u1, u2): similarity}

        self.subj_key = subj_key
        self.pol_key = pol_key
        self.subreddit_key = subreddit_key
        self.weight_key = weight_key

        self.nodes_ids = dict()  # {node: id}

    def read_comments_into_network(self, filename, from_key, to_key, node_key="author", link_key="parent_id",
                                   attrs=ATTRS, metrics=METRICS, maxsize=1e3, calculate_users_similarity=False,
                                   reverse_edges=False):
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
        :param reverse_edges: boolean, this is a way around to handle data sparsity
        e.g. if we specify edges to be (link_id -> parent_id) we need to specify link_key as link_id, but that results
        in all edges being between the same authors. So we specify link_key as parent_id, then construct the network as parent_id to link_id
        :return:
        """
        comments_authors = None
        if link_key is not None and node_key is not None:
            comments_authors = map_linkid_to_authors(filename, node_key, link_key, maxsize)

        if calculate_users_similarity:
            user_subreddit_weight = dict()  # {user: {subreddit: number of posts by that user in that subreddit}}
            subreddit_weight = dict()  # {subreddit: total number of posts in that subreddit}

        subreddits = set()

        with open(filename) as data:
            self.nodes_ids = dict()  # {node: id}

            for i, l in enumerate(data):
                if i == maxsize:
                    break

                comment = json.loads(l)
                author = comment["author"]

                if author == "[deleted]":
                    continue

                f, t = comment[from_key], comment[to_key]

                if comments_authors is not None:
                    try:
                        f, t = comments_authors[f], comments_authors[t]
                    except KeyError:
                        continue

                if f == t:
                    continue

                self.nodes_ids.setdefault(f, len(self.nodes_ids))
                self.nodes_ids.setdefault(t, len(self.nodes_ids))

                if reverse_edges:
                    f, t = t, f
                self.add_comment_to_network(self.nodes_ids[f], self.nodes_ids[t], comment, attrs)

                subreddit = comment["subreddit"]

                subreddits.add(subreddit)

                if calculate_users_similarity:
                    user = self.nodes_ids[f]

                    # updating user-subreddit connection
                    user_subreddit_weight.setdefault(user, dict())
                    user_subreddit_weight[user].setdefault(subreddit, 0)
                    user_subreddit_weight[user][subreddit] += 1

                    # updating subreddit size
                    subreddit_weight.setdefault(subreddit, 0)
                    subreddit_weight[subreddit] += 1

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
        # {node:
        # [count,
        # mle-scaled subjectivity,
        # mle-scaled polarity,
        # bl-scaled subjectivity,
        # bl-scaled polarity]}
        node_sentiments = dict()
        for f, t, data in self.ntw.edges(data=True):
            node_sentiments.setdefault(f, [0, 0, 0])

            subj=data["subjectivity"]
            pol=data["polarity"]

            ix = tuple(sorted([f, t]))
            mle_sim = mle_similarity[ix]
            bl_sim = baseline_similarity[ix]

            node_sentiments[f][0] += 1

            node_sentiments[f][1] += mle_sim * subj
            node_sentiments[f][2] += mle_sim * pol
            node_sentiments[f][3] += bl_sim * subj
            node_sentiments[f][4] += bl_sim * pol

        for n in node_sentiments:
            count, mle_subj, mle_pol, bl_subj, bl_pol = node_sentiments[n]

            self.ntw.nodes[n]["mle_subj"] = mle_subj / count
            self.ntw.nodes[n]["mle_pol"] = mle_pol / count
            self.ntw.nodes[n]["bl_subj"] = bl_subj / count
            self.ntw.nodes[n]["bl_pol"] = bl_pol / count

        self.mle_similarity, self.baseline_similarity = mle_similarity, baseline_similarity

    def add_comment_to_network(self, f, t, comment, attrs):
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

        for (user, d), (_, ind), (_, outd) in zip(self.ntw.degree(),
                                                  self.ntw.in_degree(),
                                                  self.ntw.out_degree()):

            self.ntw.nodes[user][self.subj_key] = users_subjectivity[user] / outd if user in users_subjectivity else 0
            self.ntw.nodes[user][self.pol_key] = users_polarity[user] / outd if user in users_polarity else 0

            self.ntw.nodes[user]["deg"] = d
            self.ntw.nodes[user]["indeg"] = ind
            self.ntw.nodes[user]["outdeg"] = outd


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


    def augment_nodes_similarities(self):
        for u1 in self.ntw.nodes:
            for u2 in nx.neighbors(self.ntw, u1):
                mle = self.mle_similarity[tuple(sorted([u1, u2]))]
                bl = self.baseline_similarity[tuple(sorted([u1, u2]))]


# rnu = RedditNetworkUtils()
# rnu.read_comments_into_network("../data/RC_2013-02", "parent_id", "link_id", maxsize=1e3,
#                                calculate_users_similarity=True)