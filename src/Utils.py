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
ATTRS = ["score", "body", "ups", "downs", "controversiality"]


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
    f.close()
    return list(comments.values())


def comments2df(comments, first_k=-1, columns=COLUMNS):
    """
    :param comments: list of comments
    :param first_k: only first k comments (default all)
    :param columns: selected columns names from the comments
    :return: dataframe of the comments
    """
    df = pd.DataFrame.from_dict(comments[:first_k] if first_k > 0 else comments)
    return df[columns] if columns else df


def sentiment(text):
    return TextBlob(text).sentiment


def _calculate_users_similarity(user_subreddit_weight, subreddit_weight):
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


class RedditNetworkUtils(object):
    def __init__(self):
        """
        :param ntw: network object
        """
        self.ntw = nx.Graph()
        self.mapped_comments = None
        self.users_similarity, self.users_common_e = None, None

    def read_comments_into_network(self, filename, from_key, to_key, node_key="author", link_key="parent_id",
                                   attrs=ATTRS, maxsize=1e3):
        """
        :param filename: data filename
        :param node_key: field of the comment by which nodes should be defined
        :param from_key: field of the comment by which incoming edge should be defined
        :param to_key: field of the comment by which outgoing edge should be defined
        :param link_key: in case from and to keys have different data than node key,
        specify which field links the node to its edge, default "link_id"
        :param attrs: list of field names to add as edges attributes
        :param maxsize: maximum size of comments (1e3)
        :return:
        """
        if link_key is not None and node_key is not None:
            self.mapped_comments = RedditNetworkUtils.map_comments(filename, node_key, link_key, maxsize)

        with open(filename) as f:
            for i, line in enumerate(f):
                self.add_comment_to_network(json.loads(line), from_key, to_key, link_key, attrs)
                if i+1 == maxsize:
                    break

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

        if self.ntw.has_edge(f, t):
            self.ntw.edges[(f, t)]['w'] += 1
            self.ntw.edges[(f, t)]["subjectivity"] += sentiment(comment["body"]).subjectivity
            self.ntw.edges[(f, t)]["polarity"] += sentiment(comment["body"]).polarity
        else:
            self.ntw.add_edge(f, t, w=1, subjectivity=0, polarity=0)

        for attr in attrs:
            if attr != "body":
                self.ntw.edges[(f, t)][attr] = comment[attr]

        for e in self.ntw.edges:
            for i in ["subjectivity", "polarity"]:
                val = self.ntw.edges[e][i]
                self.ntw.edges[e][i] = val / float(self.ntw.edges[e]['w'])

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
    def read_comments_into_subreddits_dict(filename, maxsize=1e3):
        """
        :param filename: data filename
        :param maxsize: max number of comments
        :return: user_subreddit_weight # {user: {subreddit: number of posts in that subreddit}}
                 subreddit_weight # {subreddit: total number of posts}
        """
        user_subreddit_weight = dict()  # {user: {subreddit: number of posts by that user in that subreddit}}
        subreddit_weight = dict()  # {subreddit: total number of posts in that subreddit}

        users_subjectivity = dict()  # {user: avg subjectivity}
        subreddit_subjectivity = dict()  # {subreddit: subjectivity}

        with open(filename) as f:
            for i,  l in enumerate(f):
                comment = json.loads(l)
                user = comment["author"]
                subreddit = comment["subreddit"]
                sub = TextBlob(comment['body']).sentiment.subjectivity

                # updating user-subreddit connection
                user_subreddit_weight.setdefault(user, dict())
                user_subreddit_weight[user].setdefault(subreddit, 0)
                user_subreddit_weight[user][subreddit] += 1

                # updating subreddit size
                subreddit_weight.setdefault(subreddit, 0)
                subreddit_weight[subreddit] += 1

                # updating user subjectivity
                users_subjectivity.setdefault(user, [0, 0])
                users_subjectivity[user][0] += sub
                users_subjectivity[user][1] += 1

                # updating subreddit subjectivity
                subreddit_subjectivity.setdefault(subreddit, 0)
                subreddit_subjectivity[subreddit] += sub

                if i + 1 == maxsize:
                    break
        for u, (sub, count) in users_subjectivity.items():
            users_subjectivity[u] = sub/float(count)

        for subreddit, subjectivity in subreddit_subjectivity.items():
            subreddit_subjectivity[subreddit] = subjectivity / subreddit_weight[subreddit]

        print("Users: {0}".format(len(user_subreddit_weight)))
        print("Subreddit: {0}".format(len(subreddit_weight)))
        return user_subreddit_weight, subreddit_weight, users_subjectivity, subreddit_subjectivity

    def calculate_users_similarity(self, filename, maxsize=1e3):
        user_subreddit_weight, subreddit_weight, users_subjectivity, subreddit_subjectivity = RedditNetworkUtils.\
            read_comments_into_subreddits_dict(filename, maxsize)

        users_similarity, users_common_e = _calculate_users_similarity(user_subreddit_weight, subreddit_weight)
        return users_similarity, users_common_e, users_subjectivity, subreddit_subjectivity


# rnu = RedditNetworkUtils()
# rnu.read_comments_into_network("../data/RC_2013-02", "parent_id", "link_id")


# rnu = RedditNetworkUtils()
# users_similarity, users_common_e = rnu.calculate_users_similarity("../data/RC_2013-02", maxsize=1e3)