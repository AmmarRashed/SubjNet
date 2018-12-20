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

# # Reporting runtime
# import time


COLUMNS = ["id", "author", "controversiality", "ups", "score", "link_id", "parent_id"]

ATTRS = ["score", "body", "ups", "downs", "controversiality", "subreddit"]

METRICS = {
    "degree":nx.degree,
    "betweenness":nx.betweenness_centrality,
    "closeness":nx.closeness_centrality,
    "eigenvector":nx.eigenvector_centrality,
    "clustering coeff":nx.clustering
}


def read_comments(filename, maxsize=10):
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


def get_users_similarity(filename, max_size=1e4):
    users_ids = dict()  # {user: id}

    links_authors = dict()  # {link: {user: number of posts in that link by that user}}

    users_posts_counts = dict()  # {user: number of posts}
    users_sentiments = dict()  # {user: [avg polarity, avg subjectivity]}
    users_subreddits = dict()  # {user: {subreddit: #  user's posts in that subreddit}}

    mle_similarities = dict()  # {e: mle similarity}
    jaccard_similarities = dict()  # {e:jaccard similarity}

    with open(filename) as f:
        for i, l in enumerate(f):
            comment = json.loads(l)
            author = comment["author"]
            link = comment["link_id"]
            subreddit = comment["subreddit"]
            sent = sentiment(comment["body"])

            users_ids.setdefault(author, len(users_ids))
            user = users_ids[author]

            users_posts_counts.setdefault(user, 0)
            users_posts_counts[user] += 1

            links_authors.setdefault(link, dict())
            links_authors[link].setdefault(user, 0)
            links_authors[link][user] += 1

            users_sentiments.setdefault(user, [0., 0.])
            users_sentiments[user][0] += sent.polarity
            users_sentiments[user][1] += sent.subjectivity

            users_subreddits.setdefault(user, dict())
            users_subreddits[user].setdefault(subreddit, 0)
            users_subreddits[user][subreddit] += 1

            if i + 1 == max_size: break

        for link, users in links_authors.items():
            for u1, e_u1 in users.items():
                for u2, u2_e in users.items():
                    if u1 == u2: continue
                    p_e_u1 = float(e_u1) / users_posts_counts[u1]
                    p_u2_e = float(u2_e) / sum(users.values())

                    mle_similarities.setdefault((u1, u2), 1)
                    mle_similarities[(u1, u2)] *= (1 - p_e_u1 * p_u2_e)

                    e = tuple(sorted((u1, u2)))
                    jaccard_similarities.setdefault(e, 0)
                    jaccard_similarities[e] += 1

        for e, w in jaccard_similarities.items():
            u1, u2 = e
            union = len([l for l, users in links_authors.items() if u1 in users or u2 in users])
            jaccard_similarities[e] = (w, union)

        for u, (pol, subj) in users_sentiments.items():
            n = users_posts_counts[u]  # posts count
            users_sentiments[u] = [pol / n, subj / n]

        warn("Similarities are not normalized. Make sure to normalize MLE as 1-Sm, and Jaccard as Sj/Union")
        return mle_similarities, jaccard_similarities, users_sentiments, users_subreddits


def component2graph(component, graph):
    comp_graph = graph.copy()
    for node in graph.nodes:
        if node not in component:
            comp_graph.remove_node(node)

    comp_nodes_count = len(comp_graph.nodes)
    comp_edges_count = len(comp_graph.edges)

    nodes_coverage = round(100. * comp_nodes_count / len(graph.nodes()), 2)
    edges_coverage = round(100. * comp_edges_count / len(graph.edges()), 2)

    print("{0} ({1}%) nodes\t\t{2} ({3}%) edges".format(
        comp_nodes_count, nodes_coverage,
        comp_edges_count, edges_coverage,
    ))
    return comp_graph


def get_biggest_component(graph, print_first_k=5):
    components = sorted([subG for subG in nx.connected_components(graph)], key=lambda x: len(x), reverse=True)
    print("There are {0} components".format(len(components)))
    cg = None
    for i, c in enumerate(components[:print_first_k]):
        if i == 0:
            cg = component2graph(c, graph)
        else:
            component2graph(c, graph)
    return cg if cg else graph


def construct_mle_network(mle_similarities, threshold=0.3, sentiments=None, subreddits=None, normalize=True):
    g = nx.Graph()
    for (u1, u2), p in mle_similarities.items():
        sim = 1 - p if normalize else p
        if sim < threshold: continue

        g.add_edge(u1, u2, mle=sim)

    if sentiments is not None:
        for n in g.nodes():
            (pol, subj) = sentiments[n]
            g.nodes[n]["polarity"] = pol
            g.nodes[n]["subjectivity"] = subj

    print("{0} nodes".format(len(g.nodes)))
    print("{0} edges".format(len(g.edges)))

    return add_subreddits(g, subreddits) if subreddits else g


def construct_jaccard_network(jaccard_similarities, threshold=3, sentiments=None, subreddits=None):
    g = nx.Graph()
    for (u1, u2), w in jaccard_similarities.items():
        try:
            w, union_counts = w
            sim = float(w) / union_counts
        except TypeError:
            sim = w

        if w < threshold: continue

        g.add_edge(u1, u2, jac=sim, w=w)

    if sentiments is not None:
        for n in g.nodes():
            (pol, subj) = sentiments[n]
            g.nodes[n]["polarity"] = pol
            g.nodes[n]["subjectivity"] = subj

    print("{0} nodes".format(len(g.nodes)))
    print("{0} edges".format(len(g.edges)))
    return add_subreddits(g, subreddits) if subreddits else g


def add_subreddits(g, users_subreddits):
    for n in g.nodes:
        subreddits = sorted([(k, 100.*v/len(users_subreddits[n]))
                             for k, v in users_subreddits[n].items()], key=lambda x:x[1], reverse=True)
        g.nodes[n]["subreddit"] = subreddits[0][0]
        g.nodes[n]["subs"] = "-".join([str(s) for s in subreddits])
    return g


def girvan_newmann(nxg):
    g = networkx_to_snappy(nxg)
    CmtyV = snap.TCnComV()
    # start = time.time()
    modularity = snap.CommunityGirvanNewman(g, CmtyV)
    # print("Took:", time.time() - start)
    print("Modularity {0}".format(modularity))

    for i, Cmty in enumerate(CmtyV, 1):
        for n in Cmty:
            nxg.nodes[n]["gn_com"] = i

    print("Communities: {0}".format(len(CmtyV)))

    return nxg


def augment_nodes(graph, metrics=METRICS, weight=None, community_detection=False):
    for m, fun in metrics.items():
        attrs = {} if weight is None else {"weight": weight}
        m = "{0}_".format(weight) + m if weight else m
        try:
            for ix, v in dict(fun(graph, **attrs)).items():
                graph.nodes[ix][m] = v
        except nx.exception.PowerIterationFailedConvergence:
            continue
        except TypeError:
            continue

    if community_detection:
        print("Diameter: {0}".format(nx.diameter(graph)))
        print("Transitivity: {0}".format(nx.transitivity(graph)))

    return girvan_newmann(graph) if community_detection else graph
