from __future__ import division, print_function, absolute_import

import multiprocessing
import re
from copy import deepcopy
from warnings import warn

# reading and processing data
import pandas as pd
import numpy as np
np.random.seed(42)
import json, pickle

# For text cleaning and sentiment analysis
from textblob import TextBlob

# Network Analysis libraries
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt


try:
    import snap
except ImportError:
    warn("Could not import snap. Using NetworkX may result in slower performance for certain analyses")

# random operation
import random


# Parallelizing the process
from joblib import Parallel, delayed


from scipy.stats import pearsonr


# # Reporting runtime
# import time


COLUMNS = ["id", "author", "controversiality", "ups", "score", "link_id", "parent_id"]

ATTRS = ["score", "body", "ups", "downs", "controversiality", "subreddit"]

# "betweenness":nx.betweenness_centrality,
#     "closeness":nx.closeness_centrality,
#     "eigenvector":nx.eigenvector_centrality,
# "pagerank":nx.pagerank

METRICS = {
    "degree":nx.degree,
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


def get_batch_stats(comments):
    users_posts_counts = dict()
    links_authors = dict()
    users_sentiments = dict()
    users_subreddits = dict()
    users_mutual_links_count = dict()

    for comment in comments:
        author = comment["author"]
        link = comment["link_id"]
        subreddit = comment["subreddit"]
        sent = sentiment(comment["body"])

        # users_ids.setdefault(author, len(users_ids))
        user = int(np.int16(hash(author)))

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

        for u in links_authors[link]:
            if user == u: continue
            key = tuple(sorted([user, u]))
            users_mutual_links_count.setdefault(key, 0)
            users_mutual_links_count[key] += 1

    return users_posts_counts, links_authors, users_sentiments, users_subreddits, users_mutual_links_count


def aggregate_batches(batches_stats, agg_users_posts_counts, agg_links_authors, agg_users_sentiments, agg_users_subreddits, agg_users_mutual_links_count):

    for users_posts_counts, links_authors, users_sentiments, users_subreddits, users_mutual_links_count in batches_stats:
        for k, v in users_posts_counts.items():
            agg_users_posts_counts.setdefault(k, 0)
            agg_users_posts_counts[k] += v

        for k, v in links_authors.items():
            agg_links_authors.setdefault(k, dict())
            for i, j in v.items():
                agg_links_authors[k].setdefault(i, 0)
                agg_links_authors[k][i] += j

        for k, v in users_sentiments.items():
            agg_users_sentiments.setdefault(k, [0., 0.])
            agg_users_sentiments[k][0] += v[0]
            agg_users_sentiments[k][1] += v[1]

        for k, v in users_subreddits.items():
            agg_users_subreddits.setdefault(k, dict())
            for i, j in v.items():
                agg_users_subreddits[k].setdefault(i, 0)
                agg_users_subreddits[k][i] += j

        for k, v in users_mutual_links_count.items():
            agg_users_mutual_links_count.setdefault(k, 0)
            agg_users_mutual_links_count[k] += v

    return agg_users_posts_counts, agg_links_authors, agg_users_sentiments, agg_users_subreddits, agg_users_mutual_links_count


def get_users_similarity(filename, max_size=1e4, threshold=3, batch=False):

    links_authors = dict()  # {link: {user: number of posts in that link by that user}}

    users_posts_counts = dict()  # {user: number of posts}
    users_sentiments = dict()  # {user: [avg polarity, avg subjectivity]}
    users_subreddits = dict()  # {user: {subreddit: #  user's posts in that subreddit}}

    users_mutual_links_count = dict()  # {(u1, u2): mutual links count}

    subj_ranks = np.zeros(10)
    subj_ranks_norms = np.zeros(10)

    subreddit_comments = dict()
    if batch:
        with open(filename) as f:
            with open("../data/batch", 'w') as b:
                for i, l in enumerate(f):
                    comment = json.loads(l)
                    b.write(l.replace("\n", "")+"\n")

                    subreddit = comment["subreddit"]
                    subreddit_comments.setdefault(subreddit, set())
                    subreddit_comments[subreddit].add(comment["id"])

                    if i + 1 > max_size: break

        max_subreddit = max(subreddit_comments, key=lambda x: len(subreddit_comments[x]))
        subreddit_comments = subreddit_comments[max_subreddit]

        with open("../data/max_subreddit", "wb") as f:
            pickle.dump([max_subreddit, subreddit_comments], f, protocol=2)

        with open(filename) as f:
            with open("../data/batch", 'w') as b:
                for i, l in enumerate(f):
                    comment = json.loads(l)
                    if comment["id"] in subreddit_comments:
                        b.write(l.replace("\n", "")+"\n")
                    if i + 1 > max_size: break

    with open("../data/max_subreddit", "rb") as f:
        max_subreddit, subreddit_comments = pickle.load(f)
    print("Max subreddit is: {0}, with {1:,} comments".format(max_subreddit, len(subreddit_comments)))

    with open("../data/batch") as f:
        chunks = {i: list() for i in range(multiprocessing.cpu_count())}
        for i, l in enumerate(f):
            comment = json.loads(l)

            subj = sentiment(comment["body"]).subjectivity
            score = comment["score"]
            ind = int(subj * 10 % 10)
            subj_ranks[ind] += score
            subj_ranks_norms[ind] += 1

            chunks[i % len(chunks)].append(comment)
            if i > 0 and i % 8000 == 0:
                batches_stats = Parallel(n_jobs=len(chunks))(delayed(get_batch_stats)(comments) for comments in chunks.values())
                users_posts_counts, links_authors, users_sentiments, users_subreddits, users_mutual_links_count = \
                    aggregate_batches(batches_stats, users_posts_counts, links_authors, users_sentiments, users_subreddits, users_mutual_links_count)
                chunks = {i: list() for i in range(multiprocessing.cpu_count())}
                print("Read {0:,} comments".format(i))

            if i+1 > max_size:
                if sum([len(k) for k in chunks.values()]):
                    batches_stats = Parallel(n_jobs=len(chunks))(
                        delayed(get_batch_stats)(comments) for comments in chunks.values())
                    users_posts_counts, links_authors, users_sentiments, users_subreddits, users_mutual_links_count = \
                        aggregate_batches(batches_stats, users_posts_counts, links_authors, users_sentiments,
                                          users_subreddits, users_mutual_links_count)
                print("Read {0:,} comments".format(i))
                break
    print("{0:,} Users".format(len(users_posts_counts)))
    print("{0:,} Posts".format(len(links_authors)))
    print("{0:,} (out of {1:,}) edges with at least {2} mutual links".format(
        len([v for v in users_mutual_links_count.values() if v >= threshold]), len(users_mutual_links_count), threshold))
    print("Calculating similarities")

    for i, v in enumerate(subj_ranks_norms):
        subj_ranks_norms[i] = max(v, 1)

    subj_ranks /= subj_ranks_norms

    bgr_similarities = dict()
    jaccard_similarities = dict()

    for link, users in links_authors.items():
        for u1, e_u1 in users.items():
            for u2, u2_e in users.items():
                key = tuple(sorted([u1, u2]))
                if key in users_mutual_links_count and users_mutual_links_count[key] >= threshold:
                    p_e_u1 = float(e_u1) / users_posts_counts[u1]
                    p_u2_e = float(u2_e) / sum(users.values())

                    bgr_similarities.setdefault((u1, u2), 1)
                    bgr_similarities[(u1, u2)] *= (1 - p_e_u1 * p_u2_e)

    for e, w in bgr_similarities.items():
        u2, u1 = e
        union = len([l for l, users in links_authors.items() if u1 in users or u2 in users])
        jaccard_similarities[e] = (users_mutual_links_count[tuple(sorted([u1, u2]))], union)

    for u, (pol, subj) in users_sentiments.items():
        n = users_posts_counts[u]  # posts count
        users_sentiments[u] = [pol / n, subj / n]

    warn("Similarities are not normalized. Make sure to normalize as follows: BGR: 1-x, Jaccard as x[0]/x[1]")
    return bgr_similarities, jaccard_similarities, users_sentiments, users_subreddits, subj_ranks, subj_ranks_norms


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
    ung = graph.to_undirected()
    components = sorted([subG for subG in nx.connected_components(ung)], key=lambda x: len(x), reverse=True)
    print("There are {0} components".format(len(components)))
    cg = None
    for i, c in enumerate(components[:print_first_k]):
        if i == 0:
            cg = component2graph(c, graph)
        else:
            component2graph(c, graph)  # just to print stats of other components

    return cg if cg else graph


def construct_network(bgr_similarities, jaccard_similarities, threshold, sentiments, subreddits):
    g = nx.DiGraph()
    for (u1, u2), (w, union_counts) in jaccard_similarities.items():
        bgr = 1 - bgr_similarities[(u1, u2)] if (u1, u2) in bgr_similarities else 0
        if (u2, u1) in bgr_similarities:
            bgr += 1 - bgr_similarities[(u2, u1)]
        bgr = bgr ** (1./w)
        jac = float(w) / union_counts
        cont = False
        for k, v in threshold.items():
            if k.lower() == "jaccard":
                t = jac
            elif k.lower() == "bgr":
                t = bgr
            elif k.lower() == "w":
                t = w
            else:
                raise Exception("Unidentified threshold key")

            if t < v:
                cont = True
                break
        if cont: continue
        if g.has_edge(u2, u1):
            old_bgr = g.edges[(u2, u1)]["bgr"]
            g.edges[(u2, u1)]["bgr"] = (old_bgr + bgr) / 2.
        else:
            g.add_edge(u1, u2, bgr=bgr, jac=jac, w=w)
    g = g.to_undirected()

    for n in g.nodes():
        (pol, subj) = sentiments[n]
        g.nodes[n]["polarity"] = pol
        g.nodes[n]["subjectivity"] = subj

    print("{0:,} nodes".format(len(g.nodes)))
    print("{0:,} edges".format(len(g.edges)))
    return add_subreddits(g, subreddits)


def add_subreddits(g, users_subreddits):
    for n in g.nodes:
        subreddits = sorted([(k, 100.*v/len(users_subreddits[n]))
                             for k, v in users_subreddits[n].items()], key=lambda x:x[1], reverse=True)
        g.nodes[n]["subreddit"] = subreddits[0][0]
        g.nodes[n]["subs"] = "-".join([str(s) for s in subreddits])
    return g


def get_modularity(nxg):
    g = networkx_to_snappy(nxg)
    CmtyV = snap.TCnComV()
    modularity = snap.CommunityGirvanNewman(g, CmtyV)
    return modularity


def get_avg_metric(nxg, metric, nodes=None):
    r = 0.
    if nodes:
        for n in nodes:
            r+= nxg.nodes[n][metric]
    else:
        for n, d in nxg.nodes(data=True):
            r += d[metric]
    return r / len(nxg.nodes)


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


def get_bridges(nxg):
    g = networkx_to_snappy(nxg)
    EdgeV = snap.TIntPrV()
    snap.GetEdgeBridges(g, EdgeV)
    return [(e.GetVal1(), e.GetVal2()) for e in EdgeV]


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


def sample_ntw(g, max_nodes=500):
    old_n = len(g.nodes())
    old_e = len(g.edges())

    sample = g.copy()
    max_nodes = old_n-1 if max_nodes < 1 else max_nodes
    while len(sample.nodes) > max_nodes:
        n = random.choice(list(g.nodes))
        if sample.has_node(n):
            sample.remove_node(n)

    new_n = len(sample.nodes)
    new_e = len(sample.edges)
    print("Nodes: {0} -->  {1} ({2}%)".format(old_n, new_n, new_n*100./old_n))
    print("Edges: {0} -->  {1} ({2}%)".format(old_e, new_e, new_e*100./old_e))

    return get_biggest_component(sample, 1)


def global_disagreement_index(g):
    gdi = 0
    nodes = list(g.nodes)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            gdi += (g.nodes[nodes[i]]["opinion"] - g.nodes[nodes[j]]["opinion"])**2
    return gdi

def network_disagreement_index(g, weight="w"):
    ndi = 0
    for (i, j) in g.edges:
        ndi += g.edges[(i, j)][weight] * (g.nodes[i]["opinion"] - g.nodes[j]["opinion"])**2
    return ndi

def assign_opinions(g, prob=0.05):
    new_g = g.copy()
    for n in g.nodes:
        new_g.nodes[n]["opinion"] = int(np.random.random() < prob)
    return new_g


def propagate_opinions(g, propagator="bgr", stubbornness=0.5):
    new_g = g.copy()
    if propagator == "w":
        max_w = max([g.edges[e]["w"] for e in g.edges])
    for i, data in g.nodes(data=True):
        if not data["opinion"]: continue
        for j in nx.neighbors(g, i):
            p = g.edges[(i, j)][propagator] if not re.search("random\s*\d+", propagator) \
                else float(propagator.replace("random", ""))
            p = p / max_w if propagator == "w" else p
            # if np.random.random() * p > g.nodes[j][stubbornness]:
            if p > g.nodes[j]["subjectivity"] if stubbornness == "subjectivity" else stubbornness:
                new_g.nodes[j]["opinion"] = 1
    return new_g


def baa(ntw, propagator, T):
    opinion_network = ntw.copy()
    metrics = {"GDI": np.zeros(T),
               "NDI jac": np.zeros(T),
               "NDI bgr": np.zeros(T)
               }
    opinion_networks = list()
    for t in range(T):
        ntw_at_t = propagate_opinions(opinion_network, propagator)
        gdi = global_disagreement_index(ntw_at_t)
#        ndi_w = network_disagreement_index(ntw_at_t, "w")
        ndi_jac = network_disagreement_index(ntw_at_t, "jac")
        ndi_bgr = network_disagreement_index(ntw_at_t, "bgr")
        for k, v in zip(["GDI", "NDI jac", "NDI bgr"], [gdi, ndi_jac, ndi_bgr]):
            metrics[k][t] = v
        opinion_network = ntw_at_t
        opinion_networks.append(opinion_network)
    return propagator, metrics, opinion_networks


def biased_assimilation_analysis(g, p, propagators, T):
    opinion_ntw = assign_opinions(g, p)
    print("P: {0}\n".format(p))

    results = dict()
    opinion_networks = dict()

    chunks = Parallel(n_jobs=min(len(propagators), multiprocessing.cpu_count()))\
        (delayed(baa)(opinion_ntw, propagator, T) for propagator in propagators)
    for propagator, metrics, opinion_network_per_t in chunks:
        results.setdefault(propagator, dict())
        for k, v in metrics.items():
            results[propagator][k] = v
        opinion_networks[propagator] = opinion_network_per_t

    plot_results(results, p)
    return results, opinion_networks


def plot_results(results, p):
    for propagator, metrics in results.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_title("Propagating with {0}, $p={1}$".format(propagator, p))
        legend = list()
        for m, vals in metrics.items():
            if m.lower() == "gdi":
                ax2 = ax.twinx()
                ax2.plot(range(len(vals)), vals, 'r')
                ax2.set_ylabel(m)
                ax2.legend(["GDI"], loc=5)
            else:
                ax.plot(range(len(vals)), vals)
                legend.append(m.split()[-1])
        ax.set_xlabel("t")
        ax.set_ylabel("NDI")
        ax.legend(legend, loc=1)


def random_walk(g, max_depth=3):
    subgraph = nx.Graph()
    seed = np.random.choice(g.nodes)
    queue = [seed, "level"]
    depth = 0
    while len(queue) > 0:
        node = queue.pop(0)
        if node == "level":
            depth += 1
        elif depth < max_depth:
            for n in nx.neighbors(g, node):
                queue.append(n)
                subgraph.add_edge(node, n)
            queue.append("level")
    return subgraph


def information_diffusion(g, p, T, propagator):
    opinion_ntw = assign_opinions(g, p)
    results = list()
    opinion_network_subj = opinion_ntw.copy()
    opinion_network_rand = opinion_ntw.copy()
    for t in range(T):
        opinion_network_subj = propagate_opinions(opinion_network_subj, propagator, stubbornness="subjectivity")
        opinion_network_rand = propagate_opinions(opinion_network_rand, propagator, stubbornness=0.5)

        gdi_subj = global_disagreement_index(opinion_network_subj)
        gdi_rand = global_disagreement_index(opinion_network_rand)

        ndi_jac_subj = network_disagreement_index(opinion_network_subj, "jac")
        ndi_jac_rand = network_disagreement_index(opinion_network_rand, "jac")

        ndi_bgr_subj = network_disagreement_index(opinion_network_subj, "bgr")
        ndi_bgr_rand = network_disagreement_index(opinion_network_rand, "bgr")

        results.append((gdi_subj, gdi_rand, ndi_jac_subj, ndi_jac_rand, ndi_bgr_subj, ndi_bgr_rand))
    return results

def plot_stubbornness(results, indx1, indx2, label):
    subj = [i[indx1] for i in results]
    rand = [i[indx2] for i in results]
    f, ax = plt.subplots(figsize=(12,7))
    ax.plot(subj)
    ax.plot(rand)
    ax.legend(["Subjectivity", "Random"])
    ax.set_ylabel(label)
    ax.set_xlabel("Time")
    plt.show()

def stats(a, b=None):
    print("min: {0}\nmax: {1}\nmean: {2}\nstd: {3}".format(min(a), max(a), np.mean(a), np.std(a)))
    if b:
        r, p = pearsonr(a, b)
        print("r: {0}\tp: {1}".format(r, p))