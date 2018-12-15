from Utils import *


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


def read_comments_into_subreddits_dict(filename, maxsize=1e3):
    """
    :param filename: data filename
    :param maxsize: max number of comments
    :return: user_subreddit_weight # {user: {subreddit: number of posts in that subreddit}}
             subreddit_weight # {subreddit: total number of posts}
    """
    user_subreddit_weight = dict()  # {user: {subreddit: number of posts by that user in that subreddit}}
    subreddit_weight = dict()  # {subreddit: total number of posts in that subreddit}

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

            if i + 1 == maxsize:
                break

    print("Users (nodes): {0}".format(len(user_subreddit_weight)))
    print("Subreddits: {0}".format(len(subreddit_weight)))
    return user_subreddit_weight, subreddit_weight


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


def calculate_similarity_space(filename, maxsize=1e3):
    user_subreddit_weight, subreddit_weight = read_comments_into_subreddits_dict(filename, maxsize)

    users_similarity, users_common_e = _calculate_users_similarity(user_subreddit_weight, subreddit_weight)
    return users_similarity, users_common_e