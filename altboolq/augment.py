import re
from collections import Counter, defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture


def same_case(a, b):
    if a.lower() == a:
        return b.lower()
    if a.upper() == a:
        return b.upper()
    if a[0].upper() == a[0]:
        return b.capitalize()
    return b.lower()


def replace_words(sentence, word_map, token_pattern="(?u)\\b\\w\\w+\\b"):
    pattern = re.compile(token_pattern)
    chars = []
    index = 0
    for match in pattern.finditer(sentence):
        word = match.group()
        start = match.start()
        end = match.end()
        while index < start:
            chars.append(sentence[index])
            index += 1

        token = word.lower()
        if token in word_map:
            new_word = word_map[token]
            chars += same_case(word, new_word)
        else:
            chars += word

        index = end

    while index < len(sentence):
        chars.append(sentence[index])
        index += 1

    return "".join(chars)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class GroupWeight:
    def __init__(self, glove, doc_freqs, alpha, max_doc_freq):
        self.glove = glove
        self.alpha = alpha
        self.doc_freqs = doc_freqs
        self.max_doc_freq = max_doc_freq

    def __call__(self, word_count_i, word_count_j):
        word_i, count_i = word_count_i
        word_j, count_j = word_count_j
        if word_i == word_j:
            return 1.0
        if word_i not in self.glove or word_j not in self.glove:
            return 0.0
        if self.doc_freqs[word_i] > self.max_doc_freq:
            return 0.0
        if self.doc_freqs[word_j] > self.max_doc_freq:
            return 0.0
        sim = cosine_similarity(self.glove[word_i], self.glove[word_j])
        count_ij = np.log(min(count_i, count_j))
        return sim - self.alpha * count_ij


class Grouper:
    def __init__(self, group_weight, grouping_cutoff):
        self.grouping_cutoff = grouping_cutoff
        self.group_weight = group_weight

    def gen_graph(self, term_counts):
        n = len(term_counts)
        weights = np.zeros((n, n))
        words = list(term_counts)
        word_to_index = {word: i for i, word in enumerate(words)}
        for word_i, count_i in term_counts.items():
            i = word_to_index[word_i]
            for word_j, count_j in term_counts.items():
                j = word_to_index[word_j]
                weights[i, j] = self.group_weight((word_i, count_i), (word_j, count_j))
        return words, weights

    def __call__(self, term_counts):
        words, graph = self.gen_graph(term_counts)
        graph = graph > self.grouping_cutoff
        n_components, labels = connected_components(graph)

        groups = [[] for _ in range(n_components)]
        for i, label in enumerate(labels):
            groups[label].append(words[i])
        groups = [tuple(group) for group in groups]
        groups = {word: group for group in groups for word in group}
        return groups


def get_term_freqs(doc, analyzer):
    words = analyzer(doc)
    counts = Counter(words)
    return counts_to_freqs(counts, len(words)), counts, len(words)


def get_doc_freqs(docs, analyzer):
    counts = Counter()
    for doc in docs:
        counts.update(set(analyzer(doc)))
    return counts_to_freqs(counts, len(docs)), counts, len(docs)


def counts_to_freqs(counts, total):
    return {word: count / total for word, count in counts.items()}


def get_tfidf(docs):
    tfidf = TfidfVectorizer(norm=None, sublinear_tf=True)
    X = tfidf.fit_transform(docs)
    features = tfidf.get_feature_names_out()
    features = {word: index for index, word in enumerate(features)}
    analyzer = tfidf.build_analyzer()
    words = [[word for word in set(analyzer(doc)) if word in features] for doc in docs]
    weights = []
    for i in range(len(words)):
        weight = [(X[i, features[word]], word) for word in words[i]]
        weight = sorted(weight, reverse=True)
        weights.append(weight)
    return tfidf, analyzer, weights


class IsOOV:
    def __init__(self, glove):
        self.glove = glove

    def __call__(self, word):
        if self.glove is None:
            return False
        return word not in self.glove


class Selector:
    def __init__(self, tfidf_cutoff, freq_cutoff, group_size, min_term_count, is_oov):
        self.tfidf_cutoff = tfidf_cutoff
        self.freq_cutoff = freq_cutoff
        self.group_size = group_size
        self.min_term_count = min_term_count
        self.is_oov = is_oov

    def select(self, groups, tfidf_weights, term_freqs, term_counts):
        total_freq = 0
        results = []
        seen = set()
        for score, word in tfidf_weights:
            if score < self.tfidf_cutoff:
                break

            if word in seen:
                continue

            if term_counts[word] < self.min_term_count:
                continue

            if self.is_oov(word):
                continue

            group = groups[word]
            if len(group) > self.group_size:
                continue
            for group_word in group:
                seen.add(group_word)
                total_freq += term_freqs[group_word]

            results.append(group)
            if total_freq > self.freq_cutoff:
                break

        return results


def get_group_accuracy(doc, selection, nlp):
    tokens = nlp(doc)
    groups = {word: i for i, group in enumerate(selection) for word in group}
    lemma_groups = defaultdict(set)
    for token in tokens:
        group_num = groups[token.lower_] if token.lower_ in groups else -1
        lemma_groups[token.lemma_.lower()].add((token.lower_, group_num))

    success = 0
    total = 0
    for _, lemma_group in lemma_groups.items():
        counter = Counter(group_num for _, group_num in lemma_group)
        value, majority = counter.most_common(1)[0]
        count = sum(counter.values())
        if majority == count and value == -1:
            continue
        success += majority
        total += count
    return success, total


def evaluate_selection(docs, selections, analyzer, nlp):
    lengths = np.array([len(s) for s in selections], dtype=int)
    results = {
        "non_empty": np.mean(lengths > 0),
        "avg_length": np.mean(lengths),
        "max_length": np.max(lengths),
    }

    group_accuracy = []
    selection_freqs = []
    for doc, selection in tqdm(zip(docs, selections)):
        success, total = get_group_accuracy(doc, selection, nlp)
        group_accuracy.append(1.0 if total == 0 else success / total)
        term_freqs, _, _ = get_term_freqs(doc, analyzer)
        selection_freqs.append(
            sum(term_freqs[word] for group in selection for word in group)
        )

    results["avg_freq"] = np.mean(selection_freqs)
    results["group_accuracy"] = np.mean(group_accuracy)
    results["exact_group_accuracy"] = np.mean(np.array(group_accuracy) == 1.0)
    return results


def select_dataset(df, config, glove):
    task_format = "{}\n{}"
    docs = [
        task_format.format(row["passage"], row["question"]) for _, row in df.iterrows()
    ]
    _, analyzer, tfidf_weights = get_tfidf(docs)
    doc_freqs, _, _ = get_doc_freqs(docs, analyzer)
    group_weight = GroupWeight(
        glove, doc_freqs, alpha=config["alpha"], max_doc_freq=config["max_doc_freq"]
    )
    grouper = Grouper(group_weight, grouping_cutoff=config["grouping_cutoff"])
    is_oov = IsOOV(glove if config["redact_oov"] else None)
    selector = Selector(
        tfidf_cutoff=config["tfidf_cutoff"],
        freq_cutoff=config["freq_cutoff"],
        group_size=config["group_size"],
        min_term_count=config["min_term_count"],
        is_oov=is_oov,
    )
    selections = []
    for i in tqdm(range(len(docs))):
        doc = docs[i]
        term_freqs, term_counts, _ = get_term_freqs(doc, analyzer)
        groups = grouper(term_counts)
        selection = selector.select(groups, tfidf_weights[i], term_freqs, term_counts)
        selections.append(selection)

    return docs, selections, analyzer


def augment_dataset(df, word_maps):
    rows = []
    for index, word_map in word_maps:
        row = df.iloc[index].copy(deep=True)
        row["passage"] = replace_words(row["passage"], word_map)
        row["question"] = replace_words(row["question"], word_map)
        rows.append(row)

    return pd.DataFrame(rows)


def none_filter(doc, selection):
    return True


class GroupAccuracyFilter:
    def __init__(self, nlp):
        self.nlp = nlp

    def __call__(self, doc, selection):
        success, total = get_group_accuracy(doc, selection, self.nlp)
        return success == total


def get_oov_words(doc, analyzer, glove):
    return [word for word in set(analyzer(doc)) if word not in glove]


class GetOOV:
    def __init__(self, glove):
        self.glove = glove

    def __call__(self, words):
        if self.glove is None:
            return []

        return [word for word in set(words) if word not in self.glove]


def make_mask_word_maps(docs, selections, analyzer, filterer, get_oov):
    word_maps = []
    for index, (doc, selection) in tqdm(enumerate(zip(docs, selections))):
        if not filterer(doc, selection):
            continue

        word_map = {}
        for group in selection:
            for word in group:
                word_map[word] = "redacted"

        for word in get_oov(analyzer(doc)):
            word_map[word] = "redacted"

        word_maps.append((index, word_map))
    return word_maps


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def get_vecs(glove, words):
    return np.array([glove[word] for word in words])


def get_unit_vecs(glove, words):
    return normalize(get_vecs(glove, words))


def gmm_cluster(config, selections, glove):
    gmm = GaussianMixture(
        n_components=config["gmm_n_components"],
        covariance_type=config["gmm_covariance_type"],
        n_init=config["gmm_n_init"],
        random_state=config["gmm_random_state"],
        verbose=True,
    )
    words = [word for selection in selections for group in selection for word in group]
    words = [word for word in words if word in glove]
    gmm_vecs = get_unit_vecs(glove, words)
    clusters = gmm.fit_predict(gmm_vecs)
    cluster_map = {}
    for cluster in range(np.max(clusters) + 1):
        cluster_map[cluster] = np.array(words)[clusters == cluster]
    return gmm, cluster_map


def most_similar_group(glove, words, cand_words, sample_word):
    all_new_words = []
    for cand_word in cand_words:
        new_words = []
        total_score = 0.0
        for word in words:
            if word == cand_word:
                new_word, score = sample_word, 1.0
            elif sample_word == cand_word:
                new_word, score = word, 1.0
            else:
                new_word, score = glove.most_similar(
                    positive=[sample_word, word], negative=[cand_word], topn=1
                )[0]
            new_words.append(new_word)
            total_score += score
        total_score /= len(words)
        all_new_words.append((total_score, new_words))
    score, new_words = max(all_new_words)
    return new_words, score


def gmm_resample(words, gmm, cluster_map, glove, rng):
    clusters = gmm.predict(get_unit_vecs(glove, words))
    sample_words = [
        (cluster, rng.choice(cluster_map[cluster])) for cluster in set(clusters)
    ]
    rng.shuffle(sample_words)

    all_new_words = []

    for cluster, sample_word in sample_words:
        cluster_words = np.array(words)[clusters == cluster]
        new_words, score = most_similar_group(glove, words, cluster_words, sample_word)
        new_clusters = gmm.predict(get_unit_vecs(glove, new_words))
        if np.array_equal(clusters, new_clusters):
            return new_words, 1.0, score
        cluster_score = (clusters == new_clusters).mean()
        all_new_words.append((cluster_score, score, new_words))

    cluster_score, score, new_words = max(all_new_words)
    return new_words, cluster_score, score


def make_gmm_word_maps(config, docs, selections, glove, analyzer, filterer, get_oov):
    gmm, cluster_map = gmm_cluster(config, selections, glove)

    rng = np.random.default_rng(seed=config["seed"])

    all_cluster_scores = []
    all_sim_scores = []
    word_maps = []
    for index, (doc, selection) in tqdm(enumerate(zip(docs, selections))):
        if not filterer(doc, selection):
            continue

        cluster_scores = []
        sim_scores = []
        word_map = {}
        for group in selection:
            new_group, cluster_score, sim_score = gmm_resample(
                group, gmm, cluster_map, glove, rng
            )
            for word, new_word in zip(group, new_group):
                word_map[word] = new_word
            cluster_scores.append(cluster_score)
            sim_scores.append(sim_score)

        all_cluster_scores.append(np.mean(cluster_scores))
        all_sim_scores.append(np.mean(sim_scores))

        for word in get_oov(analyzer(doc)):
            word_map[word] = "redacted"

        word_maps.append((index, word_map))

    results = {
        "exact_cluster_match": np.mean(np.array(all_cluster_scores) == 1.0),
        "avg_cluster_match": np.nanmean(all_cluster_scores),
        "avg_sim_scores": np.nanmean(all_sim_scores),
    }
    print(results)
    return word_maps


def make_word_maps(df, config, docs, selections, glove, nlp, analyzer):
    if config["filter"] == None:
        filterer = none_filter
    elif config["filter"] == "group_accuracy":
        filterer = GroupAccuracyFilter(nlp)
    else:
        raise ValueError('config["filter"] is invalid')

    get_oov = GetOOV(glove if config["redact_oov"] else None)

    if config["resample"] == "mask":
        return make_mask_word_maps(docs, selections, analyzer, filterer, get_oov)
    elif config["resample"] == "gmm":
        return make_gmm_word_maps(
            config, docs, selections, glove, analyzer, filterer, get_oov
        )
    else:
        raise ValueError('config["resample"] is invalid')


def set_defaults(config):
    config.setdefault("min_term_count", 0)
    config.setdefault("max_doc_freq", 1.0)
    config.setdefault("redact_oov", False)


def run_augment(dfs, config, loader, spacy):
    set_defaults(config)
    df = dfs[config["df"]]

    print("Loading glove and nlp...")
    glove = loader.load(config["glove_path"])
    nlp = spacy.load(config["nlp_path"])

    print("Selecting word groups...")
    docs, selections, analyzer = select_dataset(df, config, glove)

    print("Evaluating selections...")
    results = evaluate_selection(docs, selections, analyzer, nlp)
    print(results)

    print("Making word maps...")
    word_maps = make_word_maps(df, config, docs, selections, glove, nlp, analyzer)

    print("Augmenting dataset...")
    return augment_dataset(df, word_maps)
