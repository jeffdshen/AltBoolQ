import re
from collections import Counter, defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer


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
    def __init__(self, glove, alpha):
        self.glove = glove
        self.alpha = alpha

    def __call__(self, word_count_i, word_count_j):
        word_i, count_i = word_count_i
        word_j, count_j = word_count_j
        if word_i == word_j:
            return 1.0
        if word_i not in self.glove or word_j not in self.glove:
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


class Selector:
    def __init__(self, tfidf_cutoff, freq_cutoff, group_size):
        self.tfidf_cutoff = tfidf_cutoff
        self.freq_cutoff = freq_cutoff
        self.group_size = group_size

    def select(self, groups, tfidf_weights, term_freqs):
        total_freq = 0
        results = []
        seen = set()
        for score, word in tfidf_weights:
            if word in seen:
                continue

            if score < self.tfidf_cutoff:
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


def evaluate_selection(docs, selections, nlp):
    lengths = np.array([len(s) for s in selections], dtype=int)
    results = {
        "non_empty": np.mean(lengths > 0),
        "avg_length": np.mean(lengths),
        "max_length": np.max(lengths),
    }

    group_accuracy = []
    for doc, selection in zip(docs, selections):
        success, total = get_group_accuracy(doc, selection, nlp)
        group_accuracy.append(1.0 if total == 0 else success / total)

    results["group_accuracy"] = np.mean(group_accuracy)
    results["exact_group_accuracy"] = np.mean(np.array(group_accuracy) == 1.0)
    return results


def select_dataset(df, config, glove):
    task_format = "{}\n{}"
    docs = [
        task_format.format(row["passage"], row["question"]) for _, row in df.iterrows()
    ]
    _, analyzer, tfidf_weights = get_tfidf(docs)
    _, _, _ = get_doc_freqs(docs, analyzer)
    group_weight = GroupWeight(glove, alpha=config["alpha"])
    grouper = Grouper(group_weight, grouping_cutoff=config["grouping_cutoff"])
    selector = Selector(
        tfidf_cutoff=config["tfidf_cutoff"],
        freq_cutoff=config["freq_cutoff"],
        group_size=config["group_size"],
    )
    selections = []
    for i in tqdm(range(len(docs))):
        doc = docs[i]
        term_freqs, term_counts, _ = get_term_freqs(doc, analyzer)
        groups = grouper(term_counts)
        selection = selector.select(groups, tfidf_weights[i], term_freqs)
        selections.append(selection)

    return docs, selections


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


def make_mask_word_maps(config, docs, selections, nlp):
    if config["filter"] == None:
        filterer = none_filter
    elif config["filter"] == "group_accuracy":
        filterer = GroupAccuracyFilter(nlp)
    else:
        raise ValueError('config["filter"] is invalid')

    word_maps = []
    for index, (doc, selection) in enumerate(zip(docs, selections)):
        if not filterer(doc, selection):
            continue

        word_map = {}
        for group in selection:
            for word in group:
                word_map[word] = "redacted"

        word_maps.append((index, word_map))
    return word_maps


def make_word_maps(df, config, docs, selections, nlp):
    if config["resample"] == "mask":
        return make_mask_word_maps(config, docs, selections, nlp)
    else:
        raise ValueError('config["resample"] is invalid')


def run_augment(dfs, config, loader, spacy):
    df = dfs[config["df"]]

    print("Loading glove and nlp...")
    glove = loader.load(config["glove_path"])
    nlp = spacy.load(config["nlp_path"])

    print("Selecting word groups...")
    docs, selections = select_dataset(df, config, glove)

    print("Making word maps...")
    word_maps = make_word_maps(df, config, docs, selections, nlp)

    print("Augmenting dataset...")
    return augment_dataset(df, word_maps)
