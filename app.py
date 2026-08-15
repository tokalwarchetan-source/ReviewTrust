import csv
import io
import math
import os
import random
import threading
import webbrowser

from dateutil import parser as dateparser
from flask import Flask, jsonify, request, send_from_directory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PORT = 5000

app = Flask(__name__, static_folder="frontend", static_url_path="")

MAX_ROWS = 6000  # soft cap so an O(n^2) graph pass on a huge upload doesn't hang the server

# --------------------------------------------------------------------------
# CSV ingestion
# --------------------------------------------------------------------------

COLUMN_ALIASES = {
    "reviewer": ["reviewer", "user", "username", "account", "customer", "author", "reviewer_id", "reviewerid",
                 "reviewername", "name"],
    "product": ["product", "item", "sku", "product_name", "listing", "product_id", "productid", "asin"],
    "rating": ["rating", "stars", "score", "review_rating", "overall", "rate"],
    "minutes_ago": ["minutes_ago", "mins_ago", "minsago", "minutes"],
    "timestamp": ["timestamp", "date", "datetime", "created_at", "review_date", "time", "posted_at",
                  "reviewtime", "unixreviewtime"],
    "label": ["label", "is_fake", "fake", "fraud", "suspicious", "is_suspicious", "ground_truth",
              "target", "class"],
}


def find_column(header, aliases):
    lower = [h.strip().lower() for h in header]
    for alias in aliases:
        if alias in lower:
            return lower.index(alias)
    return -1


def ingest_csv_text(text):
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if len(rows) < 2:
        return {"ok": False, "error": "That file looks empty — need a header row plus at least one data row."}

    header = rows[0]
    reviewer_idx = find_column(header, COLUMN_ALIASES["reviewer"])
    product_idx = find_column(header, COLUMN_ALIASES["product"])
    rating_idx = find_column(header, COLUMN_ALIASES["rating"])
    mins_idx = find_column(header, COLUMN_ALIASES["minutes_ago"])
    ts_idx = find_column(header, COLUMN_ALIASES["timestamp"])
    label_idx = find_column(header, COLUMN_ALIASES["label"])

    if reviewer_idx == -1 or product_idx == -1 or (mins_idx == -1 and ts_idx == -1):
        return {
            "ok": False,
            "error": f"Couldn't find the required columns. Need a reviewer column, a product column, "
                     f"and either \"minutes_ago\" or a \"timestamp\"/\"date\" column. "
                     f"Headers found: {', '.join(header)}",
        }

    data_rows = rows[1:]
    truncated = False
    if len(data_rows) > MAX_ROWS:
        data_rows = data_rows[:MAX_ROWS]
        truncated = True

    raw = []
    for r in data_rows:
        def cell(i):
            return r[i].strip() if i != -1 and i < len(r) else ""

        reviewer = cell(reviewer_idx)
        product = cell(product_idx)

        rating = 5
        if rating_idx != -1:
            try:
                rating = max(1, min(5, round(float(cell(rating_idx)))))
            except (ValueError, TypeError):
                rating = 5

        time_val = None
        if mins_idx != -1:
            try:
                time_val = ("mins", float(cell(mins_idx)))
            except (ValueError, TypeError):
                time_val = None
        elif ts_idx != -1:
            try:
                dt = dateparser.parse(cell(ts_idx))
                time_val = ("ts", dt.timestamp())
            except (ValueError, TypeError, OverflowError):
                time_val = None

        label = None
        if label_idx != -1:
            value = cell(label_idx).strip().lower()
            if value in {"1", "true", "yes", "fake", "fraud", "suspicious", "positive"}:
                label = 1
            elif value in {"0", "false", "no", "real", "genuine", "normal", "negative"}:
                label = 0

        if reviewer and product and time_val is not None:
            item = {
                "reviewer": reviewer,
                "product": product,
                "rating": rating,
                "time_val": time_val,
            }
            if label is not None:
                item["label"] = label
            raw.append(item)

    if not raw:
        return {"ok": False, "error": "No valid rows could be parsed — check that the reviewer, product, and time columns are filled in for at least one row."}

    out = []
    if raw[0]["time_val"][0] == "mins":
        for i, r in enumerate(raw):
            item = {
                "id": i + 1,
                "reviewer": r["reviewer"],
                "product": r["product"],
                "rating": r["rating"],
                "minsAgo": r["time_val"][1],
            }
            if "label" in r:
                item["label"] = r["label"]
            out.append(item)
    else:
        max_ts = max(r["time_val"][1] for r in raw)
        for i, r in enumerate(raw):
            mins_ago = max(0, round((max_ts - r["time_val"][1]) / 60))
            item = {
                "id": i + 1,
                "reviewer": r["reviewer"],
                "product": r["product"],
                "rating": r["rating"],
                "minsAgo": mins_ago,
            }
            if "label" in r:
                item["label"] = r["label"]
            out.append(item)

    return {
        "ok": True,
        "reviews": out,
        "truncated": truncated,
        "mapping": {
            "reviewer": header[reviewer_idx],
            "product": header[product_idx],
            "rating": header[rating_idx] if rating_idx != -1 else "(defaulted to 5)",
            "time": header[ts_idx] if ts_idx != -1 else header[mins_idx],
            "label": header[label_idx] if label_idx != -1 else "(not provided)",
        },
    }


CSV_TEMPLATE = """reviewer,product,rating,timestamp,label
alex_42,Wireless Earbuds Pro,5,2026-07-11T14:32:00,1
jamie_88,Wireless Earbuds Pro,5,2026-07-11T14:34:00,1
sam_19,Wireless Earbuds Pro,5,2026-07-11T14:37:00,1
priya_61,Standing Desk Converter,4,2026-07-10T09:12:00
noah_23,Insulated Water Bottle,5,2026-07-09T18:45:00
"""

# --------------------------------------------------------------------------
# Synthetic demo dataset
# --------------------------------------------------------------------------

PRODUCTS = [
    "Wireless Earbuds Pro", "Ceramic Chef Knife Set", "Standing Desk Converter",
    "Aroma Diffuser Mini", "Running Shoes V3", "Insulated Water Bottle",
    "Mechanical Keyboard 87", "Weighted Blanket 15lb",
]
FIRSTS = ["alex", "jamie", "sam", "priya", "wei", "maria", "noah", "yuki", "liam", "fatima", "ivan", "emre", "sofia", "ben", "tara"]


def make_handle(rng):
    suffix = chr(97 + rng.randint(0, 25)) if rng.random() < 0.3 else ""
    return f"{rng.choice(FIRSTS)}_{rng.randint(10, 99)}{suffix}"


def make_synthetic_sample(seed=1337):
    rng = random.Random(seed)
    reviews = []
    idc = 1

    ring_a_products = [PRODUCTS[0], PRODUCTS[6]]
    ring_a_reviewers = [make_handle(rng) for _ in range(5)]
    for h in ring_a_reviewers:
        for p in ring_a_products:
            reviews.append({"id": idc, "reviewer": h, "product": p, "rating": 5, "minsAgo": rng.randint(5, 40)})
            idc += 1

    ring_b_product = PRODUCTS[3]
    ring_b_reviewers = [make_handle(rng) for _ in range(4)]
    for h in ring_b_reviewers:
        reviews.append({"id": idc, "reviewer": h, "product": ring_b_product, "rating": rng.randint(4, 5), "minsAgo": rng.randint(180, 225)})
        idc += 1

    for _ in range(32):
        reviews.append({
            "id": idc, "reviewer": make_handle(rng), "product": rng.choice(PRODUCTS),
            "rating": rng.randint(3, 5), "minsAgo": rng.randint(2, 1400),
        })
        idc += 1

    return reviews

BURST_WINDOW_MIN = 15

# Small ML layer. The training examples are generated by the demo dataset
# so the project can run locally without downloading a pretrained model.
ML_FEATURES = [
    "rating",
    "reviewer_review_count",
    "product_review_count",
    "reviews_in_burst",
    "reviewer_product_count",
    "shared_product_count",
]


def build_ml_features(reviews, network_lookup=None):
    reviewer_counts = {}
    product_counts = {}
    reviewer_products = {}

    for r in reviews:
        reviewer = r["reviewer"]
        product = r["product"]
        reviewer_counts[reviewer] = reviewer_counts.get(reviewer, 0) + 1
        product_counts[product] = product_counts.get(product, 0) + 1
        reviewer_products.setdefault(reviewer, set()).add(product)

    rows = []
    for r in reviews:
        burst = sum(
            1 for other in reviews
            if other["product"] == r["product"]
            and abs(other["minsAgo"] - r["minsAgo"]) <= BURST_WINDOW_MIN
        )

        shared_products = 0
        my_products = reviewer_products[r["reviewer"]]
        for other, products in reviewer_products.items():
            if other != r["reviewer"] and my_products.intersection(products):
                if r["product"] in products:
                    shared_products += 1

        network = {}
        if network_lookup:
            network = network_lookup.get(r["reviewer"], {})

        rows.append([
            r["rating"],
            reviewer_counts[r["reviewer"]],
            product_counts[r["product"]],
            burst,
            len(my_products),
            shared_products,
            network.get("degree", 0),
            network.get("density", 0),
            network.get("centrality", 0),
            network.get("maxJaccard", 0),
            network.get("avgJaccard", 0),
            network.get("clusterSize", 1),
        ])

    return rows



def build_training_network_features(reviews):
    """Build simple reviewer-network features for ML training data."""
    reviewer_products = {}
    for r in reviews:
        reviewer_products.setdefault(r["reviewer"], set()).add(r["product"])

    handles = list(reviewer_products)
    lookup = {}

    for h in handles:
        connected = []
        jaccards = []
        weighted_degree = 0

        for other in handles:
            if other == h:
                continue

            shared = len(reviewer_products[h] & reviewer_products[other])
            if shared:
                connected.append(other)
                weighted_degree += shared

                union_size = len(
                    reviewer_products[h] | reviewer_products[other]
                )
                if union_size:
                    jaccards.append(shared / union_size)

        degree = len(connected)
        density = degree / max(len(handles) - 1, 1)
        max_jaccard = max(jaccards) if jaccards else 0
        avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0

        # Approximate connected component size using the same shared-product rule.
        component = {h}
        changed = True
        while changed:
            changed = False
            for other in handles:
                if other in component:
                    continue
                if any(
                    len(reviewer_products[member] & reviewer_products[other]) >= 1
                    for member in component
                ):
                    component.add(other)
                    changed = True

        lookup[h] = {
            "degree": degree,
            "weightedDegree": weighted_degree,
            "density": density,
            "centrality": density * 100,
            "maxJaccard": max_jaccard,
            "avgJaccard": avg_jaccard,
            "clusterSize": len(component),
        }

    return lookup

def train_demo_model():
    features = []
    labels = []

    for seed in range(20):
        sample = make_synthetic_sample(seed + 100)
        ring_reviewers = set()

        for r in sample:
            if r["id"] <= 10:
                ring_reviewers.add(r["reviewer"])
            elif 11 <= r["id"] <= 14:
                ring_reviewers.add(r["reviewer"])

        sample_network = build_training_network_features(sample)
        sample_features = build_ml_features(sample, sample_network)

        for row, review in zip(sample_features, sample):
            features.append(row)
            labels.append(1 if review["reviewer"] in ring_reviewers else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_test, predictions, zero_division=0), 4
        ),
        "confusion_matrix": matrix.tolist(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    return model, metrics



def train_labeled_model(reviews):
    """Train and evaluate the Random Forest when the uploaded CSV has labels."""
    labeled = [r for r in reviews if "label" in r]
    if len(labeled) < 20:
        return None, {
            "source": "demo",
            "message": "At least 20 labeled rows are recommended before retraining.",
        }

    labels = [r["label"] for r in labeled]
    if len(set(labels)) < 2:
        return None, {
            "source": "demo",
            "message": "Labeled data must contain both normal (0) and suspicious (1) examples.",
        }

    network_lookup = build_training_network_features(labeled)
    X = build_ml_features(labeled, network_lookup)
    y = labels

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        return None, {
            "source": "demo",
            "message": "The labeled dataset is too small or imbalanced for a stratified test split.",
        }

    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    metrics = {
        "source": "uploaded_labeled_data",
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1": round(f1_score(y_test, predictions, zero_division=0), 4),
        "confusion_matrix": matrix.tolist(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "labeled_rows": len(labeled),
        "features": ML_FEATURES,
    }

    return model, metrics


ML_MODEL, ML_METRICS = train_demo_model()

# --------------------------------------------------------------------------
# Scoring engine — temporal + network + ML
# --------------------------------------------------------------------------

BURST_WINDOW_MIN = 15


def poisson_tail_probability(observed, expected):
    """Probability of seeing at least `observed` events when the expected
    number in the same window is `expected`.

    This is a small local implementation so the project does not need scipy.
    """
    if observed <= 0:
        return 1.0
    expected = max(expected, 1e-9)

    # P(X >= k) = 1 - P(X <= k-1)
    term = math.exp(-expected)
    cdf = term
    for i in range(1, observed):
        term *= expected / i
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))



def change_point_stats(review, product_reviews):
    """Compare the current hourly activity with earlier hourly activity."""
    items = product_reviews.get(review["product"], [])
    if len(items) < 4:
        return 0, 0, 0

    buckets = {}
    for item in items:
        bucket = int(item["minsAgo"] // 60)
        data = buckets.setdefault(bucket, {"count": 0, "ratings": []})
        data["count"] += 1
        data["ratings"].append(item["rating"])

    current_bucket = int(review["minsAgo"] // 60)
    current = buckets.get(current_bucket)
    if not current:
        return 0, 0, 0

    earlier = [data for bucket, data in buckets.items() if bucket > current_bucket]
    if not earlier:
        return 0, 0, 0

    baseline_volume = sum(x["count"] for x in earlier) / len(earlier)
    old_ratings = [rating for x in earlier for rating in x["ratings"]]
    baseline_rating = sum(old_ratings) / len(old_ratings) if old_ratings else 0

    current_rating = sum(current["ratings"]) / len(current["ratings"])
    volume_ratio = current["count"] / max(baseline_volume, 1)

    volume_score = min(100, max(0, (volume_ratio - 1) * 55))
    rating_score = min(100, abs(current_rating - baseline_rating) * 35)
    change_score = round(0.7 * volume_score + 0.3 * rating_score)

    return round(volume_score), round(rating_score), max(0, min(100, change_score))


def score_dataset(raw_reviews):
    reviews = [dict(r) for r in raw_reviews]

    # Product-level activity is used as the baseline. A product with only a
    # few reviews should not be treated the same as a product with hundreds.
    product_times = {}
    product_reviews = {}
    for r in reviews:
        product_times.setdefault(r["product"], []).append(r["minsAgo"])
        product_reviews.setdefault(r["product"], []).append(r)

    def burst_stats(review):
        times = product_times[review["product"]]
        observed = sum(
            1 for t in times
            if abs(t - review["minsAgo"]) <= BURST_WINDOW_MIN
        )

        if len(times) <= 1:
            expected = 1.0
        else:
            span = max(times) - min(times)
            # Include the burst window in the observation period so very small
            # datasets do not produce an unrealistically tiny expected rate.
            exposure = max(span, BURST_WINDOW_MIN)
            rate_per_min = len(times) / exposure
            expected = max(1.0, rate_per_min * BURST_WINDOW_MIN)

        tail = poisson_tail_probability(observed, expected)

        # Convert a small tail probability into a 0-100 suspicion score.
        # p=0.05 gives about 65; p=0.01 gives about 100.
        if tail >= 0.05:
            score = 50 * (1 - tail)
        else:
            score = min(100, 65 + (-math.log10(max(tail, 1e-12)) - 1) * 35)

        return observed, expected, round(max(0, min(100, score)))

    def temporal_score_of(review):
        burst_score = burst_stats(review)[2]
        _, _, change_score = change_point_stats(review, product_reviews)
        return round(0.7 * burst_score + 0.3 * change_score)

    reviewers_by_handle = {}
    for r in reviews:
        reviewers_by_handle.setdefault(r["reviewer"], set()).add(r["product"])
    handles = list(reviewers_by_handle.keys())

    shared_weight = {h: {} for h in handles}
    for i in range(len(handles)):
        for j in range(i + 1, len(handles)):
            a, b = handles[i], handles[j]
            shared = len(reviewers_by_handle[a] & reviewers_by_handle[b])
            if shared > 0:
                shared_weight[a][b] = shared
                shared_weight[b][a] = shared

    def min_gap_on_shared_product(a, b):
        min_gap = math.inf
        a_reviews = [r for r in reviews if r["reviewer"] == a]
        for ra in a_reviews:
            for rb in reviews:
                if rb["reviewer"] == b and rb["product"] == ra["product"]:
                    min_gap = min(min_gap, abs(ra["minsAgo"] - rb["minsAgo"]))
        return min_gap

    parent = {h: h for h in handles}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(handles)):
        for j in range(i + 1, len(handles)):
            a, b = handles[i], handles[j]
            w = shared_weight[a].get(b, 0)
            if w >= 2:
                union(a, b)
            elif w == 1 and min_gap_on_shared_product(a, b) <= 45:
                union(a, b)

    components = {}
    for h in handles:
        components.setdefault(find(h), []).append(h)
    ring_groups_raw = [g for g in components.values() if len(g) >= 3]

    def max_shared_for(h):
        vals = [shared_weight[h].get(o, 0) for o in handles if o != h]
        return max(vals) if vals else 0

    def component_size_for(h):
        return len(components.get(find(h), [h]))

    def network_metrics(h):
        neighbors = shared_weight.get(h, {})
        connected = [o for o, shared in neighbors.items() if shared > 0]

        degree = len(connected)
        weighted_degree = sum(neighbors.values())

        my_products = reviewers_by_handle[h]
        jaccards = []
        for other in connected:
            other_products = reviewers_by_handle[other]
            union_size = len(my_products | other_products)
            if union_size:
                jaccards.append(
                    len(my_products & other_products) / union_size
                )

        max_jaccard = max(jaccards) if jaccards else 0
        avg_jaccard = (
            sum(jaccards) / len(jaccards)
            if jaccards else 0
        )

        density = degree / max(len(handles) - 1, 1)
        centrality = density * 100
        cluster_size = component_size_for(h)
        max_overlap = max_shared_for(h)

        # Combine several graph signals. Shared products and cluster size
        # remain the strongest signals because they are directly observable.
        cluster_score = min(100, max(0, (cluster_size - 1) * 18))
        overlap_score = min(100, max_overlap * 18)
        similarity_score = min(100, max_jaccard * 100)
        connection_score = min(100, centrality)

        score = (
            0.35 * cluster_score
            + 0.30 * overlap_score
            + 0.20 * similarity_score
            + 0.15 * connection_score
        )

        return {
            "degree": degree,
            "weightedDegree": weighted_degree,
            "density": round(density, 4),
            "centrality": round(centrality),
            "maxJaccard": round(max_jaccard, 4),
            "avgJaccard": round(avg_jaccard, 4),
            "clusterSize": cluster_size,
            "maxOverlap": max_overlap,
            "score": round(max(0, min(100, score))),
        }

    def network_score_of(h):
        return network_metrics(h)["score"]

    network_lookup = {
        h: network_metrics(h)
        for h in handles
    }

    ml_rows = build_ml_features(reviews, network_lookup)
    ml_probs = ML_MODEL.predict_proba(ml_rows)[:, 1]

    for r, ml_probability in zip(reviews, ml_probs):
        t = temporal_score_of(r)
        observed, expected, burst_score = burst_stats(r)
        volume_change, rating_change, change_score = change_point_stats(r, product_reviews)
        network = network_metrics(r["reviewer"])
        n = network["score"]
        rule_score = round(0.5 * t + 0.5 * n)
        ml_score = round(float(ml_probability) * 100)

        # Upgrade 8: transparent final-risk calculation.
        temporal_component = round(t * 0.30)
        network_component = round(n * 0.25)
        ml_component = round(ml_score * 0.30)
        change_component = round(change_score * 0.15)

        final_score = max(
            0,
            min(
                100,
                temporal_component
                + network_component
                + ml_component
                + change_component,
            ),
        )
        comp_size = network["clusterSize"]
        m = network["maxOverlap"]

        reasons = []
        if burst_score >= 55:
            reasons.append("Unusual posting burst")
        if change_score >= 55:
            reasons.append("Sudden activity change")
        if rating_change >= 55:
            reasons.append("Sudden rating change")
        if comp_size >= 3:
            reasons.append(f"Network cluster ({comp_size} accounts)")
        if m >= 2:
            reasons.append(f"{m} shared products")
        if network["maxJaccard"] >= 0.5:
            reasons.append("High reviewer similarity")
        if network["degree"] >= 3:
            reasons.append(f"Connected to {network['degree']} reviewers")
        if ml_score >= 70:
            reasons.append("ML model sees a high-risk pattern")

        reviewer_review_count = sum(1 for x in reviews if x["reviewer"] == r["reviewer"])
        product_review_count = sum(1 for x in reviews if x["product"] == r["product"])

        explanation = []
        if burst_score >= 55:
            explanation.append(
                f"{observed} reviews appeared in a {BURST_WINDOW_MIN}-minute window; "
                f"the estimated Poisson baseline was {expected:.1f}"
            )
        if change_score >= 55:
            explanation.append(
                f"Review activity changed sharply compared with earlier hourly activity "
                f"(change score {change_score}/100)"
            )
        if rating_change >= 55:
            explanation.append(
                f"The average rating changed sharply compared with the earlier baseline "
                f"(rating-change score {rating_change}/100)"
            )
        if comp_size >= 3:
            explanation.append(f"Reviewer belongs to a {comp_size}-account network cluster")
        if m >= 2:
            explanation.append(f"Reviewer shares product activity with {m} other account(s)")
        if network["maxJaccard"] >= 0.5:
            explanation.append(
                f"Highest reviewer-product similarity is "
                f"{network['maxJaccard'] * 100:.0f}%"
            )
        if network["degree"] >= 3:
            explanation.append(
                f"Reviewer is connected to {network['degree']} other reviewers "
                f"in the product network"
            )
        if r["rating"] == 5 and product_review_count >= 4:
            explanation.append("This is a 5-star review in a heavily reviewed product group")
        if ml_score >= 70:
            explanation.append(f"The ML model assigns {ml_score}% risk based on the review features")
        if not explanation:
            explanation.append("No strong individual signal was found")

        r["temporal"] = t
        r["burstScore"] = burst_score
        r["burstReviews"] = observed
        r["expectedBurstReviews"] = round(expected, 2)
        r["burstPValue"] = round(poisson_tail_probability(observed, expected), 6)
        r["volumeChangeScore"] = volume_change
        r["ratingChangeScore"] = rating_change
        r["changePointScore"] = change_score
        r["network"] = n
        r["networkSignals"] = {
            "degree": network["degree"],
            "weightedDegree": network["weightedDegree"],
            "density": network["density"],
            "centrality": network["centrality"],
            "maxJaccard": network["maxJaccard"],
            "avgJaccard": network["avgJaccard"],
            "clusterSize": network["clusterSize"],
            "maxOverlap": network["maxOverlap"],
        }
        r["ruleScore"] = rule_score
        r["mlScore"] = ml_score
        r["mlProbability"] = round(float(ml_probability), 3)
        r["score"] = final_score
        r["riskBreakdown"] = {
            "temporal": {"score": t, "weight": 0.30, "contribution": temporal_component},
            "network": {"score": n, "weight": 0.25, "contribution": network_component},
            "ml": {"score": ml_score, "weight": 0.30, "contribution": ml_component},
            "changePoint": {"score": change_score, "weight": 0.15, "contribution": change_component},
            "formula": "Temporal×30% + Network×25% + ML×30% + Change Point×15%",
        }
        r["reasons"] = reasons
        r["explanation"] = explanation
        r["signals"] = {
            "burstReviews": observed,
            "expectedBurstReviews": round(expected, 2),
            "burstPValue": round(poisson_tail_probability(observed, expected), 6),
            "volumeChangeScore": volume_change,
            "ratingChangeScore": rating_change,
            "changePointScore": change_score,
            "reviewerReviews": reviewer_review_count,
            "productReviews": product_review_count,
            "clusterSize": comp_size,
            "sharedAccounts": m,
            "networkDegree": network["degree"],
            "networkDensity": network["density"],
            "networkCentrality": network["centrality"],
            "maxJaccard": network["maxJaccard"],
            "avgJaccard": network["avgJaccard"],
            "weightedDegree": network["weightedDegree"],
            "riskBreakdown": {
                "temporal": {"score": t, "weight": 0.30, "contribution": temporal_component},
                "network": {"score": n, "weight": 0.25, "contribution": network_component},
                "ml": {"score": ml_score, "weight": 0.30, "contribution": ml_component},
                "changePoint": {"score": change_score, "weight": 0.15, "contribution": change_component},
            },
        }
        r["ring"] = find(r["reviewer"]) if comp_size >= 3 else None

    ring_groups = []
    for idx, group in enumerate(ring_groups_raw):
        group_reviews = [r for r in reviews if r["reviewer"] in group]
        avg_score = round(sum(r["score"] for r in group_reviews) / len(group_reviews))
        products_set = {r["product"] for r in group_reviews}
        max_w = max(max_shared_for(h) for h in group)
        ring_groups.append({
            "ring_id": find(group[0]),
            "label": f"Ring {chr(65 + (idx % 26))}{400 + idx * 17}",
            "reviewers": group,
            "avg_score": avg_score,
            "product_count": len(products_set),
            "max_overlap": max_w,
        })

    seen = set()
    products = []
    for r in reviews:
        if r["product"] not in seen:
            seen.add(r["product"])
            products.append(r["product"])

    return {
        "reviews": reviews,
        "ring_groups": ring_groups,
        "products": products,
        "ml": {
            "enabled": True,
            "model": "Random Forest",
            "features": ML_FEATURES,
            "network_features_used": True,
            "evaluation": ML_METRICS,
            "rule_weight": 0.55,
            "ml_weight": 0.30,
            "risk_weights": {
                "temporal": 0.30,
                "network": 0.25,
                "ml": 0.30,
                "change_point": 0.15,
            },
            "risk_formula": "Temporal×30% + Network×25% + ML×30% + Change Point×15%",
            "temporal_method": "Poisson burst detection + hourly change-point analysis",
            "network_method": "Reviewer-product overlap + Jaccard similarity + graph connectivity",
        },
    }

# --------------------------------------------------------------------------

# Routes
# --------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


@app.route("/api/sample")
def get_sample():
    seed = request.args.get("seed", 1337, type=int)
    raw = make_synthetic_sample(seed)
    model = score_dataset(raw)

    current_source = ML_METRICS.get("source", "demo")
    model["training_source"] = current_source
    model["model_status"] = {
        "mode": "trained" if current_source == "uploaded_labeled_data" else "demo",
        "label": (
            "Random Forest — trained on uploaded labeled data"
            if current_source == "uploaded_labeled_data"
            else "Random Forest — demonstration model"
        ),
        "source": (
            "Uploaded labeled dataset"
            if current_source == "uploaded_labeled_data"
            else "Synthetic demonstration data"
        ),
        "is_demo": current_source != "uploaded_labeled_data",
    }

    return jsonify({"ok": True, "truncated": False, "mapping": None, **model})


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"detail": "Please choose a CSV file to upload."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"detail": "Please upload a .csv file."}), 400

    raw_bytes = file.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    result = ingest_csv_text(text)
    if not result["ok"]:
        return jsonify({"detail": result["error"]}), 400

    global ML_MODEL, ML_METRICS

    model = score_dataset(result["reviews"])

    labeled_count = sum(1 for r in result["reviews"] if "label" in r)
    if labeled_count >= 20:
        trained_model, labeled_metrics = train_labeled_model(result["reviews"])
        if trained_model is not None:
            ML_MODEL = trained_model
            ML_METRICS = labeled_metrics
            model = score_dataset(result["reviews"])
            model["ml"]["evaluation"] = ML_METRICS

    model["labeled_rows"] = labeled_count

    current_source = ML_METRICS.get("source", "demo")
    if current_source == "uploaded_labeled_data":
        model_status = {
            "mode": "trained",
            "label": "Random Forest — trained on uploaded labeled data",
            "source": "Uploaded labeled dataset",
            "is_demo": False,
        }
    else:
        model_status = {
            "mode": "demo",
            "label": "Random Forest — demonstration model",
            "source": "Synthetic demonstration data",
            "is_demo": True,
        }

    model["training_source"] = current_source
    model["model_status"] = model_status

    return jsonify({
        "ok": True,
        "truncated": result["truncated"],
        "mapping": result["mapping"],
        **model,
    })


@app.route("/api/template")
def get_template():
    return CSV_TEMPLATE, 200, {"Content-Type": "text/csv"}


# --------------------------------------------------------------------------
# Entry point — running `python app.py` starts the server AND opens the
# dashboard in your default browser automatically.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}/"

    def open_browser():
        webbrowser.open(url)

    # Flask's debug reloader re-executes this file in a subprocess; only
    # open the browser / print the banner once, from the real running process.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print(f"ReviewTrust running at {url}")
        threading.Timer(1.0, open_browser).start()

    app.run(debug=True, port=PORT)