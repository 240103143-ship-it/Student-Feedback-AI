"""Lightweight multilingual feedback analyzer (Sprint 3).

Pipeline for a single comment:
    PII masking -> language detection -> sentence split -> tokenization
    -> lexicon sentiment (negation, intensifiers, contrast) per sentence
    -> topic detection per clause -> aspect sentiment -> keywords

Where each Sprint 3 story lives:
    3.1 Sentiment analysis        sentence_sentiment(), blended with the rating in analyze()
    3.2 Topic detection           _match_topics(), keywords from the topic_categories table
    3.3 Aspect-based sentiment    the per-clause loop in analyze()
    3.4 Common problem detection  routers/analytics.py, from the stored aspects

The analyzer is rule-based and deterministic: it needs no model downloads and
sends no text to an external service.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .lexicons import (
    CONTRAST, EXCESS, INTENSIFIERS, LEXICON, NEGATIONS_AFTER, NEGATIONS_BEFORE, STOPWORDS,
)
from .pii import mask_pii

KAZAKH_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*|\d+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
CLAUSE_RE = re.compile(
    r",|;|\s[-—]\s|\b(?:but|however|although|though|whereas|while)\b"
    r"|(?<!\w)(?:но|однако|зато|а|бірақ|алайда)(?!\w)",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(r"\[(?:NAME|EMAIL|PHONE|ID)\]")

# Scores run from -1 to 1; anything between the two thresholds is neutral.
POS_THRESHOLD = 0.15
NEG_THRESHOLD = -0.15

_SINGLE = {k: v for k, v in LEXICON.items() if " " not in k}
_PHRASES = {k: v for k, v in LEXICON.items() if " " in k}
_MAX_STEM = max(len(k) for k in _SINGLE)


@dataclass
class TopicMatcher:
    name: str
    stems: list[str]
    phrases: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    text: str
    language: str
    sentiment: str
    score: float
    confidence: float
    topics: list[str]
    aspects: dict[str, str]
    keywords: list[str]
    pii_masked: bool


def detect_language(text: str) -> str:
    """kk if the text has a letter Russian lacks (ә, қ, ү …), else ru if Cyrillic outnumbers Latin, else en."""
    if any(ch in KAZAKH_CHARS for ch in text):
        return "kk"
    cyr = len(CYRILLIC_RE.findall(text))
    lat = len(re.findall(r"[a-zA-Z]", text))
    return "ru" if cyr > lat else "en"


def tokenize(text: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in TOKEN_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_RE.split(text) if p and p.strip()]
    return parts or [text.strip()]


def _lex_score(token: str) -> float:
    if token in _SINGLE:
        return _SINGLE[token]
    # longest prefix match for stems >= 4 chars (handles RU/KK inflection)
    for n in range(min(len(token), _MAX_STEM), 3, -1):
        stem = token[:n]
        if stem in _SINGLE and len(stem) >= 4:
            return _SINGLE[stem]
    return 0.0


def _normalize(raw: float, alpha: float = 4.0) -> float:
    # Squashes any raw total into (-1, 1); the same formula VADER uses.
    return raw / math.sqrt(raw * raw + alpha)


def sentence_sentiment(sentence: str) -> tuple[float, int]:
    """Returns (normalized score in [-1, 1], number of sentiment-bearing words).

    Story 3.1 rules, applied word by word:
      * lexicon words carry a weight: "excellent" +2.8, "terrible" -2.6;
      * "not" / "не" reverses (and softens) opinion words among the next three words,
        up to the next comma; Kazakh "емес" reverses the word before it;
      * intensifiers ("very", "очень", "өте") strengthen the next opinion word;
      * "too" / "слишком" / "тым" makes what follows negative ("too easy");
      * after "but" / "но" / "бірақ" the earlier part counts half, the rest 1.5x.
    """
    # negation scope ends at punctuation: "never covered, very unfair" stays negative
    tokens: list[str] = []
    for chunk in re.split(r"[,;:()]", sentence):
        tokens += tokenize(chunk) + ["<sep>"]
    low = sentence.lower()
    total = 0.0
    hits = 0
    weight = 1.0
    negate_left = 0
    boost = 1.0
    scores: list[float] = []

    excess = False
    for tok in tokens:
        if tok == "<sep>":
            if excess:  # "too many deadlines": no opinion word followed, the excess itself is negative
                total -= 0.6 * weight
                hits += 1
            negate_left, excess = 0, False
            continue
        if tok in CONTRAST:
            # the clause after "but" usually carries the real opinion
            total *= 0.5
            weight = 1.5
            negate_left = 0
            continue
        if tok in NEGATIONS_BEFORE:
            negate_left = 3
            continue
        if tok in NEGATIONS_AFTER and scores:
            # Kazakh: flip the previous sentiment word
            last = scores[-1]
            total -= 2 * last * 0.9
            scores[-1] = -last
            continue
        if tok in INTENSIFIERS:
            boost = INTENSIFIERS[tok]
            continue
        if tok in EXCESS:
            excess = True
            continue

        s = _lex_score(tok)
        if s:
            if excess:
                s, excess = -abs(s) * 1.1, False
            s *= boost * weight
            if negate_left > 0:
                s *= -0.75
            total += s
            scores.append(s)
            hits += 1
            boost = 1.0
        if negate_left > 0:
            negate_left -= 1

    for phrase, s in _PHRASES.items():
        if phrase in low:
            total += s * weight
            hits += 1

    if "!" in sentence and total:
        total *= 1.1
    return _normalize(total), hits


def label_for(score: float) -> str:
    if score >= POS_THRESHOLD:
        return "positive"
    if score <= NEG_THRESHOLD:
        return "negative"
    return "neutral"


def _match_topics(tokens: list[str], low_text: str, matchers: list[TopicMatcher]) -> Counter:
    """Story 3.2: how many words of the text match each topic's keywords.

    Keywords are stems: "lectur" matches "lecture" and "lecturer". Stems shorter
    than 4 letters must match a whole word; multi-word keywords match as phrases.
    """
    found: Counter = Counter()
    for m in matchers:
        n = 0
        for tok in tokens:
            for stem in m.stems:
                if (len(stem) < 4 and tok == stem) or (len(stem) >= 4 and tok.startswith(stem)):
                    n += 1
                    break
        for ph in m.phrases:
            if ph in low_text:
                n += 1
        if n:
            found[m.name] += n
    return found


def build_matchers(categories: list[dict]) -> list[TopicMatcher]:
    """Turns topic_categories rows (editable by admins via /api/admin/topics) into matchers."""
    out = []
    for c in categories:
        kws = [k.lower().strip() for k in c.get("keywords", []) if k and k.strip()]
        out.append(TopicMatcher(
            name=c["name"],
            stems=[k for k in kws if " " not in k],
            phrases=[k for k in kws if " " in k],
        ))
    return out


def extract_keywords(tokens: list[str], limit: int = 8) -> list[str]:
    words = [t for t in tokens if len(t) >= 4 and t not in STOPWORDS and not t.isdigit()]
    return [w for w, _ in Counter(words).most_common(limit)]


def analyze(text: str, matchers: list[TopicMatcher], rating: int | None = None) -> Analysis:
    """Analyzes one comment: overall sentiment (3.1), topics (3.2) and sentiment per topic (3.3).

    `rating` is the response's 1-5 average rating, if it has one. It affects only
    the overall sentiment; the sentiment of each topic comes from the text alone.
    Example: "The instructor was excellent, but the workload was terrible"
    -> Teaching Quality positive, Workload negative.
    """
    # 1. Personal data is masked first, so every later step sees only anonymized text.
    masked, had_pii = mask_pii(text)
    # 2. Language detection: en / ru / kk.
    lang = detect_language(masked)
    # 3. Sentence splitting: every sentence gets its own sentiment score.
    sentences = split_sentences(masked)

    sent_scores: list[tuple[float, int]] = []
    topic_hits: Counter = Counter()
    aspect_acc: dict[str, list[float]] = {}

    for s in sentences:
        # 4. Sentiment score of the sentence (story 3.1).
        score, hits = sentence_sentiment(s)
        sent_scores.append((score, hits))
        clean = PLACEHOLDER_RE.sub(" ", s)
        # aspects are scored per clause so "good teacher, but heavy workload" splits correctly
        for clause in [c for c in CLAUSE_RE.split(clean) if c and c.strip()]:
            # 5. Topic matching (story 3.2).
            t = _match_topics(tokenize(clause), clause.lower(), matchers)
            if not t:
                continue
            # 6. Aspect sentiment (story 3.3): a topic takes the sentiment of its own clause,
            #    or of the whole sentence when the clause has no opinion words.
            c_score, c_hits = sentence_sentiment(clause)
            topic_hits.update(t)
            for name in t:
                aspect_acc.setdefault(name, []).append(c_score if c_hits else score)

    # Text score: the sentence scores averaged, weighted by how many opinion words each has.
    weighted = [sc * max(h, 1) for sc, h in sent_scores]
    total_hits = sum(max(h, 1) for _, h in sent_scores)
    text_score = sum(weighted) / total_hits if total_hits else 0.0
    any_hits = sum(h for _, h in sent_scores)

    # 7. Blend with the numeric rating when present (1..5 -> -1..1): 70% text and 30% rating,
    #    or 35% / 65% when the text has no opinion words (e.g. an empty comment).
    if rating:
        r = (max(1, min(5, rating)) - 3) / 2
        score = 0.7 * text_score + 0.3 * r if any_hits else 0.35 * text_score + 0.65 * r
    else:
        score = text_score
    score = max(-1.0, min(1.0, score))

    # 8. Confidence from 0.45 to 0.99: higher when the score is far from neutral
    #    and when more opinion words (up to six) support it.
    confidence = min(0.99, 0.45 + abs(score) * 0.5 + min(any_hits, 6) * 0.03)
    # A topic mentioned in several clauses gets the label of its average score.
    aspects = {k: label_for(sum(v) / len(v)) for k, v in aspect_acc.items()}
    # Most mentioned topic first.
    topics = [name for name, _ in topic_hits.most_common()]

    return Analysis(
        text=masked,
        language=lang,
        sentiment=label_for(score),
        score=round(score, 4),
        confidence=round(confidence, 3),
        topics=topics,
        aspects=aspects,
        keywords=extract_keywords(tokenize(PLACEHOLDER_RE.sub(" ", masked))),
        pii_masked=had_pii,
    )
