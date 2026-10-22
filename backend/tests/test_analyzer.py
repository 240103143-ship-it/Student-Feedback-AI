import unittest

from app.nlp.analyzer import analyze, build_matchers
from app.nlp.lexicons import DEFAULT_TOPICS


class Sprint3AnalyzerTests(unittest.TestCase):
    """Unit tests for app/nlp/analyzer.py: one per Sprint 3 story, plus PII masking."""

    @classmethod
    def setUpClass(cls):
        cls.matchers = build_matchers(DEFAULT_TOPICS)

    def test_sentiment_analysis_uses_text_and_rating(self):
        """Story 3.1: positive text rated 5/5 is positive, with score and confidence above 0.5."""
        result = analyze("The instructor was excellent and very helpful.", self.matchers, rating=5)
        self.assertEqual(result.sentiment, "positive")
        self.assertGreater(result.score, 0.5)
        self.assertGreater(result.confidence, 0.5)

    def test_topic_detection_is_multilingual(self):
        """Story 3.2: topics are detected in English, Russian and Kazakh comments."""
        english = analyze("The course workload and deadlines were difficult.", self.matchers)
        russian = analyze("Преподаватель понятно объяснял материал.", self.matchers)
        kazakh = analyze("Жатақхана бөлмесі өте суық.", self.matchers)
        self.assertIn("Workload", english.topics)
        self.assertIn("Teaching Quality", russian.topics)
        self.assertIn("Accommodation", kazakh.topics)

    def test_aspect_sentiment_splits_contrasting_clauses(self):
        """Story 3.3: "excellent instructor, but terrible workload" keeps both opinions."""
        result = analyze(
            "The instructor was excellent, but the workload was terrible and deadlines were stressful.",
            self.matchers,
        )
        self.assertEqual(result.aspects["Teaching Quality"], "positive")
        self.assertEqual(result.aspects["Workload"], "negative")

    def test_common_problem_inputs_are_structured(self):
        """Story 3.4: three negative Workload comments, the minimum for a common problem."""
        results = [
            analyze("The workload is terrible and deadlines are stressful.", self.matchers),
            analyze("Too much homework and an impossible deadline.", self.matchers),
            analyze("Assignments are overwhelming; the workload is bad.", self.matchers),
        ]
        negative_workload = sum(r.aspects.get("Workload") == "negative" for r in results)
        self.assertEqual(negative_workload, 3)

    def test_pii_is_masked_before_analysis_output(self):
        """Story 1.4: the e-mail address never appears in the analyzer's output."""
        result = analyze("Email me at student@example.com. The lectures were great.", self.matchers)
        self.assertNotIn("student@example.com", result.text)
        self.assertIn("[EMAIL]", result.text)
        self.assertTrue(result.pii_masked)


if __name__ == "__main__":
    unittest.main()
