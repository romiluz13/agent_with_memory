"""
REAL Integration Tests for Reciprocal Rank Fusion algorithm.
Tests with actual SearchResult objects - no mocks!
"""

from src.retrieval.rrf import DEFAULT_RRF_CONSTANT, reciprocal_rank_fusion, weighted_score_fusion
from src.retrieval.vector_search import SearchResult


class TestRRFAlgorithmReal:
    """Real tests for RRF algorithm with actual data."""

    def test_rrf_constant_is_60(self):
        """RRF constant should be 60 (standard from academic papers)."""
        assert DEFAULT_RRF_CONSTANT == 60

    def test_empty_input_returns_empty(self):
        """Empty input should return empty output."""
        result = reciprocal_rank_fusion({})
        assert result == []
        assert isinstance(result, list)

    def test_single_source_preserves_order(self, sample_search_results):
        """Single source should preserve ranking order."""
        vector_only = {"vector": sample_search_results["vector"]}
        result = reciprocal_rank_fusion(vector_only)

        assert len(result) == 3
        # First result should still be doc1 (highest score)
        assert result[0].id == "doc1"
        assert result[1].id == "doc2"
        assert result[2].id == "doc3"

    def test_fusion_boosts_overlapping_results(self, sample_search_results):
        """Results appearing in multiple lists should get boosted."""
        result = reciprocal_rank_fusion(sample_search_results)

        # doc1 appears in both: rank 0 in vector, rank 2 in text
        # doc2 appears in both: rank 1 in vector, rank 0 in text
        # Both should be ranked higher than single-source results

        result_ids = [r.id for r in result]

        # doc1 and doc2 should be in top positions
        assert "doc1" in result_ids[:3]
        assert "doc2" in result_ids[:3]

    def test_rrf_formula_correctness(self, sample_search_results):
        """Verify RRF formula: score = sum(1/(k+rank))."""
        result = reciprocal_rank_fusion(sample_search_results, k=60)

        # doc2 appears at rank 1 in vector, rank 0 in text
        # RRF score = 0.5 * (1/(60+1)) + 0.5 * (1/(60+0))
        # = 0.5 * (1/61) + 0.5 * (1/60)
        # = 0.5 * 0.01639 + 0.5 * 0.01667
        # ≈ 0.01653

        doc2_result = next(r for r in result if r.id == "doc2")
        expected_score = 0.5 * (1 / 61) + 0.5 * (1 / 60)
        assert abs(doc2_result.score - expected_score) < 0.0001

    def test_top_k_limits_results(self, sample_search_results):
        """top_k should limit the number of results."""
        result = reciprocal_rank_fusion(sample_search_results, top_k=2)
        assert len(result) == 2

    def test_top_k_with_large_value(self, sample_search_results):
        """top_k larger than results should return all results."""
        result = reciprocal_rank_fusion(sample_search_results, top_k=100)
        # We have 4 unique docs (doc1, doc2, doc3, doc4)
        assert len(result) == 4

    def test_weights_favor_vector(self, sample_search_results):
        """Heavy vector weight should favor vector rankings."""
        result_vector_heavy = reciprocal_rank_fusion(
            sample_search_results, weights={"vector": 0.9, "text": 0.1}
        )

        # doc1 is rank 0 in vector, should be boosted
        assert result_vector_heavy[0].id == "doc1"

    def test_weights_favor_text(self, sample_search_results):
        """Heavy text weight should favor text rankings."""
        result_text_heavy = reciprocal_rank_fusion(
            sample_search_results, weights={"vector": 0.1, "text": 0.9}
        )

        # doc2 is rank 0 in text, should be boosted
        assert result_text_heavy[0].id == "doc2"

    def test_metadata_includes_rrf_info(self, sample_search_results):
        """Result metadata should include RRF scoring details."""
        result = reciprocal_rank_fusion(sample_search_results)

        for r in result:
            assert "rrf_score" in r.metadata
            assert "source_scores" in r.metadata
            assert "source_ranks" in r.metadata
            assert "fusion_method" in r.metadata
            assert r.metadata["fusion_method"] == "rrf"

    def test_source_scores_preserved(self, sample_search_results):
        """Original source scores should be preserved in metadata."""
        result = reciprocal_rank_fusion(sample_search_results)

        # Find doc2 which appears in both
        doc2 = next(r for r in result if r.id == "doc2")

        assert "vector" in doc2.metadata["source_scores"]
        assert "text" in doc2.metadata["source_scores"]
        assert doc2.metadata["source_scores"]["vector"] == 0.85
        assert doc2.metadata["source_scores"]["text"] == 0.90

    def test_source_ranks_correct(self, sample_search_results):
        """Source ranks should be correctly recorded."""
        result = reciprocal_rank_fusion(sample_search_results)

        doc2 = next(r for r in result if r.id == "doc2")
        assert doc2.metadata["source_ranks"]["vector"] == 1  # 0-indexed
        assert doc2.metadata["source_ranks"]["text"] == 0

    def test_custom_k_value(self, sample_search_results):
        """Custom k value should affect ranking spread."""
        result_k10 = reciprocal_rank_fusion(sample_search_results, k=10)
        result_k100 = reciprocal_rank_fusion(sample_search_results, k=100)

        # Lower k means more weight to top ranks
        # Rankings might differ with different k values
        assert len(result_k10) == len(result_k100)

    def test_three_sources(self):
        """Test with three result sources."""
        results = {
            "vector": [
                SearchResult(id="a", content="A", metadata={}, score=0.9),
                SearchResult(id="b", content="B", metadata={}, score=0.8),
            ],
            "text": [
                SearchResult(id="b", content="B", metadata={}, score=0.85),
                SearchResult(id="c", content="C", metadata={}, score=0.75),
            ],
            "semantic": [
                SearchResult(id="a", content="A", metadata={}, score=0.95),
                SearchResult(id="c", content="C", metadata={}, score=0.80),
            ],
        }

        result = reciprocal_rank_fusion(results)

        # a appears in vector (rank 0) and semantic (rank 0)
        # b appears in vector (rank 1) and text (rank 0)
        # c appears in text (rank 1) and semantic (rank 1)
        # a and b should be top results

        top_2 = [r.id for r in result[:2]]
        assert "a" in top_2
        assert "b" in top_2


class TestWeightedScoreFusionReal:
    """Real tests for weighted score fusion."""

    def test_empty_input_returns_empty(self):
        """Empty input should return empty output."""
        result = weighted_score_fusion({})
        assert result == []

    def test_single_source_normalizes(self):
        """Single source should normalize scores to 0-1."""
        results = {
            "vector": [
                SearchResult(id="a", content="A", metadata={}, score=100),
                SearchResult(id="b", content="B", metadata={}, score=50),
                SearchResult(id="c", content="C", metadata={}, score=0),
            ]
        }

        result = weighted_score_fusion(results)

        # After normalization: a=1.0, b=0.5, c=0.0
        assert result[0].id == "a"
        assert result[0].score == 1.0

    def test_fusion_combines_normalized_scores(self):
        """Fusion should combine normalized scores with weights."""
        results = {
            "vector": [
                SearchResult(id="a", content="A", metadata={}, score=1.0),
                SearchResult(id="b", content="B", metadata={}, score=0.0),
            ],
            "text": [
                SearchResult(id="b", content="B", metadata={}, score=1.0),
                SearchResult(id="a", content="A", metadata={}, score=0.0),
            ],
        }

        result = weighted_score_fusion(results, weights={"vector": 0.5, "text": 0.5})

        # Both a and b should have score 0.5 (0.5*1 + 0.5*0)
        for r in result:
            assert abs(r.score - 0.5) < 0.01

    def test_metadata_includes_weighted_info(self):
        """Result metadata should include weighted fusion info."""
        results = {"vector": [SearchResult(id="a", content="A", metadata={}, score=0.9)]}

        result = weighted_score_fusion(results)

        assert "weighted_score" in result[0].metadata
        assert "fusion_method" in result[0].metadata
        assert result[0].metadata["fusion_method"] == "weighted"

    def test_top_k_limits(self):
        """top_k should limit results."""
        results = {
            "a": [
                SearchResult(id="1", content="", metadata={}, score=0.9),
                SearchResult(id="2", content="", metadata={}, score=0.8),
                SearchResult(id="3", content="", metadata={}, score=0.7),
            ]
        }

        result = weighted_score_fusion(results, top_k=2)
        assert len(result) == 2
