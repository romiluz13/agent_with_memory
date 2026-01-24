"""Tests for MongoDB filter builders."""

from datetime import datetime

from src.retrieval.config import (
    AtlasSearchFilterConfig,
    VectorSearchFilterConfig,
)
from src.retrieval.filters import (
    build_atlas_search_filters,
    build_vector_search_filters,
    simplify_filters_for_basic_search,
    wrap_in_compound_filter,
)


class TestVectorSearchFilters:
    """Tests for MQL filter builder."""

    def test_empty_config_returns_empty_filters(self):
        """Empty config should return empty dict."""
        result = build_vector_search_filters()
        assert result == {}

    def test_agent_id_filter(self):
        """Agent ID should use $eq operator."""
        result = build_vector_search_filters(agent_id="agent_123")
        assert result == {"agent_id": {"$eq": "agent_123"}}

    def test_memory_types_filter(self):
        """Memory types should use $in operator."""
        result = build_vector_search_filters(memory_types=["episodic", "semantic"])
        assert result == {"memory_type": {"$in": ["episodic", "semantic"]}}

    def test_date_range_filter(self):
        """Date range should use $gte and $lte."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        result = build_vector_search_filters(start_date=start, end_date=end)
        assert result == {"timestamp": {"$gte": start, "$lte": end}}

    def test_importance_filter(self):
        """Importance range should use comparison operators."""
        result = build_vector_search_filters(importance_min=0.5, importance_max=0.9)
        assert result == {"importance": {"$gte": 0.5, "$lte": 0.9}}

    def test_combined_filters(self):
        """Multiple filters should combine correctly."""
        config = VectorSearchFilterConfig(
            agent_id="agent_123",
            memory_types=["episodic"],
            importance_min=0.7,
        )
        result = build_vector_search_filters(config)

        assert "agent_id" in result
        assert "memory_type" in result
        assert "importance" in result

    def test_thread_id_filter(self):
        """Thread ID should use $eq operator."""
        result = build_vector_search_filters(thread_id="thread_456")
        assert result == {"thread_id": {"$eq": "thread_456"}}

    def test_tags_filter(self):
        """Tags should use $in operator."""
        result = build_vector_search_filters(tags=["important", "work"])
        assert result == {"metadata.tags": {"$in": ["important", "work"]}}

    def test_user_id_filter(self):
        """User ID should use $eq operator."""
        result = build_vector_search_filters(user_id="user_789")
        assert result == {"user_id": {"$eq": "user_789"}}

    def test_generic_equality_filters(self):
        """Generic equality filters should be applied."""
        config = VectorSearchFilterConfig(equality_filters={"custom_field": "custom_value"})
        result = build_vector_search_filters(config)
        assert result == {"custom_field": {"$eq": "custom_value"}}

    def test_generic_in_filters(self):
        """Generic $in filters should be applied."""
        config = VectorSearchFilterConfig(in_filters={"status": ["active", "pending"]})
        result = build_vector_search_filters(config)
        assert result == {"status": {"$in": ["active", "pending"]}}

    def test_generic_comparison_filters(self):
        """Generic comparison filters should be applied."""
        config = VectorSearchFilterConfig(comparison_filters={"score": {"$gte": 10, "$lt": 100}})
        result = build_vector_search_filters(config)
        assert result == {"score": {"$gte": 10, "$lt": 100}}


class TestSimplifyFilters:
    """Tests for filter simplification."""

    def test_simplify_eq_filter(self):
        """Should simplify $eq to direct value."""
        filters = {"agent_id": {"$eq": "agent_123"}}
        result = simplify_filters_for_basic_search(filters)
        assert result == {"agent_id": "agent_123"}

    def test_preserve_complex_filter(self):
        """Should preserve complex filters."""
        filters = {"importance": {"$gte": 0.5, "$lte": 0.9}}
        result = simplify_filters_for_basic_search(filters)
        assert result == filters

    def test_simplify_multiple_eq_filters(self):
        """Should simplify multiple $eq filters."""
        filters = {"agent_id": {"$eq": "agent_123"}, "user_id": {"$eq": "user_456"}}
        result = simplify_filters_for_basic_search(filters)
        assert result == {"agent_id": "agent_123", "user_id": "user_456"}

    def test_preserve_direct_values(self):
        """Should preserve already simplified values."""
        filters = {"agent_id": "agent_123"}
        result = simplify_filters_for_basic_search(filters)
        assert result == filters


class TestAtlasSearchFilters:
    """Tests for Atlas Search filter builder."""

    def test_empty_config_returns_empty_list(self):
        """Empty config should return empty list."""
        result = build_atlas_search_filters()
        assert result == []

    def test_agent_id_filter(self):
        """Agent ID should use equals operator."""
        result = build_atlas_search_filters(agent_id="agent_123")
        assert result == [{"equals": {"path": "agent_id", "value": "agent_123"}}]

    def test_user_id_filter(self):
        """User ID should use equals operator."""
        result = build_atlas_search_filters(user_id="user_456")
        assert result == [{"equals": {"path": "user_id", "value": "user_456"}}]

    def test_date_range_filter(self):
        """Date range should use range operator."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        result = build_atlas_search_filters(start_date=start, end_date=end)
        assert len(result) == 1
        assert "range" in result[0]
        assert result[0]["range"]["path"] == "timestamp"
        assert result[0]["range"]["gte"] == start
        assert result[0]["range"]["lte"] == end

    def test_date_range_start_only(self):
        """Date range with only start should work."""
        start = datetime(2024, 1, 1)
        result = build_atlas_search_filters(start_date=start)
        assert len(result) == 1
        assert result[0]["range"]["path"] == "timestamp"
        assert result[0]["range"]["gte"] == start
        assert "lte" not in result[0]["range"]

    def test_custom_timestamp_field(self):
        """Should support custom timestamp field."""
        start = datetime(2024, 1, 1)
        result = build_atlas_search_filters(start_date=start, timestamp_field="created_at")
        assert result[0]["range"]["path"] == "created_at"

    def test_combined_filters(self):
        """Multiple filters should combine correctly."""
        config = AtlasSearchFilterConfig(agent_id="agent_123", user_id="user_456")
        result = build_atlas_search_filters(config)
        assert len(result) == 2
        # Check both filters are present
        agent_filter = next(
            (f for f in result if "equals" in f and f["equals"]["path"] == "agent_id"), None
        )
        user_filter = next(
            (f for f in result if "equals" in f and f["equals"]["path"] == "user_id"), None
        )
        assert agent_filter is not None
        assert user_filter is not None

    def test_generic_equality_filters(self):
        """Generic equality filters should be applied."""
        config = AtlasSearchFilterConfig(equality_filters={"custom_field": "custom_value"})
        result = build_atlas_search_filters(config)
        assert result == [{"equals": {"path": "custom_field", "value": "custom_value"}}]

    def test_generic_in_filters(self):
        """Generic $in filters should use compound with should."""
        config = AtlasSearchFilterConfig(in_filters={"status": ["active", "pending"]})
        result = build_atlas_search_filters(config)
        assert len(result) == 1
        assert "compound" in result[0]
        assert "should" in result[0]["compound"]
        assert result[0]["compound"]["minimumShouldMatch"] == 1
        # Check should clauses
        should_clauses = result[0]["compound"]["should"]
        assert len(should_clauses) == 2
        values = [clause["equals"]["value"] for clause in should_clauses]
        assert "active" in values
        assert "pending" in values

    def test_generic_range_filters(self):
        """Generic range filters should be applied."""
        config = AtlasSearchFilterConfig(range_filters={"score": {"gte": 10, "lt": 100}})
        result = build_atlas_search_filters(config)
        assert len(result) == 1
        assert result[0] == {"range": {"path": "score", "gte": 10, "lt": 100}}


class TestWrapInCompoundFilter:
    """Tests for compound filter wrapper."""

    def test_wrap_filters_only(self):
        """Should wrap filters in compound."""
        filters = [{"equals": {"path": "agent_id", "value": "test"}}]
        result = wrap_in_compound_filter(filters)
        assert result == {"compound": {"filter": filters}}

    def test_wrap_with_must(self):
        """Should include must clauses."""
        filters = [{"equals": {"path": "agent_id", "value": "test"}}]
        must = [{"text": {"query": "hello", "path": "content"}}]
        result = wrap_in_compound_filter(filters, must_clauses=must)
        assert result["compound"]["filter"] == filters
        assert result["compound"]["must"] == must

    def test_wrap_with_should(self):
        """Should include should clauses."""
        filters = [{"equals": {"path": "agent_id", "value": "test"}}]
        should = [{"text": {"query": "hello", "path": "content"}}]
        result = wrap_in_compound_filter(filters, should_clauses=should)
        assert result["compound"]["filter"] == filters
        assert result["compound"]["should"] == should

    def test_wrap_with_must_not(self):
        """Should include mustNot clauses."""
        filters = [{"equals": {"path": "agent_id", "value": "test"}}]
        must_not = [{"equals": {"path": "status", "value": "deleted"}}]
        result = wrap_in_compound_filter(filters, must_not_clauses=must_not)
        assert result["compound"]["filter"] == filters
        assert result["compound"]["mustNot"] == must_not

    def test_wrap_all_clauses(self):
        """Should include all clause types."""
        filters = [{"equals": {"path": "agent_id", "value": "test"}}]
        must = [{"text": {"query": "hello", "path": "content"}}]
        should = [{"text": {"query": "world", "path": "content"}}]
        must_not = [{"equals": {"path": "status", "value": "deleted"}}]
        result = wrap_in_compound_filter(
            filters, must_clauses=must, should_clauses=should, must_not_clauses=must_not
        )
        assert result["compound"]["filter"] == filters
        assert result["compound"]["must"] == must
        assert result["compound"]["should"] == should
        assert result["compound"]["mustNot"] == must_not
