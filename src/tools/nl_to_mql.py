"""
NL-to-MQL Generator.

Converts natural language questions to MongoDB queries using LLM + schema context.
Path B implementation: LLM-based (MongoDBDatabaseToolkit not available in
langchain-mongodb v0.11.0).

Security:
    - Read-only: Only find() queries, no writes
    - agent_id isolation: ALWAYS injected into every query filter
    - Collection whitelist: Only allowed memory collections
    - No $where, no JavaScript, no $function

Pattern from mongodb-natural-language-querying skill:
    Schema-aware prompting with sample documents for grounding.
"""

import json
import logging
import time
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

# Operators forbidden in generated queries for security
FORBIDDEN_OPERATORS = {"$where", "$function", "$accumulator", "$merge", "$out"}


class NLToMQLGenerator:
    """Generate MongoDB queries from natural language using LLM + schema context.

    Uses schema-aware prompting: fetches sample documents from target collections
    to ground the LLM on actual data shape, then generates find() queries.

    Args:
        db: Motor async database instance.
        allowed_collections: Optional collection whitelist. Defaults to the
            7 memory collections.
    """

    DEFAULT_COLLECTIONS = [
        "episodic_memories",
        "semantic_memories",
        "procedural_memories",
        "working_memories",
        "cache_memories",
        "entity_memories",
        "summary_memories",
    ]

    _QUERY_PROMPT = """Convert this natural language query to a MongoDB find() query.

Database schema for collection "{collection}":
{schema_context}

MANDATORY RULES:
1. Include {{"agent_id": "{agent_id}"}} in the filter (multi-tenant isolation).
2. Only generate find() queries -- NO inserts, updates, or deletes.
3. Do NOT use $where, $function, or JavaScript in filters.
4. Only query allowed collections: {allowed_collections}

Question: {question}

Return ONLY JSON: {{"collection": "<name>", "filter": {{}}, "projection": {{}}, "limit": 10}}"""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        allowed_collections: list[str] | None = None,
    ) -> None:
        self._db = db
        self._allowed_collections = allowed_collections or self.DEFAULT_COLLECTIONS

    async def generate_query(
        self,
        question: str,
        agent_id: str,
        llm: Any = None,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        """Convert natural language to MongoDB query with agent_id scoping.

        Steps:
            1. Get schema context from target collections
            2. Build prompt with schema + question
            3. Call LLM to generate MQL
            4. Validate and sanitize the generated query
            5. ALWAYS inject agent_id filter
            6. Execute and return results

        Args:
            question: Natural language question from user.
            agent_id: Agent ID for query scoping (MANDATORY).
            llm: LLM instance with async ainvoke method.

        Returns:
            Dict with generated_mql, results, execution_time, or error.
        """
        if not llm:
            return {"error": "LLM required for NL-to-MQL generation", "results": []}

        start_time = time.monotonic()

        try:
            # Respect explicit collection choice when provided, otherwise default to
            # the first allowlisted collection for a safe bounded query surface.
            target_collection = collection_name or self._allowed_collections[0]
            if target_collection not in self._allowed_collections:
                return {
                    "error": f"Collection '{target_collection}' not in allowlist",
                    "results": [],
                    "execution_time": time.monotonic() - start_time,
                }

            # Get schema context
            schema_context = await self._get_schema_context(target_collection)

            # Build and send prompt
            prompt = self._QUERY_PROMPT.format(
                collection=target_collection,
                schema_context=schema_context,
                agent_id=agent_id,
                allowed_collections=json.dumps(self._allowed_collections),
                question=question,
            )

            response = await llm.ainvoke(prompt)
            query_spec = self._parse_json_response(response.content)

            # Validate collection
            generated_collection_name = query_spec.get("collection", "")
            if generated_collection_name not in self._allowed_collections:
                return {
                    "error": f"Collection '{generated_collection_name}' not in allowlist",
                    "results": [],
                    "execution_time": time.monotonic() - start_time,
                }

            # Keep the query on the explicitly chosen collection when the caller
            # requested one. This avoids cross-collection surprises at execution time.
            if collection_name:
                generated_collection_name = collection_name

            # Validate filter for forbidden operators
            filter_query = query_spec.get("filter", {})
            validation_error = self._validate_filter(filter_query)
            if validation_error:
                return {
                    "error": validation_error,
                    "results": [],
                    "execution_time": time.monotonic() - start_time,
                }

            # ALWAYS inject agent_id (security: multi-tenant isolation)
            filter_query["agent_id"] = agent_id

            # Execute query via Motor async
            collection = self._db[generated_collection_name]
            limit = min(query_spec.get("limit", 10), 100)  # Cap at 100
            projection = query_spec.get("projection")

            cursor = collection.find(filter_query, projection)
            results = await cursor.to_list(length=limit)

            # Serialize ObjectIds
            for doc in results:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])

            elapsed = time.monotonic() - start_time

            return {
                "generated_mql": {
                    "collection": generated_collection_name,
                    "filter": filter_query,
                    "projection": projection,
                    "limit": limit,
                },
                "results": results,
                "execution_time": elapsed,
            }

        except Exception as e:
            logger.error("NL-to-MQL generation failed: %s", e)
            return {
                "error": str(e),
                "results": [],
                "execution_time": time.monotonic() - start_time,
            }

    async def _get_schema_context(self, collection_name: str) -> str:
        """Get collection schema for prompt context.

        Fetches a sample document and extracts its field structure.

        Args:
            collection_name: Name of the collection to inspect.

        Returns:
            Human-readable schema description string.
        """
        try:
            collection = self._db[collection_name]
            sample = await collection.find_one()
            if not sample:
                return f"Collection '{collection_name}' is empty. Common fields: content, agent_id, metadata, embedding, created_at."

            # Build schema from sample doc keys (exclude embedding for brevity)
            fields = []
            for key, value in sample.items():
                if key == "embedding":
                    fields.append(f"  {key}: vector (1024 dimensions)")
                else:
                    fields.append(f"  {key}: {type(value).__name__}")

            return "Fields:\n" + "\n".join(fields)

        except Exception as e:
            logger.warning("Failed to get schema context: %s", e)
            return f"Collection '{collection_name}': schema unavailable. Common fields: content, agent_id, metadata."

    @staticmethod
    def _parse_json_response(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, tolerating surrounding text.

        Args:
            content: Raw LLM response text.

        Returns:
            Parsed dict from the JSON portion.

        Raises:
            ValueError: If no valid JSON object found.
        """
        text = content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in LLM response")
        return json.loads(text[start : end + 1])

    @classmethod
    def _validate_filter(cls, filter_dict: dict[str, Any]) -> str | None:
        """Validate a query filter for forbidden operators.

        Recursively checks all keys and nested dicts/lists for:
        - $where (JavaScript execution)
        - $function (server-side JavaScript)
        - $accumulator (server-side JavaScript)

        Args:
            filter_dict: The MongoDB filter to validate.

        Returns:
            Error message if forbidden operator found, None if clean.
        """
        return cls._check_forbidden_recursive(filter_dict)

    @classmethod
    def _check_forbidden_recursive(cls, obj: Any) -> str | None:
        """Recursively check for forbidden operators in a filter.

        Args:
            obj: Any value to check (dict, list, or scalar).

        Returns:
            Error message if forbidden operator found, None if clean.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in FORBIDDEN_OPERATORS:
                    return f"Forbidden operator '{key}' not allowed in generated queries"
                result = cls._check_forbidden_recursive(value)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = cls._check_forbidden_recursive(item)
                if result:
                    return result
        return None
