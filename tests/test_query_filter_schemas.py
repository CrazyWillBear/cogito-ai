import unittest
from ai.research_agent.schemas.query import Filters, QueryAndFilters


class TestFilters(unittest.TestCase):
    """Test suite for the Filters schema."""

    def test_filters_both_none(self):
        """Test creating Filters with both fields as None."""
        filters = Filters()
        self.assertIsNone(filters.author)
        self.assertIsNone(filters.source_title)

    def test_filters_with_author_only(self):
        """Test creating Filters with only author specified."""
        filters = Filters(author="Aristotle")
        self.assertEqual(filters.author, "Aristotle")
        self.assertIsNone(filters.source_title)

    def test_filters_with_source_title_only(self):
        """Test creating Filters with only source_title specified."""
        filters = Filters(source_title="Nicomachean Ethics")
        self.assertIsNone(filters.author)
        self.assertEqual(filters.source_title, "Nicomachean Ethics")

    def test_filters_with_both_fields(self):
        """Test creating Filters with both fields specified."""
        filters = Filters(author="Plato", source_title="Republic")
        self.assertEqual(filters.author, "Plato")
        self.assertEqual(filters.source_title, "Republic")

    def test_filters_dict_conversion(self):
        """Test converting Filters to dictionary."""
        filters = Filters(author="Kant", source_title="Critique")
        filters_dict = filters.model_dump()
        self.assertEqual(filters_dict['author'], "Kant")
        self.assertEqual(filters_dict['source_title'], "Critique")


class TestQueryAndFilters(unittest.TestCase):
    """Test suite for the QueryAndFilters schema."""

    def test_query_and_filters_basic(self):
        """Test creating QueryAndFilters with basic query and empty filters."""
        filters = Filters()
        query_and_filters = QueryAndFilters(query="What is ethics?", filters=filters)
        self.assertEqual(query_and_filters.query, "What is ethics?")
        self.assertIsInstance(query_and_filters.filters, Filters)

    def test_query_and_filters_with_author_filter(self):
        """Test creating QueryAndFilters with author filter."""
        filters = Filters(author="Descartes")
        query_and_filters = QueryAndFilters(
            query="Mind-body dualism",
            filters=filters
        )
        self.assertEqual(query_and_filters.query, "Mind-body dualism")
        self.assertEqual(query_and_filters.filters.author, "Descartes")

    def test_query_and_filters_with_source_filter(self):
        """Test creating QueryAndFilters with source_title filter."""
        filters = Filters(source_title="Meditations")
        query_and_filters = QueryAndFilters(
            query="Cogito ergo sum",
            filters=filters
        )
        self.assertEqual(query_and_filters.query, "Cogito ergo sum")
        self.assertEqual(query_and_filters.filters.source_title, "Meditations")

    def test_query_and_filters_with_both_filters(self):
        """Test creating QueryAndFilters with both filters."""
        filters = Filters(author="Nietzsche", source_title="Beyond Good and Evil")
        query_and_filters = QueryAndFilters(
            query="Will to power",
            filters=filters
        )
        self.assertEqual(query_and_filters.query, "Will to power")
        self.assertEqual(query_and_filters.filters.author, "Nietzsche")
        self.assertEqual(query_and_filters.filters.source_title, "Beyond Good and Evil")

    def test_query_and_filters_dict_conversion(self):
        """Test converting QueryAndFilters to dictionary."""
        filters = Filters(author="Hume", source_title="Treatise")
        query_and_filters = QueryAndFilters(
            query="Causation",
            filters=filters
        )
        result_dict = query_and_filters.model_dump()
        self.assertEqual(result_dict['query'], "Causation")
        self.assertEqual(result_dict['filters']['author'], "Hume")
        self.assertEqual(result_dict['filters']['source_title'], "Treatise")

    def test_query_and_filters_from_dict(self):
        """Test creating QueryAndFilters from dictionary."""
        data = {
            "query": "Categorical imperative",
            "filters": {
                "author": "Kant",
                "source_title": "Groundwork"
            }
        }
        query_and_filters = QueryAndFilters(**data)
        self.assertEqual(query_and_filters.query, "Categorical imperative")
        self.assertEqual(query_and_filters.filters.author, "Kant")
        self.assertEqual(query_and_filters.filters.source_title, "Groundwork")

    def test_query_required_field(self):
        """Test that query field is required."""
        filters = Filters()
        with self.assertRaises(Exception):  # Pydantic ValidationError
            QueryAndFilters(filters=filters)

    def test_empty_query_string(self):
        """Test QueryAndFilters with empty query string."""
        filters = Filters()
        query_and_filters = QueryAndFilters(query="", filters=filters)
        self.assertEqual(query_and_filters.query, "")


if __name__ == '__main__':
    unittest.main()

