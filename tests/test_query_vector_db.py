import unittest
from unittest.mock import patch, MagicMock, call

from ai.research_agent.nodes.query_vector_db import query_vector_db


class TestQueryVectorDb(unittest.TestCase):
    """Test suite for the query_vector_db node function."""

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.psycopg2.connect')
    @patch('ai.research_agent.nodes.query_vector_db.QdrantClient')
    def test_query_vector_db_basic(self, mock_qdrant, mock_psycopg2, mock_embed):
        """Test basic vector database query without filters."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['result1', 'result2']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_qdrant.return_value = mock_qdrant_instance

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_psycopg2.return_value = mock_conn

        # Create state
        state = {
            'query': {
                'query': 'test query',
                'filters': None
            },
            'resources': []
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions
        self.assertIn('resources', result)
        self.assertEqual(len(result['resources']), 2)
        mock_embed.assert_called_once_with('test query')
        mock_qdrant_instance.query_points.assert_called_once()

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.process.extractOne')
    @patch('ai.research_agent.nodes.query_vector_db.psycopg2.connect')
    @patch('ai.research_agent.nodes.query_vector_db.QdrantClient')
    def test_query_vector_db_with_author_filter(self, mock_qdrant, mock_psycopg2, mock_extract, mock_embed):
        """Test vector database query with author filter."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_extract.return_value = ('Aristotle', 95)  # (match, score)

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['result1']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_qdrant.return_value = mock_qdrant_instance

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [('Aristotle',), ('Plato',), ('Kant',)]
        mock_conn.cursor.return_value = mock_cur
        mock_psycopg2.return_value = mock_conn

        # Create state with author filter
        state = {
            'query': {
                'query': 'virtue ethics',
                'filters': {'author': 'Aristotle'}
            },
            'resources': []
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions
        self.assertIn('resources', result)
        mock_cur.execute.assert_called()
        mock_extract.assert_called_once()

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.process.extractOne')
    @patch('ai.research_agent.nodes.query_vector_db.psycopg2.connect')
    @patch('ai.research_agent.nodes.query_vector_db.QdrantClient')
    def test_query_vector_db_with_source_filter(self, mock_qdrant, mock_psycopg2, mock_extract, mock_embed):
        """Test vector database query with source_title filter."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_extract.return_value = ('Republic', 90)

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['result1']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_qdrant.return_value = mock_qdrant_instance

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [('Republic',), ('Ethics',)]
        mock_conn.cursor.return_value = mock_cur
        mock_psycopg2.return_value = mock_conn

        # Create state with source filter
        state = {
            'query': {
                'query': 'theory of forms',
                'filters': {'source_title': 'Republic'}
            },
            'resources': []
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions
        self.assertIn('resources', result)

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.process.extractOne')
    @patch('ai.research_agent.nodes.query_vector_db.psycopg2.connect')
    @patch('ai.research_agent.nodes.query_vector_db.QdrantClient')
    def test_query_vector_db_with_both_filters(self, mock_qdrant, mock_psycopg2, mock_extract, mock_embed):
        """Test vector database query with both author and source_title filters."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_extract.side_effect = [('Kant', 95), ('Groundwork', 90)]

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['result1']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_qdrant.return_value = mock_qdrant_instance

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.side_effect = [
            [('Kant',), ('Hume',)],  # authors
            [('Groundwork',), ('Critique',)]  # sources
        ]
        mock_conn.cursor.return_value = mock_cur
        mock_psycopg2.return_value = mock_conn

        # Create state with both filters
        state = {
            'query': {
                'query': 'categorical imperative',
                'filters': {'author': 'Kant', 'source_title': 'Groundwork'}
            },
            'resources': []
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions
        self.assertIn('resources', result)
        self.assertEqual(mock_extract.call_count, 2)

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.psycopg2.connect')
    @patch('ai.research_agent.nodes.query_vector_db.QdrantClient')
    def test_query_vector_db_accumulates_resources(self, mock_qdrant, mock_psycopg2, mock_embed):
        """Test that new resources are accumulated with existing ones."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = ['new_result1', 'new_result2']
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_qdrant.return_value = mock_qdrant_instance

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_psycopg2.return_value = mock_conn

        # Create state with existing resources
        state = {
            'query': {
                'query': 'test query',
                'filters': None
            },
            'resources': ['old_result1', 'old_result2']
        }

        # Call query_vector_db
        result = query_vector_db(state)

        # Assertions - should have 4 total resources (2 old + 2 new)
        self.assertEqual(len(result['resources']), 4)
        self.assertIn('old_result1', result['resources'])
        self.assertIn('new_result1', result['resources'])

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.psycopg2.connect')
    @patch('ai.research_agent.nodes.query_vector_db.QdrantClient')
    def test_query_vector_db_closes_connections(self, mock_qdrant, mock_psycopg2, mock_embed):
        """Test that database connections are properly closed."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_qdrant.return_value = mock_qdrant_instance

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_psycopg2.return_value = mock_conn

        # Create state
        state = {
            'query': {
                'query': 'test',
                'filters': None
            },
            'resources': []
        }

        # Call query_vector_db
        query_vector_db(state)

        # Assertions - all connections should be closed
        mock_cur.close.assert_called_once()
        mock_conn.close.assert_called_once()
        mock_qdrant_instance.close.assert_called_once()

    @patch('ai.research_agent.nodes.query_vector_db.embed')
    @patch('ai.research_agent.nodes.query_vector_db.process.extractOne')
    @patch('ai.research_agent.nodes.query_vector_db.psycopg2.connect')
    @patch('ai.research_agent.nodes.query_vector_db.QdrantClient')
    def test_query_vector_db_fuzzy_matching(self, mock_qdrant, mock_psycopg2, mock_extract, mock_embed):
        """Test that fuzzy matching is used for filters."""
        # Setup mocks
        mock_embed.return_value = [0.1, 0.2, 0.3]
        # Simulate fuzzy match finding close match
        mock_extract.return_value = ('Aristoteles', 85)  # Close but not exact

        mock_qdrant_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.points = []
        mock_qdrant_instance.query_points.return_value = mock_results
        mock_qdrant.return_value = mock_qdrant_instance

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [('Aristoteles',), ('Plato',)]
        mock_conn.cursor.return_value = mock_cur
        mock_psycopg2.return_value = mock_conn

        # Create state with slightly misspelled author
        state = {
            'query': {
                'query': 'ethics',
                'filters': {'author': 'Aristotle'}  # User typed this
            },
            'resources': []
        }

        # Call query_vector_db
        query_vector_db(state)

        # Verify fuzzy matching was used
        mock_extract.assert_called_once()
        # First arg should be the query, second should be list of candidates
        args = mock_extract.call_args[0]
        self.assertEqual(args[0], 'Aristotle')


if __name__ == '__main__':
    unittest.main()

