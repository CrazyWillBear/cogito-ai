import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ai.research_agent.nodes.write_queries import write_queries


class TestWriteQuery(unittest.TestCase):
    """Test suite for the write_query node function."""

    @patch('ai.research_agent.nodes.write_query.gpt_low_temp')
    def test_write_query_basic(self, mock_gpt):
        """Test basic query writing with no previous queries."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = '{"query": "test query", "filters": {"author": null, "source_title": null}}'
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="What is ethics?")],
            'resources': [],
            'queries_made': []
        }

        # Call write_query
        result = write_queries(state)

        # Assertions
        self.assertIn('query', result)
        self.assertIn('queries_made', result)
        self.assertEqual(result['query']['query'], 'test query')
        self.assertEqual(len(result['queries_made']), 1)

    @patch('ai.research_agent.nodes.write_query.gpt_low_temp')
    def test_write_query_with_author_filter(self, mock_gpt):
        """Test query writing with author filter."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = '{"query": "virtue ethics", "filters": {"author": "Aristotle", "source_title": null}}'
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="What does Aristotle say about virtue?")],
            'resources': [],
            'queries_made': []
        }

        # Call write_query
        result = write_queries(state)

        # Assertions
        self.assertEqual(result['query']['query'], 'virtue ethics')
        self.assertEqual(result['query']['filters']['author'], 'Aristotle')

    @patch('ai.research_agent.nodes.write_query.gpt_low_temp')
    def test_write_query_with_source_filter(self, mock_gpt):
        """Test query writing with source_title filter."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = '{"query": "forms", "filters": {"author": null, "source_title": "Republic"}}'
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Explain Plato's Republic")],
            'resources': [],
            'queries_made': []
        }

        # Call write_query
        result = write_queries(state)

        # Assertions
        self.assertEqual(result['query']['filters']['source_title'], 'Republic')

    @patch('ai.research_agent.nodes.write_query.gpt_low_temp')
    def test_write_query_with_both_filters(self, mock_gpt):
        """Test query writing with both author and source_title filters."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = '{"query": "categorical imperative", "filters": {"author": "Kant", "source_title": "Groundwork"}}'
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Kant's categorical imperative in Groundwork")],
            'resources': [],
            'queries_made': []
        }

        # Call write_query
        result = write_queries(state)

        # Assertions
        self.assertEqual(result['query']['filters']['author'], 'Kant')
        self.assertEqual(result['query']['filters']['source_title'], 'Groundwork')

    @patch('ai.research_agent.nodes.write_query.gpt_low_temp')
    def test_write_query_accumulates_queries(self, mock_gpt):
        """Test that queries are accumulated in queries_made list."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = '{"query": "second query", "filters": {"author": null, "source_title": null}}'
        mock_gpt.invoke.return_value = mock_response

        # Create state with existing queries
        previous_query = {"query": "first query", "filters": {}}
        state = {
            'messages': [HumanMessage(content="Another question")],
            'resources': [],
            'queries_made': [previous_query]
        }

        # Call write_query
        result = write_queries(state)

        # Assertions
        self.assertEqual(len(result['queries_made']), 2)
        self.assertEqual(result['queries_made'][0]['query'], 'first query')
        self.assertEqual(result['queries_made'][1]['query'], 'second query')

    @patch('ai.research_agent.nodes.write_query.gpt_low_temp')
    def test_write_query_includes_previous_research(self, mock_gpt):
        """Test that previous research is included in the prompt."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = '{"query": "refined query", "filters": {"author": null, "source_title": null}}'
        mock_gpt.invoke.return_value = mock_response

        # Create state with previous resources and queries
        state = {
            'messages': [HumanMessage(content="Tell me more")],
            'resources': ['resource1', 'resource2'],
            'queries_made': [{'query': 'previous query'}]
        }

        # Call write_query
        result = write_queries(state)

        # Check that invoke was called with messages including previous research
        call_args = mock_gpt.invoke.call_args[0][0]
        self.assertIsInstance(call_args, list)
        self.assertTrue(len(call_args) >= 2)

        # Verify HumanMessage contains references to previous research
        user_message_content = call_args[1].content
        self.assertIn('previous research results', user_message_content)
        self.assertIn('previous queries made', user_message_content)

    @patch('ai.research_agent.nodes.write_query.gpt_low_temp')
    def test_write_query_handles_empty_content(self, mock_gpt):
        """Test handling when response content is empty."""
        # Setup mock with empty content
        mock_response = MagicMock()
        mock_response.content = ''
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Test")],
            'resources': [],
            'queries_made': []
        }

        # This should raise an error due to invalid JSON
        with self.assertRaises(Exception):
            write_queries(state)

    @patch('ai.research_agent.nodes.write_query.gpt_low_temp')
    def test_write_query_prompt_structure(self, mock_gpt):
        """Test that the prompt has proper structure."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = '{"query": "test", "filters": {"author": null, "source_title": null}}'
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': [],
            'queries_made': []
        }

        # Call write_query
        write_queries(state)

        # Check that invoke was called with proper message structure
        call_args = mock_gpt.invoke.call_args[0][0]

        # Should have SystemMessage and HumanMessage
        self.assertIsInstance(call_args[0], SystemMessage)
        self.assertIsInstance(call_args[1], HumanMessage)

        # System message should contain instructions
        system_content = call_args[0].content
        self.assertIn('semantic search', system_content.lower())
        self.assertIn('json', system_content.lower())


if __name__ == '__main__':
    unittest.main()

