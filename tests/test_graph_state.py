import unittest
from ai.research_agent.schemas.graph_state import ResearchAgentState


class TestResearchAgentState(unittest.TestCase):
    """Test suite for the ResearchAgentState TypedDict."""

    def test_state_has_all_fields(self):
        """Test that ResearchAgentState has all required fields."""
        # Get the annotations
        annotations = ResearchAgentState.__annotations__

        expected_fields = {
            'messages', 'response', 'response_feedback', 'response_satisfied',
            'query', 'query_filters', 'queries_made', 'query_satisfied', 'resources'
        }

        self.assertEqual(set(annotations.keys()), expected_fields)

    def test_state_creation_with_all_fields(self):
        """Test creating a state dictionary with all fields."""
        state = {
            'messages': [],
            'response': "Test response",
            'response_feedback': "Good",
            'response_satisfied': True,
            'query': {"query": "test", "filters": {}},
            'query_filters': {},
            'queries_made': [],
            'query_satisfied': False,
            'resources': []
        }

        # Should not raise any errors
        self.assertIsInstance(state, dict)
        self.assertEqual(state['response'], "Test response")
        self.assertTrue(state['response_satisfied'])

    def test_state_creation_partial(self):
        """Test creating a state dictionary with partial fields."""
        state = {
            'messages': [],
            'response': None,
            'response_feedback': None,
            'response_satisfied': False,
            'query': None,
            'query_filters': None,
            'queries_made': [],
            'query_satisfied': False,
            'resources': []
        }

        self.assertIsNone(state['response'])
        self.assertFalse(state['response_satisfied'])
        self.assertEqual(len(state['queries_made']), 0)

    def test_state_messages_field_type(self):
        """Test that messages field accepts list of BaseMessage."""
        from langchain_core.messages import HumanMessage, AIMessage

        state = {
            'messages': [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there")
            ],
            'response': None,
            'response_feedback': None,
            'response_satisfied': False,
            'query': None,
            'query_filters': None,
            'queries_made': [],
            'query_satisfied': False,
            'resources': []
        }

        self.assertEqual(len(state['messages']), 2)
        self.assertEqual(state['messages'][0].content, "Hello")

    def test_state_boolean_fields(self):
        """Test boolean fields in state."""
        state = {
            'messages': [],
            'response': "",
            'response_feedback': "",
            'response_satisfied': True,
            'query': {},
            'query_filters': {},
            'queries_made': [],
            'query_satisfied': True,
            'resources': []
        }

        self.assertTrue(state['response_satisfied'])
        self.assertTrue(state['query_satisfied'])

    def test_state_list_fields(self):
        """Test list fields in state."""
        state = {
            'messages': [],
            'response': None,
            'response_feedback': None,
            'response_satisfied': False,
            'query': None,
            'query_filters': None,
            'queries_made': [{"query": "test1"}, {"query": "test2"}],
            'query_satisfied': False,
            'resources': ["resource1", "resource2", "resource3"]
        }

        self.assertEqual(len(state['queries_made']), 2)
        self.assertEqual(len(state['resources']), 3)

    def test_state_dict_fields(self):
        """Test dictionary fields in state."""
        state = {
            'messages': [],
            'response': None,
            'response_feedback': None,
            'response_satisfied': False,
            'query': {"query": "What is philosophy?", "filters": {"author": "Plato"}},
            'query_filters': {"author": "Aristotle", "source_title": "Ethics"},
            'queries_made': [],
            'query_satisfied': False,
            'resources': []
        }

        self.assertEqual(state['query']['query'], "What is philosophy?")
        self.assertEqual(state['query_filters']['author'], "Aristotle")


if __name__ == '__main__':
    unittest.main()

