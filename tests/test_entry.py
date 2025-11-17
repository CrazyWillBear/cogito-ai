import unittest
from ai.research_agent.nodes.entry import entry


class TestEntry(unittest.TestCase):
    """Test suite for the entry node function."""

    def test_entry_returns_initial_state(self):
        """Test that entry returns properly initialized state."""
        state = {'messages': []}
        result = entry(state)

        # Check all fields are initialized
        self.assertIn('response', result)
        self.assertIn('response_feedback', result)
        self.assertIn('response_satisfied', result)
        self.assertIn('query', result)
        self.assertIn('query_filters', result)
        self.assertIn('queries_made', result)
        self.assertIn('resources', result)
        self.assertIn('query_satisfied', result)

    def test_entry_initializes_none_values(self):
        """Test that entry initializes None values correctly."""
        state = {}
        result = entry(state)

        self.assertIsNone(result['response'])
        self.assertIsNone(result['response_feedback'])
        self.assertIsNone(result['query'])
        self.assertIsNone(result['query_filters'])

    def test_entry_initializes_boolean_values(self):
        """Test that entry initializes boolean values to False."""
        state = {}
        result = entry(state)

        self.assertFalse(result['response_satisfied'])
        self.assertFalse(result['query_satisfied'])

    def test_entry_initializes_list_values(self):
        """Test that entry initializes list values as empty lists."""
        state = {}
        result = entry(state)

        self.assertEqual(result['queries_made'], [])
        self.assertEqual(result['resources'], [])
        self.assertIsInstance(result['queries_made'], list)
        self.assertIsInstance(result['resources'], list)

    def test_entry_ignores_existing_state(self):
        """Test that entry resets state regardless of existing values."""
        state = {
            'response': 'old response',
            'response_satisfied': True,
            'queries_made': ['old query'],
            'resources': ['old resource']
        }
        result = entry(state)

        # Should reset all values
        self.assertIsNone(result['response'])
        self.assertFalse(result['response_satisfied'])
        self.assertEqual(result['queries_made'], [])
        self.assertEqual(result['resources'], [])

    def test_entry_returns_dict(self):
        """Test that entry returns a dictionary."""
        state = {}
        result = entry(state)

        self.assertIsInstance(result, dict)

    def test_entry_with_empty_state(self):
        """Test entry with completely empty state."""
        state = {}
        result = entry(state)

        # Should return a fully initialized state
        self.assertEqual(len(result), 8)  # 8 fields


if __name__ == '__main__':
    unittest.main()

