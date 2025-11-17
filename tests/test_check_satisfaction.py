import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, SystemMessage

from ai.research_agent.nodes.assess_resources import assess_resources


class TestCheckSatisfaction(unittest.TestCase):
    """Test suite for the check_satisfaction node function."""

    @patch('ai.research_agent.nodes.check_statisfaction.gpt_low_temp')
    def test_check_satisfaction_returns_true_on_yes(self, mock_gpt):
        """Test that check_satisfaction returns True when LLM says yes."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "<thinking>Analysis here</thinking>\nYes"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="What is virtue?")],
            'resources': ['Good resource about virtue'],
            'queries_made': [{'query': 'virtue'}]
        }

        # Call check_satisfaction
        result = assess_resources(state)

        # Assertions
        self.assertIn('query_satisfied', result)
        self.assertTrue(result['query_satisfied'])

    @patch('ai.research_agent.nodes.check_statisfaction.gpt_low_temp')
    def test_check_satisfaction_returns_false_on_no(self, mock_gpt):
        """Test that check_satisfaction returns False when LLM says no."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "<thinking>Not enough info</thinking>\nNo"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Complex question")],
            'resources': ['Insufficient resource'],
            'queries_made': [{'query': 'test'}]
        }

        # Call check_satisfaction
        result = assess_resources(state)

        # Assertions
        self.assertFalse(result['query_satisfied'])

    @patch('ai.research_agent.nodes.check_statisfaction.gpt_low_temp')
    def test_check_satisfaction_max_queries_limit(self, mock_gpt):
        """Test that satisfaction is forced to True after 5 queries."""
        # Setup mock to say "no"
        mock_response = MagicMock()
        mock_response.content = "No"
        mock_gpt.invoke.return_value = mock_response

        # Create state with 5 queries
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': ['resource'],
            'queries_made': [
                {'query': 'q1'},
                {'query': 'q2'},
                {'query': 'q3'},
                {'query': 'q4'},
                {'query': 'q5'}
            ]
        }

        # Call check_satisfaction
        result = assess_resources(state)

        # Should return True due to max queries limit
        self.assertTrue(result['query_satisfied'])

    @patch('ai.research_agent.nodes.check_statisfaction.gpt_low_temp')
    def test_check_satisfaction_strips_thinking_tags(self, mock_gpt):
        """Test that thinking tags are properly stripped."""
        # Setup mock with thinking tags
        mock_response = MagicMock()
        mock_response.content = "<thinking>Detailed reasoning here\nMore reasoning</thinking>\nYes, sufficient"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': ['resource'],
            'queries_made': [{'query': 'test'}]
        }

        # Call check_satisfaction
        result = assess_resources(state)

        # Should correctly parse "yes" after stripping thinking tags
        self.assertTrue(result['query_satisfied'])

    @patch('ai.research_agent.nodes.check_statisfaction.gpt_low_temp')
    def test_check_satisfaction_case_insensitive(self, mock_gpt):
        """Test that yes/no detection is case-insensitive."""
        # Setup mock with uppercase YES
        mock_response = MagicMock()
        mock_response.content = "YES"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': ['resource'],
            'queries_made': [{'query': 'test'}]
        }

        # Call check_satisfaction
        result = assess_resources(state)

        # Should recognize uppercase YES
        self.assertTrue(result['query_satisfied'])

    @patch('ai.research_agent.nodes.check_statisfaction.gpt_low_temp')
    def test_check_satisfaction_prompt_structure(self, mock_gpt):
        """Test that the prompt has proper structure."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': ['resource'],
            'queries_made': [{'query': 'test'}]
        }

        # Call check_satisfaction
        assess_resources(state)

        # Check that invoke was called with proper message structure
        call_args = mock_gpt.invoke.call_args[0][0]

        # Should have SystemMessage and HumanMessage
        self.assertEqual(len(call_args), 2)
        self.assertIsInstance(call_args[0], SystemMessage)
        self.assertIsInstance(call_args[1], HumanMessage)

    @patch('ai.research_agent.nodes.check_statisfaction.gpt_low_temp')
    def test_check_satisfaction_includes_messages_in_prompt(self, mock_gpt):
        """Test that conversation messages are included in prompt."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Specific question about ethics")],
            'resources': ['resource'],
            'queries_made': [{'query': 'test'}]
        }

        # Call check_satisfaction
        assess_resources(state)

        # Get the user message
        call_args = mock_gpt.invoke.call_args[0][0]
        user_msg = call_args[1]

        # Should include messages and resources
        self.assertIn('conversation messages', user_msg.content)
        self.assertIn('research results', user_msg.content)

    @patch('ai.research_agent.nodes.check_statisfaction.gpt_low_temp')
    def test_check_satisfaction_with_empty_resources(self, mock_gpt):
        """Test check_satisfaction with empty resources."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "No"
        mock_gpt.invoke.return_value = mock_response

        # Create state with empty resources
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': [],
            'queries_made': [{'query': 'test'}]
        }

        # Call check_satisfaction
        result = assess_resources(state)

        # Should work and return False
        self.assertFalse(result['query_satisfied'])

    @patch('ai.research_agent.nodes.check_statisfaction.gpt_low_temp')
    def test_check_satisfaction_chain_of_thought_instruction(self, mock_gpt):
        """Test that chain-of-thought is requested in system prompt."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': ['resource'],
            'queries_made': [{'query': 'test'}]
        }

        # Call check_satisfaction
        assess_resources(state)

        # Get system message
        call_args = mock_gpt.invoke.call_args[0][0]
        system_msg = call_args[0]

        # Should mention chain-of-thought and thinking tags
        system_content = system_msg.content.lower()
        self.assertIn('thinking', system_content)
        self.assertIn('reasoning', system_content)


if __name__ == '__main__':
    unittest.main()

