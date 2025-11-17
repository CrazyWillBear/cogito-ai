import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, SystemMessage

from ai.research_agent.nodes.assess_summary import assess_summary


class TestAssessSummary(unittest.TestCase):
    """Test suite for the assess_summary node function."""

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_returns_true_on_yes(self, mock_gpt):
        """Test that assess_summary returns True when response is good."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "<thinking>The response is accurate</thinking>\nYes"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'resources': ['Aristotle, Ethics: virtue is excellence'],
            'response': 'According to Aristotle in Ethics, virtue is excellence.'
        }

        # Call assess_summary
        result = assess_summary(state)

        # Assertions
        self.assertIn('response_satisfied', result)
        self.assertTrue(result['response_satisfied'])

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_returns_false_on_no(self, mock_gpt):
        """Test that assess_summary returns False when response has issues."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "<thinking>Contains hallucinations</thinking>\nNo"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'resources': ['Resource A'],
            'response': 'This contains false information not in resources.'
        }

        # Call assess_summary
        result = assess_summary(state)

        # Assertions
        self.assertFalse(result['response_satisfied'])

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_strips_thinking_tags(self, mock_gpt):
        """Test that thinking tags are properly stripped."""
        # Setup mock with thinking tags
        mock_response = MagicMock()
        mock_response.content = "<thinking>Analyzing response\nChecking citations</thinking>\nYes, accurate"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'resources': ['resource'],
            'response': 'summary'
        }

        # Call assess_summary
        result = assess_summary(state)

        # Should correctly parse "yes" after stripping thinking tags
        self.assertTrue(result['response_satisfied'])

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_case_insensitive(self, mock_gpt):
        """Test that yes/no detection is case-insensitive."""
        # Setup mock with uppercase YES
        mock_response = MagicMock()
        mock_response.content = "YES"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'resources': ['resource'],
            'response': 'summary'
        }

        # Call assess_summary
        result = assess_summary(state)

        # Should recognize uppercase YES
        self.assertTrue(result['response_satisfied'])

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_prompt_structure(self, mock_gpt):
        """Test that the prompt has proper structure."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'resources': ['resource'],
            'response': 'summary'
        }

        # Call assess_summary
        assess_summary(state)

        # Check that invoke was called with proper message structure
        call_args = mock_gpt.invoke.call_args[0][0]

        # Should have SystemMessage and HumanMessage
        self.assertEqual(len(call_args), 2)
        self.assertIsInstance(call_args[0], SystemMessage)
        self.assertIsInstance(call_args[1], HumanMessage)

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_includes_response_in_prompt(self, mock_gpt):
        """Test that the response being assessed is included in prompt."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_gpt.invoke.return_value = mock_response

        # Create state with specific response
        state = {
            'resources': ['Specific resource'],
            'response': 'Specific response to assess'
        }

        # Call assess_summary
        assess_summary(state)

        # Get the user message
        call_args = mock_gpt.invoke.call_args[0][0]
        user_msg = call_args[1]

        # Should include response and resources
        self.assertIn('response to assess', user_msg.content)
        self.assertIn('research resources', user_msg.content)
        self.assertIn('Specific response to assess', user_msg.content)
        self.assertIn('Specific resource', user_msg.content)

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_checks_for_hallucinations(self, mock_gpt):
        """Test that system prompt mentions hallucinations."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'resources': ['resource'],
            'response': 'summary'
        }

        # Call assess_summary
        assess_summary(state)

        # Get system message
        call_args = mock_gpt.invoke.call_args[0][0]
        system_msg = call_args[0]

        # Should mention hallucinations and accuracy
        system_content = system_msg.content.lower()
        self.assertIn('hallucination', system_content)
        self.assertIn('accurate', system_content)
        self.assertIn('faithful', system_content)

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_with_empty_resources(self, mock_gpt):
        """Test assess_summary with empty resources."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "No"
        mock_gpt.invoke.return_value = mock_response

        # Create state with empty resources
        state = {
            'resources': [],
            'response': 'summary'
        }

        # Call assess_summary
        result = assess_summary(state)

        # Should work
        self.assertFalse(result['response_satisfied'])

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_uses_resources_from_state(self, mock_gpt):
        """Test that resources are correctly extracted from state."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_gpt.invoke.return_value = mock_response

        # Create state with .get() default pattern
        state = {
            'response': 'test response'
        }

        # Call assess_summary (should handle missing resources)
        result = assess_summary(state)

        # Should use empty list as default
        self.assertIsNotNone(result)

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_chain_of_thought_instruction(self, mock_gpt):
        """Test that chain-of-thought reasoning is requested."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'resources': ['resource'],
            'response': 'summary'
        }

        # Call assess_summary
        assess_summary(state)

        # Get system message
        call_args = mock_gpt.invoke.call_args[0][0]
        system_msg = call_args[0]

        # Should mention reasoning and thinking tags
        system_content = system_msg.content.lower()
        self.assertIn('thinking', system_content)
        self.assertIn('reasoning', system_content)

    @patch('ai.research_agent.nodes.assess_summary.gpt_low_temp')
    def test_assess_summary_mentions_false_quotes(self, mock_gpt):
        """Test that system prompt mentions checking for false quotes."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Yes"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'resources': ['resource'],
            'response': 'summary with "quotes"'
        }

        # Call assess_summary
        assess_summary(state)

        # Get system message
        call_args = mock_gpt.invoke.call_args[0][0]
        system_msg = call_args[0]

        # Should mention false quotes/citations
        system_content = system_msg.content.lower()
        self.assertTrue('false' in system_content or 'misleading' in system_content)


if __name__ == '__main__':
    unittest.main()

