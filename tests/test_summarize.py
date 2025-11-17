import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, SystemMessage

from ai.research_agent.nodes.summarize import summarize


class TestSummarize(unittest.TestCase):
    """Test suite for the summarize node function."""

    @patch('ai.research_agent.nodes.summarize.gpt_low_temp')
    def test_summarize_basic(self, mock_gpt):
        """Test basic summarization with messages and resources."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "This is a summary with citations."
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="What is virtue ethics?")],
            'resources': ['resource1', 'resource2']
        }

        # Call summarize
        result = summarize(state)

        # Assertions
        self.assertIn('response', result)
        self.assertEqual(result['response'], "This is a summary with citations.")
        mock_gpt.invoke.assert_called_once()

    @patch('ai.research_agent.nodes.summarize.gpt_low_temp')
    def test_summarize_includes_system_message(self, mock_gpt):
        """Test that summarize includes proper system message."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Summary"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': ['resource']
        }

        # Call summarize
        summarize(state)

        # Get the call arguments
        call_args = mock_gpt.invoke.call_args[0][0]

        # Should have original messages plus SystemMessage and HumanMessage
        self.assertTrue(len(call_args) >= 3)

        # Check that SystemMessage exists with proper content
        system_messages = [msg for msg in call_args if isinstance(msg, SystemMessage)]
        self.assertTrue(len(system_messages) > 0)

        system_content = system_messages[0].content
        self.assertIn('research resources', system_content.lower())
        self.assertIn('citing', system_content.lower() or 'mla', system_content.lower())

    @patch('ai.research_agent.nodes.summarize.gpt_low_temp')
    def test_summarize_includes_resources_in_prompt(self, mock_gpt):
        """Test that resources are included in the prompt."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Summary with resources"
        mock_gpt.invoke.return_value = mock_response

        # Create state with specific resources
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': ['Aristotle, Ethics', 'Plato, Republic']
        }

        # Call summarize
        summarize(state)

        # Get the call arguments
        call_args = mock_gpt.invoke.call_args[0][0]

        # The last message should be HumanMessage with resources
        last_message = call_args[-1]
        self.assertIsInstance(last_message, HumanMessage)

        # Resources should be stringified in the message
        self.assertIn('Aristotle', str(last_message.content))
        self.assertIn('Plato', str(last_message.content))

    @patch('ai.research_agent.nodes.summarize.gpt_low_temp')
    def test_summarize_with_multiple_messages(self, mock_gpt):
        """Test summarization with multiple messages in history."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Comprehensive summary"
        mock_gpt.invoke.return_value = mock_response

        # Create state with multiple messages
        state = {
            'messages': [
                HumanMessage(content="First question"),
                HumanMessage(content="Follow-up question")
            ],
            'resources': ['resource1', 'resource2', 'resource3']
        }

        # Call summarize
        result = summarize(state)

        # Assertions
        self.assertEqual(result['response'], "Comprehensive summary")

        # Verify all original messages are included
        call_args = mock_gpt.invoke.call_args[0][0]
        original_messages = [msg for msg in call_args[:2]]
        self.assertEqual(len(original_messages), 2)

    @patch('ai.research_agent.nodes.summarize.gpt_low_temp')
    def test_summarize_with_empty_resources(self, mock_gpt):
        """Test summarization with empty resources list."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Summary without resources"
        mock_gpt.invoke.return_value = mock_response

        # Create state with empty resources
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': []
        }

        # Call summarize
        result = summarize(state)

        # Should still work
        self.assertEqual(result['response'], "Summary without resources")

    @patch('ai.research_agent.nodes.summarize.gpt_low_temp')
    def test_summarize_returns_string_content(self, mock_gpt):
        """Test that summarize returns string content from response."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Test summary content"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': ['resource']
        }

        # Call summarize
        result = summarize(state)

        # Verify string is returned
        self.assertIsInstance(result['response'], str)

    @patch('ai.research_agent.nodes.summarize.gpt_low_temp')
    def test_summarize_mla8_citation_instruction(self, mock_gpt):
        """Test that MLA8 format is mentioned in system prompt."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = "Summary"
        mock_gpt.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Question")],
            'resources': ['resource']
        }

        # Call summarize
        summarize(state)

        # Get the call arguments
        call_args = mock_gpt.invoke.call_args[0][0]

        # Find SystemMessage
        system_messages = [msg for msg in call_args if isinstance(msg, SystemMessage)]
        system_content = system_messages[0].content

        # Should mention MLA8
        self.assertIn('MLA8', system_content)


if __name__ == '__main__':
    unittest.main()

