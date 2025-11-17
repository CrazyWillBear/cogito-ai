import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from ai.nodes.router import router


class TestRouter(unittest.TestCase):
    """Test suite for the router node function."""

    @patch('ai.nodes.router.llama_low_temp')
    def test_router_returns_research_on_yes(self, mock_llama):
        """Test router returns 'research' when LLM responds with 'yes'."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.lower.return_value = 'yes'
        mock_llama.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="What is quantum physics?")]
        }

        # Call router
        result = router(state)

        # Assertions
        self.assertEqual(result, 'research')
        mock_llama.invoke.assert_called_once()

    @patch('ai.nodes.router.llama_low_temp')
    def test_router_returns_chat_on_no(self, mock_llama):
        """Test router returns 'chat' when LLM responds with 'no'."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.lower.return_value = 'no'
        mock_llama.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Hello!")]
        }

        # Call router
        result = router(state)

        # Assertions
        self.assertEqual(result, 'chat')
        mock_llama.invoke.assert_called_once()

    @patch('ai.nodes.router.llama_low_temp')
    def test_router_handles_yes_in_sentence(self, mock_llama):
        """Test router detects 'yes' even when it's part of a longer response."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.lower.return_value = 'yes, this requires research'
        mock_llama.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Explain relativity")]
        }

        # Call router
        result = router(state)

        # Assertions
        self.assertEqual(result, 'research')

    @patch('ai.nodes.router.llama_low_temp')
    def test_router_handles_case_insensitive(self, mock_llama):
        """Test router handles case-insensitive 'Yes' responses."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.lower.return_value = 'yes'
        mock_llama.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Research needed question")]
        }

        # Call router
        result = router(state)

        # Assertions
        self.assertEqual(result, 'research')

    @patch('ai.nodes.router.llama_low_temp')
    def test_router_with_multiple_messages(self, mock_llama):
        """Test router with multiple messages in history."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.lower.return_value = 'no'
        mock_llama.invoke.return_value = mock_response

        # Create state with multiple messages
        state = {
            'messages': [
                HumanMessage(content="Hi"),
                AIMessage(content="Hello!"),
                HumanMessage(content="How are you?")
            ]
        }

        # Call router
        result = router(state)

        # Assertions
        self.assertEqual(result, 'chat')
        # Verify that messages were passed in prompt
        call_args = mock_llama.invoke.call_args[0][0]
        self.assertIn('messages', call_args)

    @patch('ai.nodes.router.llama_low_temp')
    def test_router_prompt_contains_instructions(self, mock_llama):
        """Test that router prompt contains proper instructions."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.lower.return_value = 'no'
        mock_llama.invoke.return_value = mock_response

        # Create state
        state = {
            'messages': [HumanMessage(content="Test message")]
        }

        # Call router
        router(state)

        # Get the prompt that was passed
        call_args = mock_llama.invoke.call_args[0][0]

        # Verify prompt contains key instructions
        self.assertIn('router', call_args.lower())
        self.assertIn('research', call_args.lower())
        self.assertIn('yes', call_args.lower())
        self.assertIn('no', call_args.lower())


if __name__ == '__main__':
    unittest.main()

