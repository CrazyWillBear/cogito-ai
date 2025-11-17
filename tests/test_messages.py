import unittest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ai.util.messages import get_last_user_message


class TestGetLastUserMessage(unittest.TestCase):
    """Test suite for the get_last_user_message utility function."""

    def test_get_last_user_message_single_human(self):
        """Test retrieving the last user message from a list with one HumanMessage."""
        messages = [HumanMessage(content="Hello, how are you?")]
        result = get_last_user_message(messages)
        self.assertEqual(result, "Hello, how are you?")

    def test_get_last_user_message_multiple_messages(self):
        """Test retrieving the last user message from a mixed list of messages."""
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="AI response"),
            HumanMessage(content="Second message"),
            AIMessage(content="Another AI response"),
        ]
        result = get_last_user_message(messages)
        self.assertEqual(result, "Second message")

    def test_get_last_user_message_no_human_message(self):
        """Test when there are no HumanMessage objects in the list."""
        messages = [
            AIMessage(content="AI response"),
            SystemMessage(content="System message"),
        ]
        result = get_last_user_message(messages)
        self.assertIsNone(result)

    def test_get_last_user_message_empty_list(self):
        """Test with an empty message list."""
        messages = []
        result = get_last_user_message(messages)
        self.assertIsNone(result)

    def test_get_last_user_message_human_at_end(self):
        """Test when HumanMessage is at the end of the list."""
        messages = [
            AIMessage(content="AI response"),
            SystemMessage(content="System message"),
            HumanMessage(content="Last user message"),
        ]
        result = get_last_user_message(messages)
        self.assertEqual(result, "Last user message")

    def test_get_last_user_message_human_at_beginning(self):
        """Test when HumanMessage is only at the beginning."""
        messages = [
            HumanMessage(content="First user message"),
            AIMessage(content="AI response 1"),
            AIMessage(content="AI response 2"),
        ]
        result = get_last_user_message(messages)
        self.assertEqual(result, "First user message")


if __name__ == '__main__':
    unittest.main()

