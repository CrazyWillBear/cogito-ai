import json
import sys
import types
import pytest


def setup_dummy_langchain_module():
    """Setup mock LangChain message classes for testing without installing langchain."""
    if "langchain_core" in sys.modules:
        return  # Already set up

    class HumanMessage:
        def __init__(self, content):
            self.content = content

        def __repr__(self):
            return f"HumanMessage(content={self.content!r})"

        def __eq__(self, other):
            return isinstance(other, HumanMessage) and self.content == other.content

    class AIMessage:
        def __init__(self, content):
            self.content = content

        def __repr__(self):
            return f"AIMessage(content={self.content!r})"

        def __eq__(self, other):
            return isinstance(other, AIMessage) and self.content == other.content

    class SystemMessage:
        def __init__(self, content):
            self.content = content

        def __repr__(self):
            return f"SystemMessage(content={self.content!r})"

        def __eq__(self, other):
            return isinstance(other, SystemMessage) and self.content == other.content

    pkg = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")
    messages.HumanMessage = HumanMessage
    messages.AIMessage = AIMessage
    messages.SystemMessage = SystemMessage
    pkg.messages = messages

    sys.modules["langchain_core"] = pkg
    sys.modules["langchain_core.messages"] = messages


# Set up dummy module before importing
setup_dummy_langchain_module()
from chat.history import ChatHistory


class TestChatHistoryInitialization:
    """Test ChatHistory object initialization."""

    def test_init_empty(self):
        """Test that ChatHistory initializes with empty lists."""
        ch = ChatHistory()
        assert ch.messages == []
        assert ch.history == []
        assert isinstance(ch.messages, list)
        assert isinstance(ch.history, list)


class TestFromJsonWithList:
    """Test from_json method with Python list input."""

    def test_from_json_valid_list(self):
        """Test from_json with valid list of message dicts."""
        ch = ChatHistory()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"}
        ]
        ch.from_json(messages)
        assert len(ch.messages) == 2
        assert ch.messages[0]["role"] == "user"
        assert ch.messages[0]["content"] == "hello"
        assert ch.messages[1]["role"] == "assistant"
        assert ch.messages[1]["content"] == "hi there"

    def test_from_json_empty_list(self):
        """Test from_json with empty list."""
        ch = ChatHistory()
        ch.from_json([])
        assert ch.messages == []

    def test_from_json_all_roles(self):
        """Test from_json with all supported role types."""
        ch = ChatHistory()
        messages = [
            {"role": "system", "content": "sys msg"},
            {"role": "user", "content": "user msg"},
            {"role": "human", "content": "human msg"},
            {"role": "assistant", "content": "assistant msg"},
            {"role": "ai", "content": "ai msg"},
            {"role": "bot", "content": "bot msg"},
        ]
        ch.from_json(messages)
        assert len(ch.messages) == 6
        assert ch.messages[0]["role"] == "system"
        assert ch.messages[1]["role"] == "user"
        assert ch.messages[2]["role"] == "human"

    def test_from_json_unknown_role(self):
        """Test from_json accepts unknown roles (they're stored but ignored in langchain conversion)."""
        ch = ChatHistory()
        messages = [
            {"role": "custom_role", "content": "custom message"},
            {"role": "user", "content": "normal message"}
        ]
        ch.from_json(messages)
        assert len(ch.messages) == 2
        assert ch.messages[0]["role"] == "custom_role"

    def test_from_json_none_content(self):
        """Test from_json converts None content to empty string."""
        ch = ChatHistory()
        messages = [{"role": "user", "content": None}]
        ch.from_json(messages)
        assert ch.messages[0]["content"] == ""

    def test_from_json_numeric_content(self):
        """Test from_json converts numeric content to string."""
        ch = ChatHistory()
        messages = [
            {"role": "user", "content": 123},
            {"role": "assistant", "content": 45.67}
        ]
        ch.from_json(messages)
        assert ch.messages[0]["content"] == "123"
        assert ch.messages[1]["content"] == "45.67"

    def test_from_json_empty_content(self):
        """Test from_json with empty string content."""
        ch = ChatHistory()
        messages = [{"role": "user", "content": ""}]
        ch.from_json(messages)
        assert ch.messages[0]["content"] == ""

    def test_from_json_multiline_content(self):
        """Test from_json with multiline content."""
        ch = ChatHistory()
        messages = [{"role": "user", "content": "line1\nline2\nline3"}]
        ch.from_json(messages)
        assert ch.messages[0]["content"] == "line1\nline2\nline3"

    def test_from_json_unicode(self):
        """Test from_json with unicode characters."""
        ch = ChatHistory()
        messages = [{"role": "user", "content": "Hello 世界 🌍"}]
        ch.from_json(messages)
        assert ch.messages[0]["content"] == "Hello 世界 🌍"

    def test_from_json_extra_fields(self):
        """Test from_json ignores extra fields in message dicts."""
        ch = ChatHistory()
        messages = [{"role": "user", "content": "hello", "extra": "ignored", "timestamp": 123456}]
        ch.from_json(messages)
        assert len(ch.messages) == 1
        assert ch.messages[0]["role"] == "user"
        assert ch.messages[0]["content"] == "hello"

    def test_from_json_multiple_calls(self):
        """Test that multiple from_json calls append to messages."""
        ch = ChatHistory()
        ch.from_json([{"role": "user", "content": "first"}])
        ch.from_json([{"role": "assistant", "content": "second"}])
        assert len(ch.messages) == 2
        assert ch.messages[0]["content"] == "first"
        assert ch.messages[1]["content"] == "second"


class TestFromJsonWithString:
    """Test from_json method with JSON string input."""

    def test_from_json_valid_string(self):
        """Test from_json with valid JSON string."""
        ch = ChatHistory()
        json_str = '[{"role": "user", "content": "hello"}]'
        ch.from_json(json_str)
        assert len(ch.messages) == 1
        assert ch.messages[0]["role"] == "user"

    def test_from_json_string_formatted(self):
        """Test from_json with pretty-printed JSON string."""
        ch = ChatHistory()
        json_str = """[
            {
                "role": "user",
                "content": "hello"
            },
            {
                "role": "assistant",
                "content": "hi"
            }
        ]"""
        ch.from_json(json_str)
        assert len(ch.messages) == 2

    def test_from_json_string_complex(self):
        """Test from_json with complex conversation as string."""
        ch = ChatHistory()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Help me with LangChain"},
            {"role": "assistant", "content": "Sure! What do you need?"}
        ]
        json_str = json.dumps(messages)
        ch.from_json(json_str)
        assert len(ch.messages) == 3
        assert ch.messages[0]["role"] == "system"


class TestFromJsonErrorCases:
    """Test from_json method error handling."""

    def test_from_json_not_list(self):
        """Test from_json raises ValueError when input is not a list."""
        ch = ChatHistory()
        with pytest.raises(ValueError, match="must be a list"):
            ch.from_json({"role": "user", "content": "hello"})

    def test_from_json_string_not_list(self):
        """Test from_json raises ValueError when JSON string is not a list."""
        ch = ChatHistory()
        with pytest.raises(ValueError, match="must be a list"):
            ch.from_json('{"role": "user", "content": "hello"}')

    def test_from_json_missing_role(self):
        """Test from_json raises ValueError when message missing 'role'."""
        ch = ChatHistory()
        with pytest.raises(ValueError, match="must be an object with 'role' and 'content'"):
            ch.from_json([{"content": "hello"}])

    def test_from_json_missing_content(self):
        """Test from_json raises ValueError when message missing 'content'."""
        ch = ChatHistory()
        with pytest.raises(ValueError, match="must be an object with 'role' and 'content'"):
            ch.from_json([{"role": "user"}])

    def test_from_json_not_dict(self):
        """Test from_json raises ValueError when message is not a dict."""
        ch = ChatHistory()
        with pytest.raises(ValueError, match="must be an object with 'role' and 'content'"):
            ch.from_json(["not a dict"])

    def test_from_json_invalid_json_string(self):
        """Test from_json raises JSONDecodeError for invalid JSON string."""
        ch = ChatHistory()
        with pytest.raises(json.JSONDecodeError):
            ch.from_json('[{"role": "user", "content": "unclosed"')

    def test_from_json_malformed_json(self):
        """Test from_json raises JSONDecodeError for malformed JSON."""
        ch = ChatHistory()
        with pytest.raises(json.JSONDecodeError):
            ch.from_json('not json at all')


    def test_from_json_list_with_mixed_valid_invalid(self):
        """Test from_json with list containing one valid and one invalid message."""
        ch = ChatHistory()
        with pytest.raises(ValueError, match="message at index 1"):
            ch.from_json([
                {"role": "user", "content": "valid"},
                {"role": "user"}  # missing content
            ])


class TestGetHistoryLangchain:
    """Test get_history_langchain method."""

    def test_get_history_langchain_basic(self):
        """Test get_history_langchain with basic conversation."""
        ch = ChatHistory()
        ch.from_json([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}
        ])
        lc_history = ch.get_history_langchain()
        assert len(lc_history) == 2
        assert type(lc_history[0]).__name__ == "HumanMessage"
        assert type(lc_history[1]).__name__ == "AIMessage"
        assert lc_history[0].content == "hello"
        assert lc_history[1].content == "hi"

    def test_get_history_langchain_all_types(self):
        """Test get_history_langchain converts all role types correctly."""
        ch = ChatHistory()
        ch.from_json([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
            {"role": "human", "content": "human"},
            {"role": "assistant", "content": "assistant"},
            {"role": "ai", "content": "ai"},
            {"role": "bot", "content": "bot"},
        ])
        lc_history = ch.get_history_langchain()
        assert len(lc_history) == 6
        assert type(lc_history[0]).__name__ == "SystemMessage"
        assert type(lc_history[1]).__name__ == "HumanMessage"
        assert type(lc_history[2]).__name__ == "HumanMessage"
        assert type(lc_history[3]).__name__ == "AIMessage"
        assert type(lc_history[4]).__name__ == "AIMessage"
        assert type(lc_history[5]).__name__ == "AIMessage"

    def test_get_history_langchain_unknown_role_skipped(self):
        """Test get_history_langchain skips unknown roles."""
        ch = ChatHistory()
        ch.from_json([
            {"role": "user", "content": "hello"},
            {"role": "unknown_role", "content": "should be skipped"},
            {"role": "assistant", "content": "hi"}
        ])
        lc_history = ch.get_history_langchain()
        assert len(lc_history) == 2
        assert type(lc_history[0]).__name__ == "HumanMessage"
        assert type(lc_history[1]).__name__ == "AIMessage"

    def test_get_history_langchain_empty(self):
        """Test get_history_langchain with no messages."""
        ch = ChatHistory()
        lc_history = ch.get_history_langchain()
        assert lc_history == []

    def test_get_history_langchain_only_unknown_roles(self):
        """Test get_history_langchain with only unknown roles returns empty."""
        ch = ChatHistory()
        ch.from_json([
            {"role": "custom1", "content": "msg1"},
            {"role": "custom2", "content": "msg2"}
        ])
        lc_history = ch.get_history_langchain()
        assert lc_history == []

    def test_get_history_langchain_content_preserved(self):
        """Test get_history_langchain preserves exact content."""
        ch = ChatHistory()
        content = "Multi\nline\nwith special chars: @#$% and unicode 世界"
        ch.from_json([{"role": "user", "content": content}])
        lc_history = ch.get_history_langchain()
        assert lc_history[0].content == content

    def test_get_history_langchain_complex_conversation(self):
        """Test get_history_langchain with realistic conversation."""
        ch = ChatHistory()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can query a vector database."},
            {"role": "user", "content": "Hey, can you help me find notes about LangChain?"},
            {"role": "assistant", "content": "Sure! I found several documents about LangChain. Would you like a summary?"},
            {"role": "user", "content": "Yes, please summarize them."},
            {"role": "assistant", "content": "LangChain helps you build LLM-powered apps that use tools and memory to reason through complex tasks."},
        ]
        ch.from_json(messages)
        lc_history = ch.get_history_langchain()

        assert len(lc_history) == 5
        assert type(lc_history[0]).__name__ == "SystemMessage"
        assert type(lc_history[1]).__name__ == "HumanMessage"
        assert type(lc_history[2]).__name__ == "AIMessage"
        assert type(lc_history[3]).__name__ == "HumanMessage"
        assert type(lc_history[4]).__name__ == "AIMessage"

        assert lc_history[0].content.startswith("You are a helpful assistant")
        assert "LangChain" in lc_history[1].content
        assert "Would you like a summary" in lc_history[2].content
        assert lc_history[3].content == "Yes, please summarize them."
        assert lc_history[4].content.startswith("LangChain helps you build")


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_full_workflow_list_input(self):
        """Test full workflow: initialize, load from list, convert to langchain."""
        ch = ChatHistory()
        assert ch.messages == []

        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"}
        ]
        ch.from_json(messages)
        assert len(ch.messages) == 2

        lc_history = ch.get_history_langchain()
        assert len(lc_history) == 2
        assert type(lc_history[0]).__name__ == "HumanMessage"
        assert type(lc_history[1]).__name__ == "AIMessage"

    def test_full_workflow_string_input(self):
        """Test full workflow: initialize, load from JSON string, convert to langchain."""
        ch = ChatHistory()
        json_str = json.dumps([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "Sure!"}
        ])
        ch.from_json(json_str)

        lc_history = ch.get_history_langchain()
        assert len(lc_history) == 3
        assert type(lc_history[0]).__name__ == "SystemMessage"
        assert type(lc_history[1]).__name__ == "HumanMessage"
        assert type(lc_history[2]).__name__ == "AIMessage"

    def test_incremental_loading(self):
        """Test loading messages incrementally."""
        ch = ChatHistory()

        ch.from_json([{"role": "user", "content": "msg1"}])
        assert len(ch.messages) == 1

        ch.from_json([{"role": "assistant", "content": "msg2"}])
        assert len(ch.messages) == 2

        ch.from_json([{"role": "user", "content": "msg3"}])
        assert len(ch.messages) == 3

        lc_history = ch.get_history_langchain()
        assert len(lc_history) == 3

    def test_large_conversation(self):
        """Test with a large conversation (100 messages)."""
        ch = ChatHistory()
        messages = []
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"message {i}"})

        ch.from_json(messages)
        assert len(ch.messages) == 100

        lc_history = ch.get_history_langchain()
        assert len(lc_history) == 100
        assert lc_history[0].content == "message 0"
        assert lc_history[99].content == "message 99"

