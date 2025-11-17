import unittest
from unittest.mock import patch, MagicMock


class TestLlamaModel(unittest.TestCase):
    """Test suite for the Llama model configuration."""

    @patch('ai.models.llama.ChatOllama')
    def test_llama_model_initialization(self, mock_chat_ollama):
        """Test that the Llama model is initialized with correct parameters."""
        from ai.models import llama

        # Verify ChatOllama was called with correct parameters
        mock_chat_ollama.assert_called_once_with(
            model="llama3.2:3b",
            temperature=0.0
        )

    def test_llama_model_exists(self):
        """Test that llama_low_temp is accessible."""
        from ai.models.llama import llama_low_temp
        self.assertIsNotNone(llama_low_temp)

    def test_llama_model_attributes(self):
        """Test that llama_low_temp has expected attributes."""
        from ai.models.llama import llama_low_temp
        # Check that it has an invoke method (duck typing)
        self.assertTrue(hasattr(llama_low_temp, 'invoke'))


class TestGPTModel(unittest.TestCase):
    """Test suite for the GPT model configuration."""

    @patch('ai.models.gpt.ChatOpenAI')
    def test_gpt_model_initialization(self, mock_chat_openai):
        """Test that the GPT model is initialized with correct parameters."""
        from ai.models import gpt

        # Verify ChatOpenAI was called with correct parameters
        mock_chat_openai.assert_called_once_with(
            model="gpt-5",
            temperature=0.0
        )

    def test_gpt_model_exists(self):
        """Test that gpt_low_temp is accessible."""
        from ai.models.gpt import gpt_low_temp
        self.assertIsNotNone(gpt_low_temp)

    def test_gpt_model_attributes(self):
        """Test that gpt_low_temp has expected attributes."""
        from ai.models.gpt import gpt_low_temp
        # Check that it has an invoke method (duck typing)
        self.assertTrue(hasattr(gpt_low_temp, 'invoke'))


if __name__ == '__main__':
    unittest.main()

