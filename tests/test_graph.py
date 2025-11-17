import unittest
from unittest.mock import patch, MagicMock

from ai.research_agent.research_agent import build_research_agent


class TestResearchAgentGraph(unittest.TestCase):
    """Test suite for the build_research_agent graph builder."""

    def test_build_research_agent_returns_compiled_graph(self):
        """Test that build_research_agent returns a compiled graph."""
        graph = build_research_agent()

        # Should return a compiled graph object
        self.assertIsNotNone(graph)
        # Compiled graphs should have invoke method
        self.assertTrue(hasattr(graph, 'invoke'))

    @patch('ai.research_agent.graph.StateGraph')
    def test_build_research_agent_creates_state_graph(self, mock_state_graph):
        """Test that a StateGraph is created with ResearchAgentState."""
        mock_graph_instance = MagicMock()
        mock_graph_instance.compile.return_value = MagicMock()
        mock_state_graph.return_value = mock_graph_instance

        build_research_agent()

        # Verify StateGraph was instantiated
        mock_state_graph.assert_called_once()

    @patch('ai.research_agent.graph.StateGraph')
    def test_build_research_agent_adds_all_nodes(self, mock_state_graph):
        """Test that all required nodes are added to the graph."""
        mock_graph_instance = MagicMock()
        mock_graph_instance.compile.return_value = MagicMock()
        mock_state_graph.return_value = mock_graph_instance

        build_research_agent()

        # Verify all nodes are added
        expected_nodes = [
            "entry",
            "write_query",
            "query_vector_db",
            "check_satisfaction",
            "assess_summary",
            "summarize"
        ]

        # Check that add_node was called for each expected node
        self.assertEqual(mock_graph_instance.add_node.call_count, 6)

        # Get all node names that were added
        added_nodes = [call[0][0] for call in mock_graph_instance.add_node.call_args_list]

        for node in expected_nodes:
            self.assertIn(node, added_nodes)

    @patch('ai.research_agent.graph.StateGraph')
    def test_build_research_agent_adds_edges(self, mock_state_graph):
        """Test that edges are added to connect nodes."""
        mock_graph_instance = MagicMock()
        mock_graph_instance.compile.return_value = MagicMock()
        mock_state_graph.return_value = mock_graph_instance

        build_research_agent()

        # Verify edges are added
        # Should have at least 4 regular edges based on the code
        self.assertTrue(mock_graph_instance.add_edge.call_count >= 4)

    @patch('ai.research_agent.graph.StateGraph')
    def test_build_research_agent_adds_conditional_edges(self, mock_state_graph):
        """Test that conditional edges are added."""
        mock_graph_instance = MagicMock()
        mock_graph_instance.compile.return_value = MagicMock()
        mock_state_graph.return_value = mock_graph_instance

        build_research_agent()

        # Should have 2 conditional edges
        self.assertEqual(mock_graph_instance.add_conditional_edges.call_count, 2)

    @patch('ai.research_agent.graph.StateGraph')
    def test_build_research_agent_compiles_graph(self, mock_state_graph):
        """Test that the graph is compiled before returning."""
        mock_graph_instance = MagicMock()
        mock_graph_instance.compile.return_value = MagicMock()
        mock_state_graph.return_value = mock_graph_instance

        result = build_research_agent()

        # Verify compile was called
        mock_graph_instance.compile.assert_called_once()
        # Verify the compiled graph is returned
        self.assertEqual(result, mock_graph_instance.compile.return_value)

    @patch('ai.research_agent.graph.StateGraph')
    def test_build_research_agent_check_satisfaction_conditional(self, mock_state_graph):
        """Test check_satisfaction conditional edge logic."""
        mock_graph_instance = MagicMock()
        mock_graph_instance.compile.return_value = MagicMock()
        mock_state_graph.return_value = mock_graph_instance

        build_research_agent()

        # Get the conditional edge for check_satisfaction
        conditional_calls = mock_graph_instance.add_conditional_edges.call_args_list

        # Find the check_satisfaction conditional
        check_satisfaction_conditional = None
        for call in conditional_calls:
            if call[0][0] == "check_satisfaction":
                check_satisfaction_conditional = call[0][1]
                break

        self.assertIsNotNone(check_satisfaction_conditional)

        # Test the lambda function
        satisfied_state = {"query_satisfied": True}
        unsatisfied_state = {"query_satisfied": False}

        self.assertEqual(check_satisfaction_conditional(satisfied_state), "summarize")
        self.assertEqual(check_satisfaction_conditional(unsatisfied_state), "write_query")

    @patch('ai.research_agent.graph.StateGraph')
    def test_build_research_agent_assess_summary_conditional(self, mock_state_graph):
        """Test assess_summary conditional edge logic."""
        mock_graph_instance = MagicMock()
        mock_graph_instance.compile.return_value = MagicMock()
        mock_state_graph.return_value = mock_graph_instance

        build_research_agent()

        # Get the conditional edge for assess_summary
        conditional_calls = mock_graph_instance.add_conditional_edges.call_args_list

        # Find the assess_summary conditional
        assess_summary_conditional = None
        for call in conditional_calls:
            if call[0][0] == "assess_summary":
                assess_summary_conditional = call[0][1]
                break

        self.assertIsNotNone(assess_summary_conditional)

        # Test the lambda function
        from langgraph.constants import END

        satisfied_state = {"response_satisfied": True}
        unsatisfied_state = {"response_satisfied": False}

        self.assertEqual(assess_summary_conditional(satisfied_state), END)
        self.assertEqual(assess_summary_conditional(unsatisfied_state), "summarize")


if __name__ == '__main__':
    unittest.main()

