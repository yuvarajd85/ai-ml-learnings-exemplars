"""
Test file for the Agentic Chatbot components
"""
import unittest
from unittest.mock import patch, MagicMock
from agent import run_agent

class TestAgentComponents(unittest.TestCase):

    def test_agent_run(self):
        """Test basic agent execution"""
        # This will just test that the function doesn't crash
        response = run_agent("What is the weather like today?")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_simple_question(self):
        """Test simple question"""
        response = run_agent("Hello!")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()