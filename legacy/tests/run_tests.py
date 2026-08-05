"""Test runner for all unit tests."""

import logging
import unittest

from test_agent_communication import TestAgentCommunication
from test_agent_creation import TestAgentCreation
from test_agent_movement import TestAgentMovement
from test_agent_rewards import TestAgentRewards
from test_agent_vision import TestAgentVision
from test_rllib_environment import TestRLlibEnvironmentIntegration
from test_rllib_hyperparameters import TestRLlibHyperparameterTuning
from test_rllib_training import TestRLlibTraining

# Import RLlib test classes
from test_rllib_wrapper import TestRLlibWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def create_test_suite():
    """Create a test suite combining all tests."""
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        # Original environment tests
        TestAgentCreation,
        TestAgentCommunication,
        TestAgentMovement,
        TestAgentVision,
        TestAgentRewards,
        # RLlib integration tests
        TestRLlibWrapper,
        TestRLlibTraining,
        TestRLlibHyperparameterTuning,
        TestRLlibEnvironmentIntegration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


def create_agent_test_suite():
    """Create a test suite with only agent tests (original functionality)."""
    suite = unittest.TestSuite()

    # Add only original agent test classes
    test_classes = [
        TestAgentCreation,
        TestAgentCommunication,
        TestAgentMovement,
        TestAgentVision,
        TestAgentRewards,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


def create_rllib_test_suite():
    """Create a test suite with only RLlib tests."""
    suite = unittest.TestSuite()

    # Add only RLlib test classes
    test_classes = [
        TestRLlibWrapper,
        TestRLlibTraining,
        TestRLlibHyperparameterTuning,
        TestRLlibEnvironmentIntegration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


def run_all_tests():
    """Run all tests."""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # logger.info summary
    logger.info("")
    logger.info("%s", "="*50)
    logger.info("Tests run: %d", result.testsRun)
    logger.info("Failures: %d", len(result.failures))
    logger.info("Errors: %d", len(result.errors))
    logger.info(
        "Success rate: %.1f%%",
        ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
    )
    logger.info("%s", "="*50)

    return result.wasSuccessful()


def run_agent_tests():
    """Run only agent tests."""
    suite = create_agent_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # logger.info summary
    logger.info("\n%s", "="*50)
    logger.info("AGENT TESTS SUMMARY")
    logger.info("Tests run: %d", result.testsRun)
    logger.info("Failures: %d", len(result.failures))
    logger.info("Errors: %d", len(result.errors))
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
    logger.info("Success rate: %.1f%%", success_rate)
    logger.info("%s", "="*50)

    return result.wasSuccessful()


def run_rllib_tests():
    """Run only RLlib tests."""
    suite = create_rllib_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # logger.info summary
    logger.info("\n%s", "="*50)
    logger.info("RLLIB TESTS SUMMARY")
    logger.info("Tests run: %d", result.testsRun)
    logger.info("Failures: %d", len(result.failures))
    logger.info("Errors: %d", len(result.errors))
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
    logger.info("Success rate: %.1f%%", success_rate)
    logger.info("%s", "="*50)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "agent":
            run_agent_tests()
        elif sys.argv[1] == "rllib":
            run_rllib_tests()
        else:
            logger.info("Usage: python run_tests.py [agent|rllib]")
            logger.info("  agent: Run only agent tests")
            logger.info("  rllib: Run only RLlib tests")
            logger.info("  (no args): Run all tests")
            run_all_tests()
    else:
        run_all_tests()
