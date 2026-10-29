"""The test command must fail when verification did not actually succeed."""
import importlib.util
import io
from pathlib import Path
import threading
import unittest
from unittest import mock

from tests.support.threads import CheckedThread

spec = importlib.util.spec_from_file_location('software_test_command', Path(__file__).parents[1] / 'scripts/test.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestRunnerTests(unittest.TestCase):
    def test_empty_discovery_is_a_failure(self):
        self.assertFalse(runner.run_python_suite(unittest.TestSuite(), stream=io.StringIO()))

    def test_uncaught_worker_assertion_cannot_produce_a_green_suite(self):
        class BrokenTest(unittest.TestCase):
            def runTest(self):
                def worker():
                    raise AssertionError('worker assertion')
                thread = threading.Thread(target=worker)
                thread.start()
                thread.join(1)
        stream = io.StringIO()
        self.assertFalse(runner.run_python_suite(unittest.TestSuite([BrokenTest()]), stream=stream))
        self.assertIn('worker assertion', stream.getvalue())

    def test_checked_thread_rethrows_in_owner(self):
        def worker():
            raise AssertionError('owner sees this')
        thread = CheckedThread(target=worker)
        thread.start()
        with self.assertRaisesRegex(AssertionError, 'owner sees this'):
            thread.join()

    def test_success_still_passes(self):
        self.assertTrue(runner.run_python_suite(unittest.TestSuite([unittest.FunctionTestCase(lambda: None)]), stream=io.StringIO()))

    def test_both_native_suites_reject_empty_ctest_discovery(self):
        with mock.patch.object(runner.subprocess, 'run') as run:
            runner.native_suites('/temporary-build')
        commands = [call.args[0] for call in run.call_args_list]
        ctests = [command for command in commands if command[0] == 'ctest']
        self.assertEqual(len(ctests), 2)
        self.assertTrue(all('--no-tests=error' in command for command in ctests))
        self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))
