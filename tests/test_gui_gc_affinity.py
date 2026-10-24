import gc
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import threading
import unittest
from unittest import mock

from PySide6 import QtCore, QtWidgets
from squeakview.apps.operator.gui.gc_affinity import GuiGarbageCollector, install_gui_gc


class GuiGarbageCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.was_enabled = gc.isenabled()
        self.guards = []

    def tearDown(self):
        for guard in self.guards:
            guard.close()
            guard.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        (gc.enable if self.was_enabled else gc.disable)()

    def test_disables_automatic_collection_and_restores_previous_state(self):
        for was_enabled in (True, False):
            with self.subTest(was_enabled=was_enabled):
                (gc.enable if was_enabled else gc.disable)()
                guard = GuiGarbageCollector(self.app)
                self.guards.append(guard)
                self.assertFalse(gc.isenabled())
                self.assertTrue(guard._timer.isActive())
                guard.close()
                self.assertEqual(gc.isenabled(), was_enabled)
                self.assertFalse(guard._timer.isActive())
                guard.close()

    def test_collection_occurs_on_gui_thread_and_worker_cannot_collect(self):
        guard = GuiGarbageCollector(self.app)
        self.guards.append(guard)
        calls, errors = [], []
        with mock.patch('squeakview.apps.operator.gui.gc_affinity.gc.collect',
                        side_effect=lambda: calls.append(threading.get_ident())):
            guard.collect()
            def worker():
                try: guard.collect()
                except RuntimeError as exc: errors.append(str(exc))
            thread = threading.Thread(target=worker)
            thread.start(); thread.join()
        self.assertEqual(calls, [threading.get_ident()])
        self.assertIn('Qt thread', errors[0])

    def test_install_is_idempotent_and_owned_by_application(self):
        first = install_gui_gc(self.app)
        self.guards.append(first)
        second = install_gui_gc(self.app)
        self.assertIs(first, second)
        self.assertIs(first.parent(), self.app)
        self.assertGreaterEqual(first._timer.interval(), 1000)
