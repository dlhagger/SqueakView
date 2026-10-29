"""Real Qt window and supervisor proxy, driven as an offscreen operator."""
import json
from pathlib import Path
import sys
import time
import traceback
from unittest import mock

from PySide6 import QtWidgets
from squeakview.apps.operator.gui.main_window import MainWindow
from squeakview.common import run_context


def main():
    output = Path(sys.argv[1])
    serial_enabled = len(sys.argv) > 2 and sys.argv[2] == 'v2-free'
    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    window = None
    result = {}
    messages = []
    def dialog(_parent, title, message, *_args, **_kwargs):
        messages.append(f'{title}: {message}')
        return QtWidgets.QMessageBox.Ok
    dialogs = [mock.patch.object(QtWidgets.QMessageBox, kind, side_effect=dialog)
               for kind in ('warning', 'critical')]
    for patcher in dialogs:
        patcher.start()
    def wait(predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            app.processEvents()
            if messages:
                raise RuntimeError('\n'.join(messages))
            if predicate():
                return
            time.sleep(.01)
        raise TimeoutError('operator transition did not finish')
    try:
        with mock.patch.object(MainWindow, '_show_launch_dialog', return_value=True):
            window = MainWindow()
        window.show()
        wait(lambda: window._preview_window_id is not None)
        config = dict(window._config_data, serial_enabled=serial_enabled, serial_port='/dev/software-controller', trigger_on=False, inference_enabled=False,
                      ds_cfg='', width=64, height=64, mouse_id='mouse-A', experiment_name='software-test')
        window._apply_config(config)
        panel = window.bottle_panel
        panel.left_fluid_combo.setCurrentText('water')
        panel.right_fluid_combo.setCurrentText('sucrose')
        panel.left_initial_weight_edit.setText('30')
        panel.right_initial_weight_edit.setText('32')
        panel.mouse_weight_edit.setText('25')
        window.run_btn.click()
        wait(lambda: window._recording_active and not window._start_in_progress)
        root = window.backend.current_snapshot.run_dir
        assert root is not None
        initial = run_context.bottle_setup(root)
        assert initial['mouse_weight_g'] == 25, initial
        assert initial['sides']['left']['initial_weight_g'] == 30, initial
        assert initial['sides']['right']['fluid'] == 'sucrose', initial
        assert not panel.left_final_weight_edit.isEnabled()
        window.stop_btn.click()
        wait(lambda: window._stop_in_progress and panel.left_final_weight_edit.isEnabled())
        assert window.stop_overlay.isHidden()
        assert not window.run_btn.isEnabled()
        assert not panel.save_button.isEnabled()
        panel.left_final_weight_edit.setText('29')
        panel.right_final_weight_edit.setText('31')
        (output.parent/'release-validation').touch()
        wait(lambda: not window._stop_in_progress and not window._recording_active)
        status = run_context.read_json(run_context.status_path(root))
        assert status['overall_validation_passed'], status
        assert panel.left_initial_weight_edit.text() in ('30', '30.0')
        assert panel.mouse_weight_edit.text() in ('25', '25.0')
        assert panel.left_final_weight_edit.text() == '29'
        assert panel.right_final_weight_edit.text() == '31'
        panel.save_button.click()
        saved = run_context.bottle_setup(root)
        assert saved['sides']['left']['final_weight_g'] == 29, saved
        assert saved['sides']['right']['final_weight_g'] == 31, saved
        assert 'saved' in panel.save_button.text().lower()
        panel.mouse_weight_edit.clear()
        panel.save_button.click()
        assert run_context.bottle_setup(root)['mouse_weight_g'] is None
        panel.mouse_weight_edit.setText('26')
        panel.save_button.click()
        assert run_context.bottle_setup(root)['mouse_weight_g'] == 26
        # A second subject must get its own measurements and fresh finals;
        # editing the next form must not rewrite the previous run.
        window._apply_config(dict(config, mouse_id='mouse-B'))
        panel.left_initial_weight_edit.setText('40')
        panel.right_initial_weight_edit.setText('42')
        panel.mouse_weight_edit.setText('27')
        window.run_btn.click()
        wait(lambda: window._recording_active and not window._start_in_progress)
        next_root = window.backend.current_snapshot.run_dir
        assert next_root != root
        assert panel.left_final_weight_edit.text() == ''
        assert run_context.bottle_setup(next_root)['mouse_weight_g'] == 27
        assert run_context.bottle_setup(next_root)['sides']['left']['initial_weight_g'] == 40
        window.stop_btn.click()
        wait(lambda: not window._stop_in_progress and not window._recording_active)
        assert run_context.read_json(run_context.status_path(next_root))['overall_validation_passed']
        panel.left_final_weight_edit.setText('39')
        panel.right_final_weight_edit.setText('41')
        panel.save_button.click()
        assert run_context.bottle_setup(next_root)['sides']['left']['final_weight_g'] == 39
        assert run_context.bottle_setup(root)['mouse_weight_g'] == 26
        assert run_context.bottle_setup(root)['sides']['left']['final_weight_g'] == 29
        result = {'passed': True, 'run_dir': str(root), 'next_run_dir': str(next_root)}
    except BaseException:
        result = {'passed': False, 'error': traceback.format_exc()}
    finally:
        output.write_text(json.dumps(result))
        if window is not None:
            window.backend.shutdown()
            window.close()
            app.processEvents()
        for patcher in reversed(dialogs):
            patcher.stop()
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
