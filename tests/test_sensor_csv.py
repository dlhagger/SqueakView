from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from squeakview.common.serial import SerialHandle, SERIAL_HEADER
from squeakview.common.run_layout import initialize_layout
from squeakview.common.controller_protocol_v2 import Frame, MessageType
from squeakview.common.sensor_csv import controller_timestamp


class SensorCsvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        initialize_layout(self.root)
        self.logs = []
        self.handle = SerialHandle('/dev/test',115200,self.logs.append)
        self.addCleanup(self.handle.close)

    def open(self):
        self.assertTrue(self.handle.set_csv_path(self.root))

    def rows(self):
        if self.handle._csv_file:
            self.handle._csv_file.flush()
        with (self.root/'sensors.csv').open(newline='') as source:
            return list(csv.DictReader(source))

    def publish(self, text, kind=MessageType.EVENT, timestamp=1000):
        frame = Frame(int(kind),0,12,9,1,timestamp,text.encode(),b'')
        self.handle._publish_v2_frame(frame,2000000,3000000)

    def test_all_startup_and_protocol_messages_stay_in_the_same_csv(self):
        self.open()
        lines=['RTC_TYPE,DS3231','Debounce DT=0 DR=0','TTH/RTH electrode 1 = 2/0',
               'SYSTEM_START,1793045604101673,15722733','MOUSEHOUSE_BUILD,Oct 23 2026,15:31:03',
               'ACK_PROTO,2']
        for line in lines:
            self.handle._write_csv_line(line)
        self.handle.log_marker('PROTOCOL_V2_NEGOTIATED')
        rows=self.rows()
        self.assertEqual([r['rawLine'] for r in rows],lines+['MARKER,PROTOCOL_V2_NEGOTIATED'])
        self.assertTrue(all(r['hostUnixNs'] and r['hostMonotonicNs'] for r in rows))
        self.assertEqual(rows[0]['eventType'],'RTC_TYPE')
        self.assertEqual(rows[0]['unixTime'],'')
        self.assertEqual(rows[0]['rp2040Time'],'')
        self.assertEqual(rows[0]['reason'],'DS3231')
        self.assertEqual(rows[1]['eventType'],'CONTROLLER_LOG')
        self.assertEqual(rows[1]['reason'],lines[1])
        self.assertEqual(rows[4]['unixTime'],'')
        self.assertEqual(rows[4]['reason'],'Oct 23 2026,15:31:03')
        self.assertEqual(rows[5]['context'],'CONTROL')
        self.assertFalse((self.root/'_metadata/diagnostics/controller_messages.csv').exists())
        with (self.root/'sensors.csv').open(newline='') as source:
            self.assertTrue(all(len(row)==len(SERIAL_HEADER) for row in csv.reader(source)))

    def test_clock_sync_uses_controller_midpoint_not_host_echo(self):
        self.open()
        text='CLOCK_SYNC,0,1793045604264983204,15850150,15850238,1793045604229180,RTC_VALID'
        self.publish(text,MessageType.COMMAND_RESULT,timestamp=15850260)
        row=self.rows()[0]
        self.assertEqual(row['controller_time_us'],'15850194')
        self.assertEqual(row['rp2040Time'],'15850194')
        self.assertEqual(row['unixTime'],'1793045604229136')
        self.assertEqual(row['controller_time_method'],'clock_sync_midpoint')
        self.assertEqual(row['hostUnixNs'],'2000000')
        self.assertEqual(row['side'],'')
        self.assertEqual(row['rawLine'],text)

    def test_schedule_and_camera_records_have_event_specific_columns(self):
        self.open()
        self.publish('ACTIVE_POKE_SCHEDULE,123456,local_day=20752,side=RIGHT',timestamp=900)
        self.publish('CAMERA_EPOCH,count=1,timestamp_us=1000,period_us=33333,pulse_us=10,health=0x3,queue=0/8,suppressed=0,reason=Start',MessageType.CAMERA_EPOCH,timestamp=1100)
        schedule,camera=self.rows()
        self.assertEqual(schedule['side'],'RIGHT')
        self.assertEqual(schedule['rp2040Time'],'900')
        self.assertEqual(schedule['duration'],'')
        self.assertEqual(camera['controller_time_us'],'1000')
        self.assertEqual(camera['controller_time_method'],'exact_anchor')
        self.assertEqual(camera['count'],'1')
        self.assertEqual(camera['rp2040Time'],'1000')
        self.assertEqual(camera['unixTime'],'')

    def test_standard_events_and_ack_events_preserve_existing_fields(self):
        self.open()
        self.publish('POKE_START,123456,321,L,1,10,20,30,Eligible,nan')
        self.publish('ACK_START_BEHAVIOR,123456,800,nan,0,0,69420,69420,Eligible,nan')
        first,ack=self.rows()
        self.assertEqual(first['side'],'L')
        self.assertEqual(first['duration'],'10')
        self.assertEqual(first['controller_time_us'],'321')
        self.assertEqual(ack['controller_time_us'],'800')
        self.assertEqual(ack['eventType'],'ACK_START_BEHAVIOR')

    def test_unknown_message_keeps_payload_and_frame_timestamp(self):
        self.open()
        self.publish('CUSTOM_ACTIVITY,mode=active',timestamp=1234)
        row=self.rows()[0]
        self.assertEqual(row['eventType'],'CUSTOM_ACTIVITY')
        self.assertEqual(row['rp2040Time'],'1234')
        self.assertEqual(row['reason'],'mode=active')
        self.assertEqual(row['controller_time_method'],'direct_record')

    def test_adopted_temp_file_keeps_all_rows_and_formats_setup_text(self):
        self.handle._open_temp_csv()
        self.handle._write_csv_line('RTC_TYPE,DS3231')
        self.handle._write_csv_line('POKE_START,123,321,L,1')
        self.open()
        rows=self.rows()
        self.assertEqual([r['eventType'] for r in rows],['RTC_TYPE','POKE_START'])
        self.assertEqual(rows[0]['unixTime'],'')
        self.assertEqual(rows[0]['reason'],'DS3231')
        self.handle.log_marker('START_SENT')
        self.assertEqual(self.rows()[-1]['reason'],'START_SENT')

    def test_rows_buffered_before_any_file_is_open_are_formatted(self):
        self.handle._write_csv_line('RTC_TYPE,DS3231')
        self.handle._write_csv_line('POKE_START,123,321,L,1')
        self.open()
        rows=self.rows()
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['reason'],'DS3231')
        self.assertEqual(rows[0]['unixTime'],'')

    def test_clock_status_and_bottle_state_have_correct_native_timestamps(self):
        self.open()
        self.handle._write_csv_line('CLOCK_STATUS,VALID,1793045604000000,15621057,502,Boot')
        self.publish('BOTTLE_STATE,1,none,ACTIVE,0,0,22906848,READY',MessageType.COMMAND_RESULT)
        clock,bottle=self.rows()
        self.assertEqual(clock['unixTime'],'1793045604000000')
        self.assertEqual(clock['rp2040Time'],'15621057')
        self.assertEqual(clock['controller_time_method'],'rtc_anchor')
        self.assertEqual(bottle['rp2040Time'],'22906848')
        self.assertEqual(bottle['context'],'BOTTLE_SWAP')
        self.assertEqual(bottle['unixTime'],'')
        value,_=controller_timestamp('ACK_RESEND_EVENTS,12,345'.split(','))
        self.assertEqual(value,'')

    def test_terminal_uses_same_ten_columns_as_sensor_csv(self):
        self.open()
        self.handle._emit_serial_logs = True
        self.handle._handle_semantic_line('RTC_TYPE,DS3231',received_ns=1000)
        payload = self.logs[-1].split('【SER】',1)[1].strip()
        cells = next(csv.reader([payload]))
        row = self.rows()[0]
        self.assertEqual(cells,[row[name] for name in SERIAL_HEADER[:10]])
        self.assertEqual(len(cells),10)

    def test_terminal_clock_and_event_rows_share_csv_native_times(self):
        self.open()
        self.handle._emit_serial_logs = True
        self.publish('CLOCK_SYNC,0,1793045604264983204,15850150,15850238,1793045604229180,RTC_VALID',MessageType.COMMAND_RESULT)
        payload = self.logs[-1].split('【SER】',1)[1].strip()
        self.assertEqual(next(csv.reader([payload])),[self.rows()[0][name] for name in SERIAL_HEADER[:10]])

    def test_command_log_is_formatted_without_changing_wire_bytes(self):
        from unittest.mock import Mock
        self.open()
        port = Mock(is_open=True)
        port.write.side_effect = len
        port.close.side_effect = lambda:setattr(port,'is_open',False)
        self.handle.ser = port
        self.handle._queue_line('START,30', priority=10, wait=True)
        port.write.assert_called_once_with(b'START,30\n')
        payload = next(line for line in self.logs if '【SER→】' in line).split('【SER→】',1)[1].strip()
        cells = next(csv.reader([payload]))
        self.assertEqual(len(cells),10)
        self.assertEqual(cells[0],'HOST_COMMAND')
        self.assertEqual(cells[8:],["TX","START,30"])

    def test_status_log_and_quoted_reasons_are_single_common_records(self):
        from squeakview.common.sensor_csv import format_controller_log
        from squeakview.common.dashboard import DashboardEvent
        result = format_controller_log('[16:00:00] [SER] WARN: first line\nsecond line, with comma')
        self.assertNotIn('\n',result)
        cells = next(csv.reader([result.split('[SER]',1)[1].strip()]))
        self.assertEqual(len(cells),10)
        self.assertEqual(cells[0],'HOST_LOG')
        self.assertEqual(cells[8],'HOST')
        self.assertIn('second line, with comma',cells[9])
        text = 'CONTROLLER_LOG,,,,,,,,DIAGNOSTIC,"a, b and ""quoted"" text"'
        event = DashboardEvent.parse(text)
        self.assertEqual(event.reason,'a, b and "quoted" text')

    def test_quoted_controller_reason_matches_terminal_and_file(self):
        self.open()
        self.handle._emit_serial_logs = True
        raw = 'TASK_INFO,123456,321,L,1,0,0,0,Eligible,"a, b and ""quoted"" text"'
        self.handle._handle_semantic_line(raw,received_ns=1000)
        row = self.rows()[0]
        self.assertEqual(row['reason'],'a, b and "quoted" text')
        self.assertEqual(row['rawLine'],raw)
        payload = self.logs[-1].split('【SER】',1)[1].strip()
        self.assertEqual(next(csv.reader([payload])),[row[name] for name in SERIAL_HEADER[:10]])

    def test_shared_logger_preserves_all_existing_sensor_fields(self):
        from squeakview.common.sensor_csv import EVENT_FIELDS, make_event_record
        fields = ['POKE_START','123456','321','L','001','69420','20','30','Eligible','nan']
        raw = ','.join(fields)
        record = make_event_record(fields,raw,host_unix_ns=1000,host_monotonic_ns=2000)
        self.assertEqual([record[name] for name in EVENT_FIELDS],fields)
        self.assertEqual(record['rawLine'],raw)
        self.assertEqual(record['hostUnixNs'],1000)
        self.assertEqual(record['hostMonotonicNs'],2000)

if __name__=='__main__':
    unittest.main()
