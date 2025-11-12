from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPixmap
import sys, datetime
import serial.tools.list_ports
import time, os
import pandas as pd
from scipy.signal import argrelextrema, medfilt
import numpy as np

class SerialWorker(QThread):
    """Поток для выполнения операций с последовательным портом"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    data_ready = pyqtSignal(bytes)
    
    def __init__(self, port, command, timeout, read_size=None):
        super().__init__()
        self.port = port
        self.command = command
        self.timeout = timeout
        self.read_size = read_size
        
    def run(self):
        try:
            with serial.Serial(self.port, baudrate=921600, bytesize=8, 
                             stopbits=1, timeout=self.timeout) as serial_conn:
                serial_conn.write(self.command.encode())
                
                if self.read_size:
                    raw_data = serial_conn.read(self.read_size)
                    self.data_ready.emit(raw_data)
                else:
                    time.sleep(self.timeout)
                    
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Ошибка последовательного порта: {str(e)}")

class DataProcessor:
    """Класс для обработки данных"""
    
    @staticmethod
    def process_raw_data(raw_data: bytes) -> pd.DataFrame:
        """Перевод байтовой строки в числа"""
        data = []
        for num in range(0, 2097152, 2):
            hi_byte = raw_data[num]
            hi_byte = hi_byte if hi_byte < 128 else hi_byte - 256
            lo_byte = raw_data[num + 1]    
            data.append(hi_byte * 256 + lo_byte)

        encoder = []
        for num in range(2097152, len(raw_data) - 1, 2):
            hi_byte = raw_data[num]
            hi_byte = hi_byte if hi_byte < 128 else hi_byte - 256
            lo_byte = raw_data[num + 1]    
            encoder.append(hi_byte * 256 + lo_byte)
            
        return pd.DataFrame({'encoder': encoder, 'data': data})
    
    @staticmethod
    def apply_median_filter(df: pd.DataFrame, window_size:int=3) -> pd.DataFrame:
        """Применение медианного фильтра для удаления дельта-выбросов"""
        df_filtered = pd.DataFrame(columns=['encoder', 'data'])
        df_filtered['data'] = medfilt(df['data'], kernel_size=window_size)[window_size*2:-window_size*2]
        df_filtered['encoder'] = medfilt(df['encoder'], kernel_size=window_size)[window_size*2:-window_size*2]
        return df_filtered

    @staticmethod
    def integrate_df(df_filtered: pd.DataFrame, step:int=25) -> pd.DataFrame:
        """Полное интегрирование данных - основной метод"""
        bins = range(0, len(df_filtered) + step, step)

        split_points = df_filtered.index[(df_filtered['encoder'].shift(1) == 9999) & (df_filtered['encoder'] == 0)]
        split_points = [0] + split_points.tolist() + [len(df_filtered)]

        datasets = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            dataset_part = df_filtered.iloc[start_idx:end_idx].copy()
            dataset_part.encoder = dataset_part.encoder + 10000 * i

            dataset_part['bin'] = pd.cut(dataset_part['encoder'], bins=bins, right=False, labels=bins[:-1])
            grouped = dataset_part.groupby('bin', observed=True)['data'].sum()
            datasets.append(grouped)    

        df_encoder = pd.concat(datasets).reset_index()
        df_encoder['integrated_data'] = -1*df_encoder['data'].cumsum()/32767*2.5*(10**-5)

        x = df_encoder.integrated_data.index.values
        y = df_encoder.integrated_data.values

        coefficients = np.polyfit(x, y, 1)
        trend = np.polyval(coefficients, x)

        df_encoder['data'] = y - trend
        df_encoder['deg'] = df_encoder.index/10000/step*360

        df = df_encoder.reindex(columns=['deg', 'data'])
        
        local_maxima = argrelextrema(df.data.values, np.greater, order=100)[0]
        local_minima = argrelextrema(df.data.values, np.less, order=100)[0]

        df['is_local_max'] = False
        df['is_local_min'] = False

        df.loc[local_maxima, 'is_local_max'] = True
        df.loc[local_minima, 'is_local_min'] = True

        return df

    @staticmethod
    def get_amplitude(df: pd.DataFrame) -> tuple:
        """Вычисление амплитуды"""
        maxima = df[df['is_local_max']]
        minima = df[df['is_local_min']]

        mean_max = maxima['data'].mean()
        mean_min = minima['data'].mean()

        amplitude = (mean_max - mean_min)/2

        std_max = maxima['data'].std(ddof=1)
        std_error_max = std_max / np.sqrt(len(maxima))

        std_min = minima['data'].std(ddof=1)
        std_error_min = std_min / np.sqrt(len(minima))

        absolute_error = np.sqrt(std_error_max**2 + std_error_min**2) / 2
        relative_error = absolute_error / amplitude * 100

        return (amplitude, absolute_error, relative_error)

class MotorController:
    """Класс для управления мотором"""
    
    @staticmethod
    def run_motor(port, revolutions=20, distance=111, speed=100):
        """Запуск мотора на определенное количество оборотов"""
        try:
            with serial.Serial(port, baudrate=57600, bytesize=8, 
                             parity='N', stopbits=1, timeout=0) as serial_conn:
                
                command = f'ON\rMOVE L(-{int(revolutions * distance)})F({int(speed)})\rOFF\r'
                return serial_conn.write(command.encode("utf-8"))

        except Exception as e:
            print(f"Ошибка управления мотором: {e}")
            return 0

class MeasurementManager:
    """Менеджер управления измерениями"""
    
    def __init__(self):
        self.measurements = []
        self.current_measurement = 0
        self.total_measurements = 3
        self.current_measurement_data = None  # Данные текущего измерения
    
    def start_measurement_cycle(self):
        """Начать цикл измерений"""
        self.measurements = []
        self.current_measurement = 0
        self.current_measurement_data = None
    
    def add_measurement(self, amplitude, absolute_error, relative_error):
        """Добавить результат измерения"""
        self.measurements.append({
            'amplitude': amplitude,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        })
        self.current_measurement += 1
    
    def save_current_measurement_data(self, amplitude, absolute_error, relative_error):
        """Сохранить данные текущего измерения (для возможного повтора)"""
        self.current_measurement_data = {
            'amplitude': amplitude,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }
    
    def confirm_current_measurement(self):
        """Подтвердить текущее измерение и перейти к следующему"""
        if self.current_measurement_data:
            self.measurements.append(self.current_measurement_data.copy())
            self.current_measurement += 1
            self.current_measurement_data = None
    
    def is_complete(self):
        """Проверить завершение всех измерений"""
        return self.current_measurement >= self.total_measurements
    
    def get_current_measurement_number(self):
        """Получить номер текущего измерения"""
        return self.current_measurement + 1
    
    def get_final_result(self):
        """Получить финальный результат по формуле: sqrt((m1)**2 + (m2)**2 + (m3)**2)/2"""
        if len(self.measurements) != 3:
            return None
        
        # Извлекаем амплитуды из всех трех измерений
        amplitudes = [m['amplitude'] for m in self.measurements]
        
        # Вычисляем финальный результат по формуле
        sum_of_squares = sum(amp**2 for amp in amplitudes) / 2
        final_amplitude = np.sqrt(sum_of_squares)
        
        # Вычисляем погрешность финального результата
        # Погрешность по формуле для функции f = sqrt(a^2 + b^2 + c^2)/2
        absolute_errors = [m['absolute_error'] for m in self.measurements]
        
        # Частные производные: df/dai = (ai) / (2 * sqrt(a1^2 + a2^2 + a3^2))
        denominator = 2 * np.sqrt(sum_of_squares)
        partial_derivatives = [amp / denominator for amp in amplitudes]
        
        # Погрешность финального результата
        final_absolute_error = np.sqrt(
            sum((partial_derivatives[i] * absolute_errors[i])**2 for i in range(3))
        )
        
        # Относительная погрешность
        final_relative_error = final_absolute_error / final_amplitude * 100
        
        return (final_amplitude, final_absolute_error, final_relative_error)
    
    def get_individual_results(self):
        """Получить результаты отдельных измерений"""
        return self.measurements.copy()

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        
        # Загрузка UI из файла
        loadUi("HelmUI_modern.ui", self)
        
        self.serial_worker = None
        self.df = None
        self.data_processor = DataProcessor()
        self.motor_controller = MotorController()
        self.measurement_manager = MeasurementManager()
        
        self.init_ui()
        self.init_graph()
        
    def init_ui(self):
        """Инициализация интерфейса"""
        # Загрузка логотипа
        self.load_logo()
        
        # Привязка сигналов
        self.refreshPortsBtn.clicked.connect(self.refresh_ports)
        self.pBtn_GetData.clicked.connect(self.start_measurement_cycle)
        
        # Список меток измерений для удобного доступа
        self.measurement_labels = [
            self.measurement1Label,
            self.measurement2Label, 
            self.measurement3Label
        ]
        
        # Инициализация портов
        self.refresh_ports()
        
        # Скрываем прогресс бар изначально
        self.progressBar.setVisible(False)
        
        self.update_buttons_state(True)
        self.show_status_message("Ready to work")

    def load_logo(self):
        """Загрузка логотипа"""
        try:
            logo_pixmap = QPixmap("AMTClogo.png")
            if not logo_pixmap.isNull():
                self.logoLabel.setPixmap(logo_pixmap.scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"Не удалось загрузить логотип: {e}")

    def refresh_ports(self):
        """Обновление списка доступных портов"""
        self.cBox_MotorPort.clear()
        self.cBox_SensorPort.clear()
        
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.cBox_MotorPort.addItem(port.device)
            self.cBox_SensorPort.addItem(port.device)
        
        if self.cBox_MotorPort.count() > 1:
            self.cBox_MotorPort.setCurrentIndex(1)
        if self.cBox_SensorPort.count() > 0:
            self.cBox_SensorPort.setCurrentIndex(0)

    def init_graph(self):
        """Инициализация графика"""
        self.chart = QChart()
        self.chart.setTheme(QChart.ChartThemeBlueIcy)
        self.chart.legend().setVisible(False)
        
        self.series = QLineSeries()
        self.chart.addSeries(self.series)
        self.chart.createDefaultAxes()
        
        self.chart.axisX().setLabelFormat("%.1f")
        self.chart.axisY().setLabelFormat("%.2e")
        
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chartLayout.addWidget(self.chart_view)
   
    def update_buttons_state(self, enabled):
        """Обновление состояния кнопок"""
        self.pBtn_GetData.setEnabled(enabled)
        self.refreshPortsBtn.setEnabled(enabled)
                
    def set_wait_cursor(self, waiting):
        """Установка курсора ожидания"""
        if waiting:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
    
    def start_measurement_cycle(self):
        """Начать цикл из 3 измерений"""
        self.motor_port = self.cBox_MotorPort.currentText()
        self.sensor_port = self.cBox_SensorPort.currentText()
        
        if not self.motor_port or not self.sensor_port:
            QMessageBox.warning(self, "Error", "Select motor and sensor ports")
            return
        
        self.measurement_manager.start_measurement_cycle()
        self.update_buttons_state(False)
        self.set_wait_cursor(True)
        self.progressBar.setVisible(True)
        self.progressBar.setMaximum(3)
        self.progressBar.setValue(0)
        
        # Сброс результатов
        for label in self.measurement_labels:
            label.setText("---")
        self.finalResultLabel.setText("Final Result: ---")
        
        self.show_status_message("Starting measurement cycle...")
        self.start_single_measurement()
    
    def start_single_measurement(self):
        """Запуск одиночного измерения"""
        current_meas = self.measurement_manager.get_current_measurement_number()
        self.progressLabel.setText(f"Measurement {current_meas}/3 in progress...")
        self.progressBar.setValue(current_meas - 1)
        
        try:
            self.motor_controller.run_motor(self.motor_port)
        except Exception as e:
            self.on_serial_error(f"Motor start error: {str(e)}")
            return
        
        QTimer.singleShot(500, self.read_sensor)
    
    def read_sensor(self):
        """Чтение датчика после запуска мотора"""
        self.serial_worker = SerialWorker(self.sensor_port, 'R', 11)
        self.serial_worker.finished.connect(self.on_read_sensor_finished)
        self.serial_worker.error.connect(self.on_serial_error)
        self.serial_worker.start()
    
    def on_read_sensor_finished(self):
        """Обработка завершения чтения датчика"""
        self.show_status_message("Sensor read, getting data...")
        QTimer.singleShot(500, self.get_data)
    
    def get_data(self):
        """Получение данных после чтения датчика"""
        self.serial_worker = SerialWorker(self.sensor_port, 'S', 47, 4194305)
        self.serial_worker.data_ready.connect(self.on_data_received)
        self.serial_worker.error.connect(self.on_serial_error)
        self.serial_worker.finished.connect(self.on_get_data_finished)
        self.serial_worker.start()
    
    def on_data_received(self, raw_data):
        """Обработка полученных данных"""
        try:
            df_raw = self.data_processor.process_raw_data(raw_data)
            df_filtered = self.data_processor.apply_median_filter(df_raw, window_size=3)
            self.df = self.data_processor.integrate_df(df_filtered)

            self.update_graph(self.df)

            amplitude, absolute_error, relative_error = self.data_processor.get_amplitude(self.df)
            
            # Сохраняем данные текущего измерения (для возможного повтора)
            self.measurement_manager.save_current_measurement_data(amplitude, absolute_error, relative_error)
            
            # Обновляем интерфейс с результатом текущего измерения
            current_idx = self.measurement_manager.current_measurement
            if current_idx < len(self.measurement_labels):
                self.measurement_labels[current_idx].setText(
                    f"Measurement {current_idx + 1}: {amplitude:.5f} ± {absolute_error:.5f} ({relative_error:.2f}%)"
                )
            
            self.show_status_message(f'Measurement {current_idx + 1}/3 completed!')

        except Exception as e:
            self.on_serial_error(f"Data processing error: {str(e)}")
    
    def on_get_data_finished(self):
        """Обработка завершения получения данных"""
        # После завершения измерения спрашиваем пользователя
        self.ask_measurement_action()
    
    def ask_measurement_action(self):
        """Спросить пользователя о дальнейших действиях"""
        current_meas_num = self.measurement_manager.get_current_measurement_number()
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Measurement Complete")
        msg.setText(f"Measurement {current_meas_num}/3 completed successfully!\n\nWhat would you like to do?")
        msg.setIcon(QMessageBox.Question)
        
        # Добавляем кнопки
        repeat_btn = msg.addButton("Repeat Measurement", QMessageBox.ActionRole)
        next_btn = msg.addButton("Next Measurement", QMessageBox.AcceptRole)
        cancel_btn = msg.addButton("Cancel Cycle", QMessageBox.RejectRole)
        
        msg.setDefaultButton(next_btn)
        
        msg.exec_()
        
        clicked_button = msg.clickedButton()
        
        if clicked_button == repeat_btn:
            # Повторить измерение
            self.show_status_message(f"Repeating measurement {current_meas_num}...")
            QTimer.singleShot(1000, self.start_single_measurement)
            
        elif clicked_button == next_btn:
            # Подтвердить текущее измерение и перейти к следующему
            self.measurement_manager.confirm_current_measurement()
            
            if self.measurement_manager.is_complete():
                # Все измерения завершены
                self.measurement_cycle_complete()
            else:
                # Перейти к следующему измерению
                self.show_status_message(f"Starting next measurement...")
                QTimer.singleShot(1000, self.start_single_measurement)
                
        elif clicked_button == cancel_btn:
            # Отменить цикл измерений
            self.measurement_cycle_complete(aborted=True)
    
    def measurement_cycle_complete(self, aborted=False):
        """Завершение цикла измерений"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
        self.progressBar.setVisible(False)
        
        if aborted:
            self.progressLabel.setText("Measurement cycle aborted")
            self.show_status_message("Measurement cycle aborted by user")
        else:
            self.progressLabel.setText("Measurement cycle completed")
            
            # Показываем финальный результат по новой формуле
            final_results = self.measurement_manager.get_final_result()
            if final_results:
                amplitude, absolute_error, relative_error = final_results
                self.finalResultLabel.setText(
                    f"Final Result: {amplitude:.5f} ± {absolute_error:.5f} ({relative_error:.2f}%)"
                )
                
                # Показываем детали расчетов
                individual_results = self.measurement_manager.get_individual_results()
                details = "Calculation details:\n"
                for i, result in enumerate(individual_results, 1):
                    details += f"M{i} = {result['amplitude']:.5f}\n"
                details += f"Final = √({individual_results[0]['amplitude']:.5f}² + {individual_results[1]['amplitude']:.5f}² + {individual_results[2]['amplitude']:.5f}²) / 2"
                
                self.show_status_message(f"Measurement cycle completed! Final amplitude: {amplitude:.5f}")
                
                # Показываем диалог с деталями расчета
                QMessageBox.information(self, "Measurement Complete", 
                                      f"Measurement cycle completed successfully!\n\n"
                                      f"{details}\n\n"
                                      f"Final result: {amplitude:.5f} ± {absolute_error:.5f} ({relative_error:.2f}%)")
    
    def on_serial_error(self, error_msg):
        """Обработка ошибок последовательного порта"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
        self.progressBar.setVisible(False)
        QMessageBox.critical(self, "Error", error_msg)
        print(f"Error: {error_msg}")

    def update_graph(self, df):
        """Обновление графика текущими данными"""
        self.chart.removeAllSeries()
        data_series = QLineSeries()
        
        for row in df.itertuples():
            data_series.append(row.deg, row.data)
        
        self.chart.addSeries(data_series)
        self.chart.createDefaultAxes()
        self.chart.axisX().setLabelFormat("%.1f")
        self.chart.axisY().setLabelFormat("%.2e")
    
    def show_status_message(self, message, timeout=5000):
        """Показать сообщение в статус баре"""
        self.statusBar.showMessage(message, timeout)

if __name__ == '__main__':
    app = QApplication([])
    
    def exception_handler(exctype, value, traceback):
        print(f"Unhandled exception: {exctype.__name__}: {value}")
        QMessageBox.critical(None, "Error", f"An error occurred:\n{value}")
    
    sys.excepthook = exception_handler
    
    window = MainUI()
    window.show()
    
    sys.exit(app.exec_())