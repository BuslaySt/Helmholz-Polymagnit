from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QMessageBox
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter
import sys, datetime
import serial.tools.list_ports
import time, os
import pandas as pd
from scipy import signal
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
                    # Для операций только записи ждем завершения
                    time.sleep(self.timeout)
                    
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Ошибка последовательного порта: {str(e)}")


class DataProcessor:
    """Класс для обработки данных"""
    
    @staticmethod
    def process_raw_data(raw_data):
        """Обработка сырых данных"""
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
    def apply_median_filter(df, window_size=3):
        """Применение медианного фильтра"""
        df_filtered = pd.DataFrame(columns=['encoder', 'data'])
        df_filtered['data'] = signal.medfilt(df['data'], kernel_size=window_size)[window_size*2:-window_size*2]
        df_filtered['encoder'] = signal.medfilt(df['encoder'], kernel_size=window_size)[window_size*2:-window_size*2]
        return df_filtered
    
    @staticmethod
    def split_data_segments(df_filtered):
        """1. Разделение данных на сегменты по encoder"""
        # Находим точки разделения по encoder
        split_points = df_filtered.index[
            (df_filtered['encoder'].shift(1) == 9999) & 
            (df_filtered['encoder'] == 0)
        ]
        
        # Создаем сегменты данных
        segments = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            segment = df_filtered.iloc[start_idx:end_idx].copy()
            # Cуммирование данных по отсчётам encoder
            segments.append(
                segment.groupby('encoder', as_index=False)['data'].sum()
            )
        
        return segments
    
    @staticmethod
    def integrate_segment(segment, step=20):
        """2. Интегрирование одного сегмента данных"""
        bins = list(range(0, 10000, step))
        
        # Разделение и группировка
        segment['bin'] = pd.cut(
            segment['encoder'], bins=bins, right=False, labels=bins[:-1]
        )
        grouped = segment.groupby('bin', observed=False)['data'].sum().reset_index()
        
        # Линейное интегрирование
        integrated_data = grouped['data'].cumsum()
        
        return integrated_data
    
    @staticmethod
    def apply_linear_correction(integrated_data):
        """3. Линейная коррекция интегрированного сегмента"""
        difference = integrated_data.iloc[0] - integrated_data.iloc[-1]
        n = len(integrated_data)
        
        corrected_data = []
        for i, value in enumerate(integrated_data):
            correction = (difference * i) / (n - 1) if n > 1 else 0
            corrected_data.append(value + correction)
        
        return corrected_data
    
    @staticmethod
    def average_segments(segments_data, step=20):
        """4. Усреднение всех сегментов"""
        bins = list(range(0, 10000, step))
        df_integral = pd.DataFrame(index=range(len(bins) - 1))
        
        for i, segment_data in enumerate(segments_data):
            df_integral[f'integral_data_{i+1}'] = segment_data
        
        # Возвращаем усредненные данные по нескольким поворотам
        result_df = pd.DataFrame(columns=['encoder', 'data'], index=range(len(bins) - 1))
        result_df['encoder'] = range(len(bins) - 1)
        result_df['data'] = df_integral.mean(axis=1)
        
        return result_df
    
    @staticmethod
    def integrate_df(df_filtered, step=20):
        """Полное интегрирование данных - основной метод
        df_filtered - обработанные данные
        step - шаг интегрирования - 10000 / 20 = 500 градусов
        """
        # 1. Разделение на сегменты
        segments = DataProcessor.split_data_segments(df_filtered)
        
        # 2-3. Интегрирование и коррекция каждого оборота
        integrated_segments = []
        for segment in segments:
            integrated_data = DataProcessor.integrate_segment(segment, step)
            integrated_segments.append(integrated_data)
        
        # 4. Усреднение всех оборотов
        result_df = DataProcessor.average_segments(integrated_segments, step)
        
        return result_df


class MotorController:
    """Класс для управления мотором"""
    
    @staticmethod
    def run_motor(port, revolutions=20, distance=111, speed=100):
        """Запуск мотора на определенное количество оборотов"""
        try:
            with serial.Serial(port, baudrate=57600, bytesize=8, 
                             parity='N', stopbits=1, timeout=0) as serial_conn:
                
                command = f'ON\rMOVE L({int(revolutions * distance)})F({int(speed)})\rOFF\r'
                return serial_conn.write(command.encode("utf-8"))

        except Exception as e:
            print(f"Ошибка управления мотором: {e}")
            return 0


class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        loadUi("HelmUI.ui", self)
        
        self.serial_worker = None
        self.df = None
        self.df_processed = None  # Для хранения обработанных данных
        self.data_processor = DataProcessor()
        self.motor_controller = MotorController()
        
        self.init_ui()
        self.init_graph()
        
    def init_ui(self):
        """Инициализация интерфейса"""
        # Заполнение списков портов
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.cBox_MotorPort.addItem(port.device)
            self.cBox_SensorPort.addItem(port.device)
        self.cBox_MotorPort.setCurrentIndex(1)
        self.cBox_SensorPort.setCurrentIndex(0)

        # Привязка кнопок
        self.pBtn_Show.clicked.connect(self.show_data)
        self.pBtn_Save.clicked.connect(self.save_data)

        # self.pBtn_Motor.clicked.connect(self.run_motor)
        # self.pBtn_ReadSensor.clicked.connect(self.read_sensor)
        # self.pBtn_GetData.clicked.connect(self.get_data)
        self.pBtn_GetData.clicked.connect(self.read_data)

        self.pBtn_Outliers.clicked.connect(self.remove_outliers)
        self.pBtn_Integrate.clicked.connect(self.integrate_data)
        self.pBtn_Slope.clicked.connect(self.remove_slope)
        

                
        # Блокировка кнопок во время операций
        self.update_buttons_state(True)

        # Инициализация статус бара
        self.show_status_message("Готов к работе")

    def show_status_message(self, message, timeout=5000):
        """Показать сообщение в статус баре"""
        self.statusBar.showMessage(message, timeout)

    def init_graph(self):
        """Инициализация графика"""
        self.chart = QChart()
        # self.chart.setTitle("Числовые данные")
        self.chart.legend().setVisible(False)
        
        self.series = QLineSeries()
        self.chart.addSeries(self.series)
        self.chart.createDefaultAxes()
        
        self.chart.axisX().setLabelFormat("%d")
        # axis_x = self.chart.axisX()
        # axis_x.setTitleText("Сигнал")
        # axis_x.setLabelFormat("%d")
        
        self.chart.axisY().setLabelFormat("%d")
        # axis_y = self.chart.axisY()
        # axis_y.setTitleText("Индекс")
        # axis_y.setLabelFormat("%d")
        
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.Layout_3h.addWidget(self.chart_view)
    
    def update_buttons_state(self, enabled):
        """Обновление состояния кнопок"""
        self.pBtn_Show.setEnabled(enabled)
        self.pBtn_Save.setEnabled(enabled)

        # self.pBtn_Motor.setEnabled(enabled)
        # self.pBtn_ReadSensor.setEnabled(enabled)
        self.pBtn_GetData.setEnabled(enabled)

        self.pBtn_Outliers.setEnabled(enabled)
        self.pBtn_Integrate.setEnabled(enabled)
        self.pBtn_Slope.setEnabled(enabled)
                
    def set_wait_cursor(self, waiting):
        """Установка курсора ожидания"""
        if waiting:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
    
    def read_data(self):
        """Объединенная функция: запуск мотора, чтение датчика и получение данных"""
        self.motor_port = self.cBox_MotorPort.currentText()
        self.sensor_port = self.cBox_SensorPort.currentText()
        
        if not self.motor_port or not self.sensor_port:
            QMessageBox.warning(self, "Ошибка", "Выберите порты мотора и датчика")
            return
        
        self.update_buttons_state(False)
        self.set_wait_cursor(True)
        self.show_status_message("Запуск процесса сбора данных...")
        
        # Последовательное выполнение операций с задержками
        try:
            self.motor_controller.run_motor(self.motor_port)
        except Exception as e:
            self.on_serial_error(f"Ошибка запуска мотора: {str(e)}")
        QTimer.singleShot(500, lambda: self.read_sensor())  # Задержка полсекунды перед чтением датчика
    
    
    def read_sensor(self):
        """Чтение датчика после запуска мотора"""
        self.serial_worker = SerialWorker(self.sensor_port, 'R', 11)
        self.serial_worker.finished.connect(self.on_read_sensor_finished)
        self.serial_worker.error.connect(self.on_serial_error)
        self.serial_worker.start()
    
    def on_read_sensor_finished(self):
        """Обработка завершения чтения датчика"""
        self.show_status_message("Датчик прочитан, получение данных...")
        print("Датчик прочитан")
        QTimer.singleShot(500, self.get_data)  # Задержка полсекунды перед получением данных
    
    def get_data(self):
        """Получение данных после чтения датчика"""
        # port = self.cBox_SensorPort.currentText()
        self.serial_worker = SerialWorker(self.sensor_port, 'S', 47, 4194305)
        self.serial_worker.data_ready.connect(self.on_data_received)
        self.serial_worker.error.connect(self.on_serial_error)
        self.serial_worker.finished.connect(self.on_get_data_finished)
        self.serial_worker.start()
    
    def on_data_received(self, raw_data):
        """Обработка полученных данных"""
        try:
            self.df = self.data_processor.process_raw_data(raw_data)
            self.df_processed = self.df.copy()  # Сохраняем копию для обработки
            self.show_status_message('Данные получены! Готово к интегрированию!')
            print('Данные получены! Готово к интегрированию!')
            self.update_graph(self.df_processed)  # Обновляем график после получения данных
        except Exception as e:
            self.on_serial_error(f"Ошибка обработки данных: {str(e)}")
    
    def on_get_data_finished(self):
        """Обработка завершения получения данных"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
        self.show_status_message("Процесс сбора данных завершен!")
    
    def on_serial_error(self, error_msg):
        """Обработка ошибок последовательного порта"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
        QMessageBox.critical(self, "Ошибка", error_msg)
        print(f"Ошибка: {error_msg}")
    
    def integrate_data(self):
        """Интегрирование данных"""
        if self.df_processed is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для интегрирования")
            return
        
        # Интегрирование может занять время, показываем прогресс
        self.set_wait_cursor(True)
        try:
            self.df_processed = self.data_processor.integrate_df(self.df_processed)
            self.show_status_message('Данные проинтегрированы!')
            print('Данные проинтегрированы!')
            self.update_graph(self.df_processed)  # Обновляем график после интегрирования
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка интегрирования: {str(e)}")
        finally:
            self.set_wait_cursor(False)
    
    def remove_outliers(self):
        """Удаление выбросов из сырых данных"""
        if self.df_processed is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для обработки")
            return
        
        self.set_wait_cursor(True)
        try:
            self.df_processed = self.data_processor.apply_median_filter(self.df, window_size=3)
            self.show_status_message('Выбросы удалены!')
            print('Выбросы удалены!')
            self.update_graph(self.df_processed)  # Обновляем график после удаления выбросов
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка удаления выбросов: {str(e)}")
        finally:
            self.set_wait_cursor(False)
    
    def remove_slope(self):
        """Удаление линейного наклона из данных"""
        if not hasattr(self, 'df_processed') or self.df_processed is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для обработки")
            return
        
        self.set_wait_cursor(True)
        try:
            self.df_processed.data = self.data_processor.apply_linear_correction(self.df_processed.data)
            self.show_status_message('Наклон удален!')
            print('Наклон удален!')
            self.update_graph(self.df_processed)  # Обновляем график после удаления наклона
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка удаления наклона: {str(e)}")
        finally:
            self.set_wait_cursor(False)
    
    def update_graph(self, df):
        """Обновление графика текущими данными"""
        if self.df_processed is None:
            return
        
        # Обновляем график
        self.chart.removeAllSeries()

        # Создаем новую серию
        series = QLineSeries()
        
        # Добавляем точки данных
        for row in df.itertuples():
            series.append(row.Index, row.data)
        
        self.chart.addSeries(series)
        self.chart.createDefaultAxes()
        self.chart.axisX().setLabelFormat("%d")
        self.chart.axisY().setLabelFormat("%d")
    
    def show_data(self):
        """Отображение данных на графике"""
        if self.df_processed is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для отображения, загрузка тестовых данных")
            self.df = pd.read_csv(os.path.join('data', 'data2.csv'), index_col=0)
            self.df_processed = self.df.copy()  # Сохраняем копию для обработки
            return
        
        self.show_status_message('Выводим данные...', 500)
        self.set_wait_cursor(True)
        self.update_graph(self.df_processed)
        self.set_wait_cursor(False)
    
    def run_motor(self):
        """Запуск мотора на определенное количество оборотов"""
        port = self.cBox_MotorPort.currentText()
        if not port:
            QMessageBox.warning(self, "Ошибка", "Выберите порт мотора")
            return
        
        # Запускаем в отдельном потоке, чтобы не блокировать UI
        QTimer.singleShot(0, lambda: self.motor_controller.run_motor(port))
    
    def save_data(self):
        """Сохранение данных в файл"""
        if self.df_processed is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return
        
        try:
            datadir = 'data'
            filename = f"data_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            
            self.df_processed.to_csv(os.path.join(datadir, filename))
            
            self.show_status_message(f'Данные сохранены в файл: {filename}')
            print(f'Данные сохранены в файл: {filename}')
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения данных: {str(e)}")


if __name__ == '__main__':
    app = QApplication([])
    
    # Установка обработчика неперехваченных исключений
    def exception_handler(exctype, value, traceback):
        print(f"Необработанное исключение: {exctype.__name__}: {value}")
        QMessageBox.critical(None, "Ошибка", f"Произошла ошибка:\n{value}")
    
    sys.excepthook = exception_handler
    
    window = MainUI()
    window.show()
    
    sys.exit(app.exec_())