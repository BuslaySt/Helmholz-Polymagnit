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
    def integrate_data(df_filtered, step=20):
        """Интегрирование данных"""
        # фильтрация сделана сразу после получения данных
        # df_filtered = DataProcessor.apply_median_filter(df)
        
        # Находим точки разделения по encoder
        split_points = df_filtered.index[
            (df_filtered['encoder'].shift(1) == 9999) & 
            (df_filtered['encoder'] == 0)
        ]
        
        # Создаем интервалы
        bins = list(range(0, 10000, step))
        datasets = []
        
        # Обрабатываем каждый сегмент данных
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            dataset_part = df_filtered.iloc[start_idx:end_idx].copy()
            # Усреднение данных по encoder
            datasets.append(
                dataset_part.groupby('encoder', as_index=False)['data'].mean()
            )
        
        # Интегрируем данные
        df_integral = pd.DataFrame(index=range(len(bins) - 1))
        
        for i, dataset in enumerate(datasets):
            # Разделение и группировка
            dataset['bin'] = pd.cut(
                dataset['encoder'], bins=bins, right=False, labels=bins[:-1]
            )
            grouped = dataset.groupby('bin', observed=False)['data'].mean().reset_index()
            
            # Линейное интегрирование с коррекцией
            line = (grouped['data'] * step).cumsum()
            difference = line.iloc[0] - line.iloc[-1]
            n = len(line)
            
            corrected_data = []
            for j, value in enumerate(line):
                correction = (difference * j) / (n - 1) if n > 1 else 0
                corrected_data.append(value + correction)
            
            df_integral[f'integral_data_{i+1}'] = corrected_data
        
        # Возвращаем усредненные данные
        result_df = pd.DataFrame(columns=['encoder', 'data'], index=range(500))
        result_df['encoder'] = range(500)
        result_df['data'] = df_integral.mean(axis=1)
        
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
    
    # @staticmethod
    # def start_motor(port, direction=1, speed=50):
    #     """Непрерывный запуск мотора"""
    #     try:
    #         with serial.Serial(port, baudrate=57600, bytesize=8, 
    #                          parity='N', stopbits=1, timeout=0) as serial_conn:
    #             command = f'ON\rMOVE C(inp1)D({direction})F({speed})\rOFF\r'
    #             return serial_conn.write(command.encode("utf-8"))
    #     except Exception as e:
    #         print(f"Ошибка запуска мотора: {e}")
    #         return 0
    

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        loadUi("ReadData.ui", self)
        
        self.serial_worker = None
        self.df = None
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
        self.pBtn_ReadSensor.clicked.connect(self.read_sensor)
        self.pBtn_GetData.clicked.connect(self.get_data)

        self.pBtn_Outliers.clicked.connect(self.remove_outliers)
        self.pBtn_Integrte.clicked.connect(self.integrate_data)
        self.pBtn_Slope.clicked.connect(self.remove_slope)
        
        self.pBtn_Motor.clicked.connect(self.run_motor)
        self.pBtn_Show.clicked.connect(self.show_data)
        self.pBtn_Save.clicked.connect(self.save_data)
                
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
        self.chart.setTitle("Числовые данные")
        self.chart.legend().setVisible(False)
        
        self.series = QLineSeries()
        self.chart.addSeries(self.series)
        self.chart.createDefaultAxes()
        
        axis_x = self.chart.axisX()
        axis_x.setTitleText("Сигнал")
        axis_x.setLabelFormat("%d")
        
        axis_y = self.chart.axisY()
        axis_y.setTitleText("Индекс")
        axis_y.setLabelFormat("%d")
        
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.Layout_3h.addWidget(self.chart_view)
    
    def update_buttons_state(self, enabled):
        """Обновление состояния кнопок"""
        self.pBtn_Read.setEnabled(enabled)
        self.pBtn_Get.setEnabled(enabled)
        self.pBtn_Int.setEnabled(enabled)
        self.pBtn_Motor.setEnabled(enabled)
        self.pBtn_MotorStart.setEnabled(enabled)
        self.pBtn_Show.setEnabled(enabled)
    
    def set_wait_cursor(self, waiting):
        """Установка курсора ожидания"""
        if waiting:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
    
    def read_sensor(self):
        """Чтение данных с датчика в отдельном потоке"""
        port = self.cBox_SensorPort.currentText()
        if not port:
            QMessageBox.warning(self, "Ошибка", "Выберите порт датчика")
            return
        
        self.update_buttons_state(False)
        self.set_wait_cursor(True)
        self.serial_worker = SerialWorker(port, 'R', 11)
        self.serial_worker.finished.connect(self.on_read_finished)
        self.serial_worker.error.connect(self.on_serial_error)
        self.serial_worker.start()
    
    def on_read_finished(self):
        """Обработка завершения чтения"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
        self.show_status_message('Датчик прочитан успешно! Теперь можно получить данные!')
        print('Датчик прочитан успешно! Теперь можно получить данные!')
    
    def get_data(self):
        """Получение данных в отдельном потоке"""
        port = self.cBox_SensorPort.currentText()
        if not port:
            QMessageBox.warning(self, "Ошибка", "Выберите порт датчика")
            return
        
        self.update_buttons_state(False)
        self.set_wait_cursor(True)
        self.serial_worker = SerialWorker(port, 'S', 47, 4194305)
        self.serial_worker.data_ready.connect(self.on_data_received)
        self.serial_worker.error.connect(self.on_serial_error)
        self.serial_worker.finished.connect(self.on_get_data_finished)
        self.serial_worker.start()
    
    def on_data_received(self, raw_data):
        """Обработка полученных данных"""
        try:
            df = self.data_processor.process_raw_data(raw_data)
            self.df = DataProcessor.apply_median_filter(df)
            self.show_status_message('Данные получены! Готово к интегрированию!')
            print('Данные получены! Готово к интегрированию!')
        except Exception as e:
            self.on_serial_error(f"Ошибка обработки данных: {str(e)}")
    
    def on_get_data_finished(self):
        """Обработка завершения получения данных"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
    
    def on_serial_error(self, error_msg):
        """Обработка ошибок последовательного порта"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
        QMessageBox.critical(self, "Ошибка", error_msg)
        print(f"Ошибка: {error_msg}")
    
    def integrate_data(self):
        """Интегрирование данных"""
        if not hasattr(self, 'df') or self.df is None:
            try:
                self.df = pd.read_csv(os.path.join('data', 'data1.csv'), index_col=0)
            except FileNotFoundError:
                QMessageBox.warning(self, "Ошибка", "Файл данных не найден")
                return
        
        # Интегрирование может занять время, показываем прогресс
        self.set_wait_cursor(True)
        try:
            self.df = self.data_processor.integrate_data(self.df)
            self.show_status_message('Данные проинтегрированы!')
            print('Данные проинтегрированы!')
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка интегрирования: {str(e)}")
        finally:
            self.set_wait_cursor(False)
    
    def show_data(self):
        """Отображение данных на графике"""
        # Устанавливаем курсор ожидания
        self.show_status_message('Выводим данные...', 500)
        self.set_wait_cursor(True)

        if not hasattr(self, 'df') or self.df is None:
            try:
                df = pd.read_csv(os.path.join('data', 'data1.csv'), index_col=0)
                self.df = DataProcessor.apply_median_filter(df)
            except FileNotFoundError:
                QMessageBox.warning(self, "Ошибка", "Нет данных для отображения")
                return
        
        # self.series.clear()
        self.series = QLineSeries()
        
        # Добавляем точки данных
        for row in self.df.itertuples():
            self.series.append(row.Index, row.data)
        
        # Обновляем график
        self.chart.removeAllSeries()
        self.chart.addSeries(self.series)
        self.chart.createDefaultAxes()

        # Восстанавливаем обычный курсор
        self.set_wait_cursor(False)
    
    def run_motor(self):
        """Запуск мотора на определенное количество оборотов"""
        port = self.cBox_MotorPort.currentText()
        if not port:
            QMessageBox.warning(self, "Ошибка", "Выберите порт мотора")
            return
        
        # Запускаем в отдельном потоке, чтобы не блокировать UI
        QTimer.singleShot(0, lambda: self.motor_controller.run_motor(port))
    
    def start_motor(self):
        """Непрерывный запуск мотора"""
        port = self.cBox_MotorPort.currentText()
        if not port:
            QMessageBox.warning(self, "Ошибка", "Выберите порт мотора")
            return
        
        QTimer.singleShot(0, lambda: self.motor_controller.start_motor(port))
    
    def stop_motor(self):
        """Остановка мотора"""
        port = self.cBox_MotorPort.currentText()
        if not port:
            QMessageBox.warning(self, "Ошибка", "Выберите порт мотора")
            return
        
        QTimer.singleShot(0, lambda: self.motor_controller.stop_motor(port))


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