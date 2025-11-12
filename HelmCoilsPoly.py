from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QScatterSeries
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QColor
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
                    # Для операций только записи ждем завершения
                    time.sleep(self.timeout)
                    
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Ошибка последовательного порта: {str(e)}")

class DataProcessor:
    """Класс для обработки данных"""
    
    @staticmethod
    def process_raw_data(raw_data: bytes) -> pd.DataFrame:
        """Перевод байтовой строки в числа
        data - поток данных
        encoder - измерения энкодера
        """
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
        """Полное интегрирование данных - основной метод
        df_filtered - обработанные данные
        step - шаг интегрирования - 10000 / 20 = 500 градусов
        """
        # Создание интервалов
        bins = range(0, len(df_filtered) + step, step)

        # Находим индексы, где происходит переход с 9999 на 0
        split_points = df_filtered.index[(df_filtered['encoder'].shift(1) == 9999) & (df_filtered['encoder'] == 0)]

        # Добавляем начало и конец датасета
        split_points = [0] + split_points.tolist() + [len(df_filtered)]

        # Создаем список разделенных датафреймов с группировкой по encoder
        datasets = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            dataset_part = df_filtered.iloc[start_idx:end_idx].copy()
            dataset_part.encoder = dataset_part.encoder + 10000 * i

            # Группировка по encoder и усреднение data
            dataset_part['bin'] = pd.cut(dataset_part['encoder'], bins=bins, right=False, labels=bins[:-1])
            grouped = dataset_part.groupby('bin', observed=True)['data'].sum()
            datasets.append(grouped)    

        df_encoder = pd.concat(datasets).reset_index()
        df_encoder['integrated_data'] = -1*df_encoder['data'].cumsum()/32767*2.5*(10**-5) # 2.5/32767*10**-5 - коэф. для перевода в Вольты*сек

        x = df_encoder.integrated_data.index.values
        y = df_encoder.integrated_data.values

        # Линейная регрессия для выделения тренда
        coefficients = np.polyfit(x, y, 1)  # 1 - линейный тренд
        trend = np.polyval(coefficients, x)

        # Детрендированные данные
        df_encoder['data'] = y - trend
        df_encoder['deg'] = df_encoder.index/10000/step*360

        df = df_encoder.reindex(columns=['deg', 'data'])
        
        # Локальные максимумы
        local_maxima = argrelextrema(df.data.values, np.greater, order=100)[0]
        # Локальные минимумы
        local_minima = argrelextrema(df.data.values, np.less, order=100)[0]

        # Добавляем метки в DataFrame
        df['is_local_max'] = False
        df['is_local_min'] = False

        df.loc[local_maxima, 'is_local_max'] = True
        df.loc[local_minima, 'is_local_min'] = True

        return df

    @staticmethod
    def get_amplitude(df: pd.DataFrame) -> set:
        """Вычисление амплитуды"""
        # Извлечение экстремумов
        maxima = df[df['is_local_max']]
        minima = df[df['is_local_min']]

        # Вычисляем средние значения
        mean_max = maxima['data'].mean()
        mean_min = minima['data'].mean()

        # Половина размаха разности средних - амплитуда сигнала
        amplitude = (mean_max - mean_min)/2

        # Погрешность среднего значения максимумов
        std_max = maxima['data'].std(ddof=1)
        std_error_max = std_max / np.sqrt(len(maxima))

        # Погрешность среднего значения минимумов
        std_min = minima['data'].std(ddof=1)
        std_error_min = std_min / np.sqrt(len(minima))

        # Абсолютная погрешность амплитуды (по формуле погрешности разности)
        absolute_error = np.sqrt(std_error_max**2 + std_error_min**2) / 2

        # Относительная погрешность амплитуды
        relative_error = absolute_error / amplitude * 100  # в процентах

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
        self.pBtn_GetData.clicked.connect(self.read_data)

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
        # self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.Layout_3h.addWidget(self.chart_view)
    
    def update_buttons_state(self, enabled):
        """Обновление состояния кнопок"""
        # self.pBtn_Show.setEnabled(enabled)
        # self.pBtn_Save.setEnabled(enabled)

        self.pBtn_GetData.setEnabled(enabled)

        # self.pBtn_Outliers.setEnabled(False)
        # self.pBtn_Integrate.setEnabled(False)
        # self.pBtn_Slope.setEnabled(False)
                
    def set_wait_cursor(self, waiting):
        """Установка курсора ожидания"""
        if waiting:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
    
    def read_data(self):
        """Запуск мотора и чтение датчиков"""
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
            df_raw = self.data_processor.process_raw_data(raw_data) # Получаем датафрейм с датчика
            df_filtered = self.data_processor.apply_median_filter(df_raw, window_size=3) # Убираем выбросы
            self.df = self.data_processor.integrate_df(df_filtered) # Интегрируем данные и ищем локальные максимумы

            self.update_graph(self.df)  # Обновляем график после получения данных

            amplitude, absolute_error, relative_error = self.data_processor.get_amplitude(self.df)
            self.show_status_message(f"Полная амплитуда: {amplitude:.5f} ± {absolute_error:.5f} ({relative_error:.2f}%)", 200000)

            self.lbl_1.setText(f"Полная амплитуда: {amplitude:.5f} ± {absolute_error:.5f} ({relative_error:.2f}%)")

            self.show_status_message('Данные получены!')

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

    def update_graph(self, df):
        """Обновление графика текущими данными"""
        
        # Обновляем график
        self.chart.removeAllSeries()

        # Создаем новую серию
        data_series = QLineSeries()
        # Метки максимумов
        # max_series = QScatterSeries()
        # max_series.setColor(QColor(255, 0, 0))  # Красный цвет
        # max_series.setMarkerSize(8)  # Размер маркера
        # # Метки минимумов
        # min_series = QScatterSeries()
        # min_series.setColor(QColor(0, 0, 255))  # Синий цвет
        # min_series.setMarkerSize(8)  # Размер маркера
        
        # Добавляем точки данных
        for row in df.itertuples():
            data_series.append(row.deg, row.data)
            # if row.is_local_max:
            #     max_series.append(row.deg, row.data)
            # if row.is_local_min:
            #     min_series.append(row.deg, row.data)
        
        self.chart.addSeries(data_series)
        # self.chart.addSeries(max_series)
        # self.chart.addSeries(min_series)
        self.chart.createDefaultAxes()
        self.chart.axisX().setLabelFormat("%d")
        self.chart.axisY().setLabelFormat("%d")
    
    def run_motor(self):
        """Запуск мотора на определенное количество оборотов"""
        port = self.cBox_MotorPort.currentText()
        if not port:
            QMessageBox.warning(self, "Ошибка", "Выберите порт мотора")
            return
        
        # Запускаем в отдельном потоке, чтобы не блокировать UI
        QTimer.singleShot(0, lambda: self.motor_controller.run_motor(port))
    
    # def save_data(self):
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