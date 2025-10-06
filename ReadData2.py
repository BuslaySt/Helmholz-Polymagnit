import sys
import time
import os
from typing import Optional, Tuple
import serial
import serial.tools.list_ports
import pandas as pd
import numpy as np
from scipy import signal

from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QMessageBox
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QTimer


class SerialDevice:
    """Класс для работы с последовательным портом"""
    
    def __init__(self, port: str, baudrate: int = 921600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
    
    def __enter__(self):
        self.serial_conn = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=8,
            stopbits=1,
            timeout=self.timeout
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
    
    def send_command(self, command: str) -> bool:
        """Отправка команды на устройство"""
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.write(command.encode())
                return True
        except serial.SerialException as e:
            print(f"Ошибка отправки команды: {e}")
        return False
    
    def read_data(self, size: int) -> Optional[bytes]:
        """Чтение данных из порта"""
        try:
            if self.serial_conn and self.serial_conn.is_open:
                return self.serial_conn.read(size)
        except serial.SerialException as e:
            print(f"Ошибка чтения данных: {e}")
        return None

class DataProcessor:
    """Класс для обработки и фильтрации данных"""
    
    @staticmethod
    def process_raw_data(raw_data: bytes) -> Tuple[list, list]:
        """Обработка сырых данных в числовые массивы"""
        if len(raw_data) < 4194305:
            raise ValueError(f"Недостаточно данных: получено {len(raw_data)} байт")
        
        data = []
        encoder = []
        
        # Обработка данных датчиков (первые 2097152 байта)
        for num in range(0, 2097152, 2):
            hi_byte = raw_data[num]
            hi_byte = hi_byte if hi_byte < 128 else hi_byte - 256
            lo_byte = raw_data[num + 1]    
            data.append(hi_byte * 256 + lo_byte)
        
        # Обработка данных энкодера
        for num in range(2097152, len(raw_data) - 1, 2):
            hi_byte = raw_data[num]
            hi_byte = hi_byte if hi_byte < 128 else hi_byte - 256
            lo_byte = raw_data[num + 1]    
            encoder.append(hi_byte * 256 + lo_byte)
        
        return data, encoder
    
    @staticmethod
    def apply_median_filter(data: list, window_size: int = 3) -> np.ndarray:
        """Применение медианного фильтра"""
        return signal.medfilt(data, kernel_size=window_size)

    @staticmethod
    def apply_median_filter(data: list, window_size: int = 3) -> np.ndarray:
        """Применение медианного фильтра"""
        return signal.medfilt(data, kernel_size=window_size)


class DataVisualizer:
    """Класс для визуализации данных"""
    
    @staticmethod
    def create_chart(data: list, title: str = "График числовых данных") -> QChart:
        """Создание графика с данными"""
        chart = QChart()
        chart.setTitle(title)
        chart.legend().setVisible(False)
        
        # Создание серии данных
        series = QLineSeries()
        for i, value in enumerate(data):
            series.append(i, value)
        
        chart.addSeries(series)
        
        # Настройка осей
        axis_x = QValueAxis()
        axis_x.setTitleText("Индекс")
        axis_x.setLabelFormat("%d")
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText("Значение")
        axis_y.setLabelFormat("%d")
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        
        return chart

class MainUI(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._setup_connections()
        self._initialize_variables()
    
    def _setup_ui(self) -> None:
        """Инициализация пользовательского интерфейса"""
        loadUi("ReadData.ui", self)
        self._populate_com_ports()
    
    def _setup_connections(self) -> None:
        """Настройка сигналов и слотов"""
        self.pBtn1_Read.clicked.connect(self.read_data)
        self.pBtn2_Get.clicked.connect(self.get_data)
        # self.pBtn3_Int.clicked.connect(self.fit_data)
        self.pBtn_Show.clicked.connect(self.show_data)
    
    def _initialize_variables(self) -> None:
        """Инициализация переменных"""
        self.df = None
        self.chart_view = None
        self.data_processor = DataProcessor()
        self.visualizer = DataVisualizer()
    
    def _populate_com_ports(self) -> None:
        """Заполнение списка COM-портов"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.cBox_COMPort.addItem(port.device)
    
    def _show_error_message(self, title: str, message: str) -> None:
        """Показать сообщение об ошибке"""
        QMessageBox.critical(self, title, message)
    
    def _show_info_message(self, title: str, message: str) -> None:
        """Показать информационное сообщение"""
        QMessageBox.information(self, title, message)
    
    def read_data(self) -> None:
        """Чтение данных с устройства"""
        try:
            port = self.cBox_COMPort.currentText()
            if not port:
                self._show_error_message("Ошибка", "COM-порт не выбран")
                return
            
            with SerialDevice(port, timeout=11) as device:
                if device.send_command('R'):
                    # Используем QTimer для неблокирующей задержки
                    QTimer.singleShot(11000, lambda: self._show_info_message("Готово", "Данные прочитаны успешно!"))
                else:
                    self._show_error_message("Ошибка", "Не удалось отправить команду чтения")
        
        except Exception as e:
            self._show_error_message("Ошибка", f"Ошибка при чтении данных: {str(e)}")
    
    def get_data(self) -> None:
        """Получение данных с устройства"""
        try:
            port = self.cBox_COMPort.currentText()
            if not port:
                self._show_error_message("Ошибка", "COM-порт не выбран")
                return
            
            with SerialDevice(port, timeout=47) as device:
                if device.send_command('S'):
                    raw_data = device.read_data(4194305)
                    
                    if raw_data and len(raw_data) >= 4194305:
                        data, encoder = self.data_processor.process_raw_data(raw_data)
                        self.df = pd.DataFrame({'encoder': encoder, 'data': data})
                        self._show_info_message("Успех", "Данные получены и обработаны")
                    else:
                        self._show_error_message("Ошибка", "Не удалось получить данные или данные неполные")
                else:
                    self._show_error_message("Ошибка", "Не удалось отправить команду получения данных")
        
        except Exception as e:
            self._show_error_message("Ошибка", f"Ошибка при получении данных: {str(e)}")
    
    def show_data(self) -> None:
        """Отображение данных на графике"""
        try:
            # Загрузка тестовых данных если нет реальных
            if not hasattr(self, 'df') or self.df is None:
                test_data_path = os.path.join('data', 'data1.csv')
                if os.path.exists(test_data_path):
                    self.df = pd.read_csv(test_data_path)
                else:
                    self._show_error_message("Ошибка", "Нет данных для отображения")
                    return
            
            # Очистка предыдущего графика
            if self.chart_view:
                self.chart_view.deleteLater()
            
            # Фильтрация данных
            filtered_data = self.data_processor.apply_median_filter(self.df['data'].values)
            
            # Создание и отображение графика
            chart = self.visualizer.create_chart(filtered_data, "График данных с медианным фильтром")
            self.chart_view = QChartView(chart)
            
            # Добавление графика в layout
            # if hasattr(self, 'verticalSpacer'):
            #     print('verticalSpacer found')
            self.Layout_3h.removeItem(self.Layout_3h.itemAt(0))
            self.Layout_3h.addWidget(self.chart_view)

        except Exception as e:
            self._show_error_message("Ошибка", f"Ошибка при отображении данных: {str(e)}")


def main():
    """Точка входа в приложение"""
    try:
        app = QApplication(sys.argv)
        
        window = MainUI()
        window.show()
        
        sys.exit(app.exec_())
    
    except Exception as e:
        print(f"Критическая ошибка при запуске приложения: {e}")
        return 1


if __name__ == '__main__':
    main()