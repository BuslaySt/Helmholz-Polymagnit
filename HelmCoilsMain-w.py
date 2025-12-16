from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QLabel
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from pyqtgraph import PlotWidget
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPixmap
import sys, time, os, datetime
import serial.tools.list_ports
import pandas as pd
import numpy as np
import fastgoertzel as fg
from scipy import integrate
from scipy.signal import argrelextrema, medfilt

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
        except Exception as e:
            self.error.emit(f"Ошибка последовательного порта: {str(e)}")
        finally:
            self.finished.emit()

class DataProcessor:
    """Класс для обработки данных"""
    
    @staticmethod
    def process_raw_data(raw_data: bytes) -> pd.DataFrame:
        """Перевод байтовой строки в числа"""
        rawdata = np.frombuffer(raw_data, dtype=np.uint8)

        raw_signal = rawdata[:2097152]
        # Чтение по два байта (старший-младший) в нотации "big-endian" с переводом в нотацию компилятора (little-endian)
        data = raw_signal.view(dtype='>i2').astype(np.int16)

        raw_encoder = rawdata[2097152:]
        encoder = raw_encoder.view(dtype='>i2').astype(np.int16)

        # df_raw = pd.DataFrame({
        #     'encoder': encoder,
        #     'data': data
        # })
        
        # data = []
        # for num in range(0, 2097152, 2):
        #     hi_byte = raw_data[num]
        #     hi_byte = hi_byte if hi_byte < 128 else hi_byte - 256
        #     lo_byte = raw_data[num + 1]    
        #     data.append(hi_byte * 256 + lo_byte)

        # encoder = []
        # for num in range(2097152, len(raw_data) - 1, 2):
        #     hi_byte = raw_data[num]
        #     hi_byte = hi_byte if hi_byte < 128 else hi_byte - 256
        #     lo_byte = raw_data[num + 1]    
        #     encoder.append(hi_byte * 256 + lo_byte)
            
        return pd.DataFrame({'encoder': encoder, 'data': data})
    
    @staticmethod
    def apply_median_filter(df: pd.DataFrame, window_size:int=3) -> pd.DataFrame:
        """Применение медианного фильтра для удаления дельта-выбросов"""
        # df_filtered = pd.DataFrame(columns=['encoder', 'data'])
        # df_filtered['data'] = medfilt(df['data'], kernel_size=window_size)[window_size*2:-window_size*2]
        # df_filtered['encoder'] = medfilt(df['encoder'], kernel_size=window_size)[window_size*2:-window_size*2]
        trim_offset = window_size // 2
        return pd.DataFrame({
                    'encoder': medfilt(df.encoder, kernel_size=window_size)[trim_offset:-trim_offset],
                    'data': medfilt(df.data, kernel_size=window_size)[trim_offset:-trim_offset]    
                })

    @staticmethod
    def truncate_marginal_periods(df: pd.DataFrame) -> pd.DataFrame:
        """Отсечение целых оборотов по переходу нуля энкодера"""
        # Найти индексы, где энкодер "прыгнул"
        diff_enc = df['encoder'].shift(1) - df['encoder']
        split_points = df.index[diff_enc.abs() > 1000]  # по модулю, чтобы не зависить от направления (?)

        # print(f'Найдено {len(split_points)-1} периодов энкодера')

        if len(split_points) in (0, 1):
            # print("Warning: Нет переходов нуля энкодера.")
            return df
        else:
            start_idx = split_points[0]
            end_idx = split_points[-1]
            # Проверка, чтобы start_idx был меньше end_idx
            if start_idx >= end_idx:
                # print(f"Warning: start_idx ({start_idx}) >= end_idx ({end_idx}). Returning original DataFrame.")
                return df
            else:
                return df.iloc[start_idx:end_idx].copy()

    @staticmethod
    def integrate_df(df_trimmed: pd.DataFrame, step:int=25) -> pd.DataFrame:
        """Полное интегрирование данных - основной метод"""
        # Усредняем по значениям encoder и вычисляем интеграл по всему периоду данных
        # 1. Группировка по периодам (непрерывные одинаковые значения encoder)
        df_trimmed['period'] = (df_trimmed['encoder'] != df_trimmed['encoder'].shift()).cumsum()

        # 2. Группируем по периоду, затем вычисляем среднее, после чего сбрасываем индекс и период
        df_res = df_trimmed.groupby('period').agg({'data': 'sum', 'encoder': 'first'}).reset_index(drop=True)

        # 3.1 Интеграл (кумулятивная сумма)
        # df_res['integral'] = -1.0*df_res.data.cumsum()

        # 3.2 Интеграл (трапециями по единичному отрезку)
        dt = 1
        # минус из формулы интегрирования
        df_res['integral'] = -1.0 * integrate.cumulative_trapezoid(df_res['data'], dx=dt, initial=0)

        # 4. Пересчет в Вольты*метры*секунды
        #  2.5/32767 - коэф. для перевода в Вольты, 1/96937 в сек (timebase), 1/1144.8 в м (постоянная катушки)
        df_res['volts'] = (2.5/32767 * 1/96937 * 1/1144.8)*df_res['integral']

        # 5. Угол в градусах
        df_res['deg'] = (df_res['encoder']/10000)*360

        return df_res

        # bins = range(0, len(df_filtered) + step, step)

        # split_points = df_filtered.index[(df_filtered['encoder'].shift(1) - df_filtered['encoder'] > 1000)] # Сплитуем по переходу больше 1000
        # # split_points = [0] + split_points.tolist() + [len(df_filtered)]

        # datasets = []
        # for i in range(len(split_points) - 1):
        #     start_idx = split_points[i]
        #     end_idx = split_points[i + 1]
        #     dataset_part = df_filtered.iloc[start_idx:end_idx].copy()
        #     dataset_part.encoder = dataset_part.encoder + 10000 * i

        #     dataset_part['bin'] = pd.cut(dataset_part['encoder'], bins=bins, right=False, labels=bins[:-1])
        #     grouped = dataset_part.groupby('bin', observed=True)['data'].sum()
        #     datasets.append(grouped)    

        # df_encoder = pd.concat(datasets).reset_index()
        # df_encoder['integrated_data'] = -(2.5/32767 * 1/96937)*df_encoder['data'].cumsum()
        # # 2.5/32767 - коэф. для перевода в Вольты, 1/96937 в сек (timebase), минус из формулы интегрирования

        # x = df_encoder.integrated_data.index.values
        # y = df_encoder.integrated_data.values

        # # Координаты первой и последней точек
        # x0, x1 = x[0], x[-1]
        # y0, y1 = y[0], y[-1]

        # # Уравнение прямой через две точки: y_trend = a * x + b
        # a = (y1 - y0) / (x1 - x0) if x1 != x0 else 0
        # b = y0 - a * x0

        # # Вычисляем трендовую составляющую и вычитаем
        # trend = a * x + b
        # df_encoder['data'] = (y - trend) / 1144.8 # 1144.8 - Постоянная катушки [1/м]
        
        # df_encoder['deg'] = df_encoder.index/10000/step*360 # В градусах

        # df = df_encoder.reindex(columns=['deg', 'data'])
        
        # local_maxima = argrelextrema(df.data.values, np.greater, order=100)[0]
        # local_minima = argrelextrema(df.data.values, np.less, order=100)[0]

        # df['is_local_max'] = False
        # df['is_local_min'] = False

        # df.loc[local_maxima, 'is_local_max'] = True
        # df.loc[local_minima, 'is_local_min'] = True

        return df

    @staticmethod
    def get_amplitude(df_res: pd.DataFrame) -> tuple:
        """Вычисление амплитуды алгоритмом Гёрцеля"""
        norm_freq = 1 / 10000
        f_amp, f_phase = fg.goertzel(df_res.volts.values, norm_freq)
        # сдвиг рассчитанной фазы на +π/2 и перевод в градусы
        f_phase_deg = (f_phase+np.pi/2)*180/np.pi
        # print(f'Fast Goertzel Amp: {f_amp:.5e}, {f_phase = :.3f}°')

        return (f_amp, f_phase+np.pi/2, f_phase_deg)

        # maxima = df[df['is_local_max']]
        # minima = df[df['is_local_min']]

        # mean_max = maxima['data'].mean()
        # mean_min = minima['data'].mean()

        # amplitude = (mean_max - mean_min)/2

        # std_max = maxima['data'].std(ddof=1)
        # std_error_max = std_max / np.sqrt(len(maxima))

        # std_min = minima['data'].std(ddof=1)
        # std_error_min = std_min / np.sqrt(len(minima))

        # absolute_error = np.sqrt(std_error_max**2 + std_error_min**2) / 2
        # relative_error = absolute_error / amplitude * 100

        # return (amplitude, absolute_error, relative_error)

class MotorController:
    """Класс для управления мотором"""
    
    @staticmethod
    def run_motor(port, revolutions=26, distance=1000/9, speed=100):
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
    
    def add_measurement(self, amp, phase_deg):
        """Добавить результат измерения"""
        self.measurements.append({
            'amplitude': amp,
            'phase': phase_deg,
        })
        self.current_measurement += 1
    
    def save_current_measurement_data(self, amp, phase_deg):
        """Сохранить данные текущего измерения (для возможного повтора)"""
        self.current_measurement_data = {
            'amplitude': amp,
            'phase': phase_deg,
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
    
    def get_current_measurement_number(self) -> int:
        """Получить номер текущего измерения"""
        return self.current_measurement + 1
    
    def get_final_result(self) -> set:
        """Получить финальный результат измерений"""
        if len(self.measurements) != 3:
            return None
        
        # Извлекаем амплитуды из всех трех измерений
        amplitudes = sorted([m['amplitude'] for m in self.measurements])
        
        # Вычисляем финальный результат по формуле
        sum_of_squares = sum(amp**2 for amp in amplitudes) / 2
        final_amplitude = np.sqrt(sum_of_squares)
        
        # Вычисляем погрешность финального результата
        # Погрешность по формуле для функции f = sqrt(a^2 + b^2 + c^2)/2
        # absolute_errors = [m['absolute_error'] for m in self.measurements]
        
        # Частные производные: df/dai = (ai) / (2 * sqrt(a1^2 + a2^2 + a3^2))
        # denominator = 2 * np.sqrt(sum_of_squares)
        # partial_derivatives = [amp / denominator for amp in amplitudes]
        
        # Погрешность финального результата
        # final_absolute_error = np.sqrt(
        #     sum((partial_derivatives[i] * absolute_errors[i])**2 for i in range(3))
        # )
        
        # Относительная погрешность
        # final_relative_error = final_absolute_error / final_amplitude * 100

        # Угол отклонения от нормали (оси z)
        M_xy = amplitudes[0]
        M_yz = amplitudes[1]
        M_zx = amplitudes[2]

        theta_rad = np.arctan(M_xy / (np.sqrt(M_yz**2 + M_zx**2 - M_xy**2)/2))
        theta_deg = np.degrees(theta_rad)
        
        return (final_amplitude, theta_deg)
    
    def get_individual_results(self):
        """Получить результаты отдельных измерений"""
        return self.measurements.copy()

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        
        # Загрузка UI из файла
        loadUi("HelmCoilsUI.ui", self)
        
        self.serial_worker = None
        self.df = None
        self.data_processor = DataProcessor()
        self.motor_controller = MotorController()
        self.measurement_manager = MeasurementManager()

        self.motor_speed = 100
        
        self.init_ui()
        self.init_graph()
        
    def init_ui(self):
        """Инициализация интерфейса"""
        # Загрузка логотипа
        self.load_logo()
        
        # Привязка сигналов к кнопкам и меню
        self.refreshPortsBtn.clicked.connect(self.refresh_ports)
        self.pBtn_GetData.clicked.connect(self.start_measurement_cycle)
        self.pBtn_SaveData.clicked.connect(self.save_data)
        self.action_SaveData.triggered.connect(self.save_data)
        self.action_Settings.triggered.connect(self.settings)
        
        # Список меток измерений для удобного доступа
        self.measurement_labels = [
            self.lbl_measurement1,
            self.lbl_measurement2, 
            self.lbl_measurement3
        ]
        
        # Заполнение поля для заголовка файла
        self.txtEd_FileHeader.setStyleSheet('QPlainTextEdit {color: white}')
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        temperature = "20°C"
        header_text = f'{now}; {temperature}\nНомер магнита: \nОписание: '
        self.txtEd_FileHeader.setPlainText(header_text)
        
        # Инициализация портов
        self.refresh_ports()
        
        # Скрываем прогресс бар изначально
        self.progressBar.setVisible(False)
        
        self.update_buttons_state(True)
        self.show_status_message("Готов к работе")

    def load_logo(self):
        """Загрузка логотипа"""
        try:
            logo_pixmap_1 = QPixmap("AMTClogo.png")
            logo_pixmap_2 = QPixmap("AMTClogo-L.png")
            if not logo_pixmap_1.isNull():
                self.lbl_Logo.setPixmap(logo_pixmap_1.scaled(65, 65, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if not logo_pixmap_2.isNull():
                self.lbl_Logo_2.setPixmap(logo_pixmap_2.scaled(300, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
        self.plot_widget = PlotWidget(self)
        self.chartLayout.addWidget(self.plot_widget)

        # (Опционально) Настраиваем внешний вид
        self.plot_widget.setBackground('w') # Белый фон, например
        # self.plot_widget.setTitle("Название графика") # При желании
        self.plot_widget.setLabel('left', 'Значение (В⋅м⋅с)') # Подпись оси Y
        self.plot_widget.setLabel('bottom', 'Угол (°)') # Подпись оси X
        self.plot_widget.showGrid(x=True, y=True) # Показать сетку

        # Серия данных будет храниться как объект внутри класса
        self.graph_plot = None # Инициализируем переменную для графика


    def init_graph2(self):
        """Инициализация графика"""
        self.chart = QChart()
        # self.chart.setTheme(QChart.ChartThemeBlueIcy)
        self.chart.setTheme(QChart.ChartThemeBlueCerulean)
        self.chart.setBackgroundRoundness(0)
        self.chart.setDropShadowEnabled(True)
        self.chart.legend().setVisible(False)
        
        self.series = QLineSeries()
        self.chart.addSeries(self.series)
        self.chart.createDefaultAxes()
        
        self.chart.axisX().setLabelFormat("%d")
        self.chart.axisY().setLabelFormat("%.1e")
        
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
            QMessageBox.warning(self, "Error", "Выберите порты двигателя и датчика")
            return
        
        self.measurement_manager.start_measurement_cycle()
        self.update_buttons_state(False)
        self.set_wait_cursor(True)
        self.pBtn_GetData.setVisible(False)
        self.portsFrame.setVisible(False)
        self.progressBar.setVisible(True)
        self.progressBar.setMaximum(3)
        self.progressBar.setValue(0)
        
        # Сброс результатов
        for label in self.measurement_labels:
            label.setText("---")
        self.lbl_finalResult.setText("Результат: ---")
        
        self.show_status_message("Начинаем цикл измерений...")
        self.start_single_measurement()
    
    def start_single_measurement(self):
        """Запуск одиночного измерения"""
        current_meas = self.measurement_manager.get_current_measurement_number()
        self.progressLabel.setText(f"Измерение {current_meas}/3 в процессе...")
        self.progressBar.setValue(current_meas - 1)
        
        try: # Запуск вращения
            self.motor_controller.run_motor(self.motor_port, speed=self.motor_speed)
        except Exception as e:
            self.on_serial_error(f"Motor start error: {str(e)}")
            return
        
        QTimer.singleShot(5000, self.read_sensor) # Через 5 секунд начинаем считывать данные
    
    def read_sensor(self):
        """Чтение датчика после запуска мотора"""
        self.serial_worker = SerialWorker(self.sensor_port, 'R', 11)
        self.serial_worker.finished.connect(self.on_read_sensor_finished)
        self.serial_worker.error.connect(self.on_serial_error)
        self.serial_worker.start()
    
    def on_read_sensor_finished(self):
        """Обработка завершения чтения датчика"""
        self.show_status_message("Датчик прочитан, получаем данные... Можно перевернуть магнит", timeout=60000)
        QTimer.singleShot(500, self.get_data)
        
    
    def get_data(self):
        """Получение данных после чтения датчика"""
        self.serial_worker = SerialWorker(self.sensor_port, 'S', 46, 4194304)
        self.serial_worker.data_ready.connect(self.on_data_received)
        self.serial_worker.error.connect(self.on_serial_error)
        self.serial_worker.finished.connect(self.on_get_data_finished)
        self.serial_worker.start()
    
    def on_data_received(self, raw_data):
        """Обработка полученных данных"""
        try:
            if len(raw_data) != 4194304:
                QMessageBox.warning(self, "Считывание", "Недостаточно данных с датчика. Проверьте подключение.")
            df_raw = self.data_processor.process_raw_data(raw_data)
            df_filtered = self.data_processor.apply_median_filter(df_raw, window_size=3)
            df_truncated = self.data_processor.truncate_marginal_periods(df_filtered)
            self.df = self.data_processor.integrate_df(df_truncated)

            amplitude, phase, phase_deg = self.data_processor.get_amplitude(self.df)

            self.update_graph(amplitude, phase)
            
            # Сохраняем данные текущего измерения (для возможного повтора)
            self.measurement_manager.save_current_measurement_data(amplitude, phase_deg)
            
            # Обновляем интерфейс с результатом текущего измерения
            current_idx = self.measurement_manager.current_measurement
            if current_idx < len(self.measurement_labels):
                self.measurement_labels[current_idx].setText(
                    f"Измерение {current_idx + 1}: {amplitude:.3e} [В*с*м], фаза {phase_deg:.1f}°"
                )
            
            self.show_status_message(f'Измерение {current_idx + 1}/3 завершено!')

        except Exception as e:
            self.on_serial_error(f"Ошибка обработки данных: {str(e)}")
    
    def on_get_data_finished(self):
        """Обработка завершения получения данных"""
        # После завершения измерения спрашиваем пользователя
        self.ask_measurement_action()

    def save_data(self):
        """Сохраняет заголовок и результат измерения в выбранный пользователем файл."""
        final_results = self.measurement_manager.get_final_result()
        if not final_results:
            QMessageBox.warning(self, "Сохранение", "Нет завершённых измерений для сохранения.")
            return

        # Получаем текст из заголовка
        header_text = self.txtEd_FileHeader.toPlainText().strip()
        if not header_text:
            header_text = "Empty"

        amplitude, theta_deg = final_results

        # Открываем диалог сохранения
        file_path, type = QFileDialog.getSaveFileName(
            self,
            caption = "Добавить результаты в файл",
            directory = "measurements",  # начальное название файла
            filter = "Текст (*.txt);;Таблица (*.csv)",
            initialFilter = "Таблица (*.csv)",
            options = QFileDialog.DontConfirmOverwrite
        )

        if file_path:
            if type == "Текст (*.txt)":
                result_line = f"Полный момент: {amplitude:.4} [В*с*м]; Отклонение от нормали θz: {theta_deg:.2f}°"
                full_content = f"{header_text}\n{result_line}\n" + "=" * 60 + "\n"

                try:
                    # Если файл уже существует — добавляем в начало, иначе создаём новый
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            existing = f.read()
                        full_content = full_content + existing

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(full_content)

                    self.show_status_message(f"Результат добавлен в {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
            elif type == "Таблица (*.csv)":
                    try:
                        # Подготавливаем строку данных
                        header_text = self.txtEd_FileHeader.toPlainText().strip().replace('\n', ' | ').replace(',', ';')  # убираем переносы и запятые
                        new_row = pd.DataFrame([{
                            "Магнит": header_text,
                            "Момент": amplitude,
                            "Угол": theta_deg
                        }])

                        # Проверяем, существует ли файл
                        file_exists = os.path.isfile(file_path)

                        # Записываем: если файл есть — без заголовка, иначе — с заголовком
                        new_row.to_csv(
                            file_path,
                            mode='a' if file_exists else 'w',
                            header=not file_exists,
                            index=False,
                            encoding='utf-8'
                        )

                        self.show_status_message(f"Результат добавлен в {file_path}")
                    except Exception as e:
                        QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить CSV:\n{str(e)}")

    def ask_measurement_action(self):
        """Спросить пользователя о дальнейших действиях"""
        current_meas_num = self.measurement_manager.get_current_measurement_number()
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Измерение завершено")
        msg.setText(f"Измерение {current_meas_num}/3 завершено!\n\nПоверните магнит в следующее положение или надо повторить измерение?")
        msg.setIcon(QMessageBox.Question)
        
        # Добавляем кнопки
        if current_meas_num < 3:
            next_btn = msg.addButton("Следующее положение", QMessageBox.AcceptRole)
        else:
            next_btn = msg.addButton("Завершить", QMessageBox.AcceptRole)
        repeat_btn = msg.addButton("Повторить", QMessageBox.ActionRole)
        cancel_btn = msg.addButton("Отмена цикла", QMessageBox.RejectRole)

        self.set_wait_cursor(False)
        msg.setDefaultButton(next_btn)

        msg.exec_()
        
        self.set_wait_cursor(True)

        clicked_button = msg.clickedButton()
        if clicked_button == next_btn:
            # Подтвердить текущее измерение и перейти к следующему
            self.measurement_manager.confirm_current_measurement()
            
            if self.measurement_manager.is_complete():
                # Все измерения завершены
                self.measurement_cycle_complete()
            else:
                # Перейти к следующему измерению
                self.show_status_message(f"Следующее измерение...")
                QTimer.singleShot(500, self.start_single_measurement)

        elif clicked_button == repeat_btn:
            # Повторить измерение
            self.show_status_message(f"Повтор измерения {current_meas_num}...")
            QTimer.singleShot(500, self.start_single_measurement)
                
        elif clicked_button == cancel_btn:
            # Отменить цикл измерений
            self.measurement_cycle_complete(aborted=True)
    
    def measurement_cycle_complete(self, aborted=False):
        """Завершение цикла измерений"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
        self.progressBar.setVisible(False)
        
        self.pBtn_GetData.setVisible(True)
        self.portsFrame.setVisible(True)

        if aborted:
            self.progressLabel.setText("Цикл измерений остановлен")
            self.show_status_message("Цикл измерений остановлен пользователем")
        else:
            self.progressLabel.setText("Цикл измерений завершён")
            
            # Показываем финальный результат
            final_results = self.measurement_manager.get_final_result()
            if final_results:
                amplitude, theta_deg = final_results
                self.lbl_finalResult.setText(
                    f"Полный момент: {amplitude:.3} [В*с*м]; Отклонение от нормали θz: {theta_deg:.2f}°"
                )
                self.show_status_message(f"Цикл измерений завершён! Полный момент: {amplitude:.4}; Отклонение θz: {theta_deg:.3f}°")
            self.save_data()
                
    def on_serial_error(self, error_msg):
        """Обработка ошибок последовательного порта"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
        self.progressBar.setVisible(False)
        QMessageBox.critical(self, "Error", error_msg)
        print(f"Error: {error_msg}")

    def update_graph(self, amp, phase):
        """Обновление графика текущими данными"""
        x = self.df.index/10000*360
        y = amp * np.sin(2 * np.pi * 1/360 * x + phase)


        self.plot_widget.clear()
        self.plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2), name="Sine Data")
        self.plot_widget.plot(x, self.df.volts, pen=pg.mkPen(color='r', width=1), name="Raw Data")

    def update_graph2(self, amp, phase):
        """Обновление графика текущими данными"""
        self.chart.removeAllSeries()
        data_series = QLineSeries()

        x = self.df.index.values/10000*360
        y = amp * np.sin(2 * np.pi * 1/360 * x + phase)

        data_series.append(x, y)

        # for a, b in zip(x, y):
        #     data_series.append(a, b)
        
        self.chart.addSeries(data_series)
        self.chart.createDefaultAxes()
        self.chart.legend().setVisible(False)
        self.chart.axisX().setLabelFormat("%d")
        self.chart.axisY().setLabelFormat("%.1e")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)

    def show_status_message(self, message, timeout=5000):
        """Показать сообщение в статус баре"""
        self.statusBar.showMessage(message, timeout)

    def settings(self):
        """Открывает окно настроек для изменения скорости двигателя."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Настройки")

        layout = QFormLayout()

        speed_input = QLineEdit()
        speed_input.setText(str(self.motor_speed))
        layout.addRow("Скорость двигателя:", speed_input)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            try:
                new_speed = int(speed_input.text())
                if 1 <= new_speed <= 1000:  # допустимый диапазон для MOVE F(...)
                    self.motor_speed = new_speed
                    self.show_status_message(f"Скорость двигателя установлена: {self.motor_speed}")
                else:
                    QMessageBox.warning(self, "Ошибка", "Скорость должна быть от 1 до 1000.")
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "Введите корректное целое число.")

if __name__ == '__main__':
    app = QApplication([])
    
    def exception_handler(exctype, value, traceback):
        print(f"Unhandled exception: {exctype.__name__}: {value}")
        QMessageBox.critical(None, "Error", f"An error occurred:\n{value}")
    
    sys.excepthook = exception_handler
    
    window = MainUI()
    window.show()
    
    sys.exit(app.exec_())