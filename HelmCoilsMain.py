# --- НАЧАЛО: Пользовательские переменные ---
# Параметры вращения мотора
MOTOR_REVOLUTIONS = 9          # Количество оборотов мотора за измерение
MOTOR_DISTANCE_FACTOR = 115     # Фактор расстояния на один оборот мотора
MOTOR_DEFAULT_SPEED = 100       # Скорость мотора по умолчанию

# Параметры соединения
MOTOR_SERIAL_BAUDRATE = 57600
SENSOR_SERIAL_BAUDRATE = 2000000
SENSOR_SERIAL_BAUDSPEED = 100000 # количество выборок/c, целое от 1 до 300000
DATA_READ_SIZE = 400000  # Размер данных для чтения с датчика (в байтах)
DATA_READ_SIZE_STEP = 1000  # Размер окна данных для чтения с датчика (в байтах)

# Параметры калибровки/пересчета
ADC_VOLT_REFERENCE = 2.5        # Опорное напряжение АЦП [В]
ADC_BIT_COUNT = 32767           # Максимальное значение АЦП (±16 бит)
TIMEBASE_CONSTANT = 100000       # Постоянная времени системы [мкс]
COIL_CONSTANT = 1144.8          # Постоянная катушки [1/м]
ENCODER_PULSES_PER_REV = 10000  # Количество импульсов энкодера на оборот

# --- КОНЕЦ: Пользовательские переменные ---

from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
import sys, os, datetime
import serial.tools.list_ports
import pandas as pd
import numpy as np
import pyqtgraph as pg
import fastgoertzel as fg
from scipy.signal import medfilt
import crcmod

class DataProcessor:
    """Класс для обработки данных"""
    @staticmethod
    def process_raw_data(ADC: bytes, EDC: bytes) -> pd.DataFrame:
        """Перевод байтовой строки в числа"""
        rawADC = np.frombuffer(ADC, dtype=np.uint8)
        rawEDC = np.frombuffer(EDC, dtype=np.uint8)

        # Чтение по два байта (старший-младший) в нотации "big-endian" с переводом в нотацию компилятора (little-endian)
        data = rawADC.view(dtype='>i2').astype(np.int16)
        encoder = rawEDC.view(dtype='>i2').astype(np.int16)

        return pd.DataFrame({'encoder': encoder, 'data': data})
    
    @staticmethod
    def apply_median_filter(df: pd.DataFrame, window_size:int=3) -> pd.DataFrame:
        """Применение медианного фильтра для удаления дельта-выбросов"""
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
        split_points = df.index[diff_enc.abs() > 1000]  # по модулю, чтобы не зависеть от направления

        if len(split_points) in (0, 1):
            return df
        else:
            start_idx = split_points[0]
            end_idx = split_points[-1]
            # Проверка, чтобы start_idx был меньше end_idx
            if start_idx >= end_idx:
                return df
            else:
                return df.iloc[start_idx:end_idx].copy()

    @staticmethod
    def integrate_df(df_trimmed: pd.DataFrame) -> pd.DataFrame:
        """Полное интегрирование данных - основной метод"""
        # Усредняем по значениям encoder и вычисляем интеграл по всему периоду данных
        # 1. Группировка по одинаковым значениям encoder
        df_trimmed['period'] = (df_trimmed['encoder'] != df_trimmed['encoder'].shift()).cumsum()

        # 2. Группируем по периоду, затем вычисляем среднее, после чего сбрасываем индекс и период
        df_res = df_trimmed.groupby('period').agg({'data': 'sum', 'encoder': 'first', 'period': 'first'}).reset_index(drop=True)

        # 3 Интеграл (кумулятивная сумма)
        df_res['integral'] = -1.0*df_res.data.cumsum()

        # 4. Пересчет в Вольты*метры*секунды
        #  ADC_VOLT_REFERENCE/ADC_BIT_COUNT - коэф. для перевода в Вольты, 1/TIMEBASE_CONSTANT в сек (timebase), 1/COIL_CONSTANT в м (постоянная катушки)
        df_res['volts'] = (ADC_VOLT_REFERENCE/ADC_BIT_COUNT / TIMEBASE_CONSTANT / COIL_CONSTANT)*df_res['integral']

        # 5. Детренд
        x = df_res.index.values
        y = df_res.volts.values
        # Координаты первой и последней точек
        x0, x1 = x[0], x[-1]
        y0, y1 = y[0], y[-1]
        # Уравнение прямой через две точки: y_trend = a * x + b
        a = (y1 - y0) / (x1 - x0) if x1 != x0 else 0
        b = y0 - a * x0
        # Вычисляем трендовую составляющую и вычитаем
        y_trend = a * x + b
        df_res['detrend'] = y - y_trend

        # 6. Угол поворота в градусах
        df_res['deg'] = (df_res['period']/ENCODER_PULSES_PER_REV)*360

        return df_res

    @staticmethod
    def get_amplitude(df: pd.DataFrame) -> tuple:
        """Вычисление амплитуды алгоритмом Гёрцеля"""
        norm_freq = 1 / ENCODER_PULSES_PER_REV
        f_amp, f_phase = fg.goertzel(df['detrend'].values, norm_freq)
        # сдвиг рассчитанной фазы на +π/2 и перевод в градусы
        f_phase_deg = np.degrees(f_phase+np.pi/2)

        return (f_amp, f_phase+np.pi/2, f_phase_deg)

class MotorController:
    """Класс для управления мотором"""
    
    @staticmethod
    def run_motor(port, distance=MOTOR_REVOLUTIONS*MOTOR_DISTANCE_FACTOR, speed=MOTOR_DEFAULT_SPEED):
        """Запуск мотора на определенное количество оборотов"""
        try:
            with serial.Serial(port, baudrate=MOTOR_SERIAL_BAUDRATE, bytesize=8, 
                             parity='N', stopbits=1, timeout=0) as serial_conn:
                
                command = f'ON\rMOVE L(-{int(distance)})F({int(speed)})\rOFF\r' # MOVE L(15*115)F(100)
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
        
        # Извлекаем амплитуды из всех трех измерений и сортируем
        sorted_results = sorted([m for m in self.measurements], key = lambda measure: measure['amplitude'])
        
        # Вычисляем финальный результат модуля амплитуды
        sum_of_squares = sum(result['amplitude']**2 for result in sorted_results) / 2
        final_amplitude = np.sqrt(sum_of_squares)
        
        # Угол отклонения от нормали (оси z)
        M_xy = sorted_results[0]['amplitude']
        M_yz = sorted_results[1]['amplitude']
        M_zx = sorted_results[2]['amplitude']

        theta_rad = np.arctan(M_xy / (np.sqrt(M_yz**2 + M_zx**2 - M_xy**2)/2))
        theta_deg = np.degrees(theta_rad)

        # Фаза проекции момента магнита на плоскость xy
        phase_xy = sorted_results[0]['phase']

        return (final_amplitude, theta_deg, phase_xy)
    
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

        self.motor_speed = MOTOR_DEFAULT_SPEED
        
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
        self.pBtn_refreshGraph.clicked.connect(self.update_graph)
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
        
        # Скрываем прогресс бар изначально
        self.progressBar.setVisible(False)
        
        self.update_buttons_state(True)
        self.show_status_message("Готов к работе")
        # Инициализация портов
        QTimer.singleShot(100, self.refresh_ports)

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
            try:
                with serial.Serial(port=port.device, baudrate=MOTOR_SERIAL_BAUDRATE, bytesize=8, parity='N', stopbits=1, timeout=1) as serialData:
                    
                    command = f'ON\rSHOW inp1\rOFF\r' # SHOW inp1 - выводит состояние входа inp1 контроллера двигателя
                    serialData.write(command.encode("utf-8"))
                    motor_answer = serialData.readline()
                    if motor_answer:
                        self.motor_port = port.device
                        self.cBox_MotorPort.setCurrentText(port.device)
            except Exception as e:
                self.on_serial_error(f"Ошибка чтения порта мотора: {str(e)}")
            
            self.cBox_SensorPort.addItem(port.device)
            try:
                with serial.Serial(port=port.device, baudrate=SENSOR_SERIAL_BAUDRATE, bytesize=8, parity='N', stopbits=1, timeout=1) as serialData:
                    command = f'R0;1\n'
                    serialData.write(command.encode())
                    dataRead = serialData.readline()
                    dataReady = serialData.readline()
                    if dataReady == b'1\n':
                        self.sensor_port = port.device
                        self.cBox_SensorPort.setCurrentText(port.device)
            except Exception as e:
                print(f"Ошибка чтения порта датчика: {str(e)}")
                # self.on_serial_error(f"Ошибка чтения порта датчика: {str(e)}")
        
        if not hasattr(self, 'motor_port'):
            self.on_serial_error(f"Ошибка чтения порта двигателя")
        if not hasattr(self, 'sensor_port'):
            self.on_serial_error(f"Ошибка чтения порта датчика")

    def init_graph(self):
        """Инициализация графика"""
        self.plot_widget = pg.PlotWidget(self)
        self.chartLayout.addWidget(self.plot_widget)

        # Настраиваем внешний вид
        self.plot_widget.setBackground('w') # Белый фон
        # self.plot_widget.setTitle("Проекция момента")
        self.plot_widget.setLabel('left', 'Проекция момента (В⋅с⋅м)')
        self.plot_widget.setLabel('bottom', 'Угол (°)')
        self.plot_widget.showGrid(x=True, y=True)

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
        
        QTimer.singleShot(2000, self.read_sensor) # Через 2 секунды начинаем считывать данные
    
    def read_sensor(self):
        """Чтение датчика после запуска мотора"""
        try:
            with (serial.Serial(port=self.sensor_port, baudrate=SENSOR_SERIAL_BAUDRATE, bytesize=8, stopbits=1, timeout=None)) as serialData:
                # Read data from Sensor
                command = f'R{SENSOR_SERIAL_BAUDSPEED};{DATA_READ_SIZE}\n'
                # Send the command to the DataPort
                serialData.write(command.encode())
                dataRead = serialData.readline()
                dataReady = serialData.readline()
        except Exception as e:
            self.on_serial_error(f"Ошибка чтения датчика: {str(e)}")
        if dataReady == b'1\n':
            TIMEBASE_CONSTANT = int(float(dataRead.decode().split('F=')[1].split()[0].replace(',','.'))*1000)  # Постоянная времени системы [мкс]
            self.on_read_sensor_finished()

    def on_read_sensor_finished(self):
        """Обработка завершения чтения датчика"""
        self.show_status_message("Измерение проведено, получаем данные... Можно перевернуть магнит", timeout=60000)
        QTimer.singleShot(500, self.get_data)
    
    def get_data(self):
        """Получение данных после чтения датчика"""
        ADC = b''
        EDC = b''
        max_retries = 3
        # Создаём объект CRC-16-CCITT-ZERO
        crc16_func = crcmod.mkCrcFun(
            poly=0x11021,      # Полином: x^16 + x^12 + x^5 + 1 (0x1021, но с битом переноса)
            initCrc=0x0000,  # Начальное значение — 0x0000 (ZERO)
            xorOut=0x0000,   # Окончательный XOR — 0x0000
            rev=False          # Прямой порядок битов (normal)
        )
        try:
            with (serial.Serial(port=self.sensor_port, baudrate=SENSOR_SERIAL_BAUDRATE, bytesize=8, stopbits=1, timeout=1)) as serialData:
                for i in range(0, DATA_READ_SIZE, DATA_READ_SIZE_STEP):

                    for retry in range(max_retries):
                        retry_read = False
                        ADCsuccess = False
                        EDCsuccess = False

                        command = f'S{i};{DATA_READ_SIZE_STEP}\n'
                        serialData.write(command.encode())

                        lineADC = serialData.read(2*DATA_READ_SIZE_STEP+9)
                        dataADC = lineADC[6:-3]
                        crcADC = lineADC[-3:-1]
                        if crc16_func(dataADC) == int.from_bytes(crcADC, 'big'):
                            ADCsuccess = True
                        else:
                            # self.on_serial_error(f"Ошибка контрольной суммы данных датчика")
                            print('DataADC not ok')
                            retry_read = True

                        lineEDC = serialData.read(2*DATA_READ_SIZE_STEP+9)
                        dataEDC = lineEDC[6:-3]
                        crcEDC = lineEDC[-3:-1]
                        if crc16_func(dataEDC) == int.from_bytes(crcEDC, 'big'):

                            EDCsuccess = True
                        else:
                            # self.on_serial_error(f"Ошибка контрольной суммы данных энкодера")
                            print('DataEDC not ok')
                            retry_read = True
                        if not retry_read:
                            break

                    if (ADCsuccess and EDCsuccess):
                        ADC += dataADC
                        EDC += dataEDC
                    else:
                        self.on_serial_error(f"Ошибка cчитывания данных с порта датчика")
        except Exception as e:
            self.on_serial_error(f"Ошибка cчитывания данных с порта датчика: {str(e)}")

        if len(ADC) == len(EDC):
            self.on_data_received(ADC, EDC)
            self.on_get_data_finished()
        else:
            QMessageBox.warning(self, "Считывание", "Данные с датчика не совпадают с данными энкодера. Проверьте подключение.")
    
    def on_data_received(self, ADC, EDC):
        """Обработка полученных данных"""
        try:
            df_raw = self.data_processor.process_raw_data(ADC, EDC)
            df_filtered = self.data_processor.apply_median_filter(df_raw, window_size=3)
            df_truncated = self.data_processor.truncate_marginal_periods(df_filtered)
            self.df = self.data_processor.integrate_df(df_truncated)

            amplitude, phase, phase_deg = self.data_processor.get_amplitude(self.df)

            self.update_graph()
            
            # Сохраняем данные текущего измерения (для возможного повтора)
            self.measurement_manager.save_current_measurement_data(amplitude, phase_deg)
            
            # Обновляем интерфейс с результатом текущего измерения      
            current_idx = self.measurement_manager.current_measurement
            if current_idx < len(self.measurement_labels):
                self.measurement_labels[current_idx].setText(
                    f"Измерение {current_idx + 1}: {amplitude:.3e} [Вб⋅м]" #, фаза {phase_deg:.1f}°
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

        amplitude, theta_deg, phase_xy = final_results

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
                result_line = f"Полный момент: {amplitude:.3e} [Вб⋅м]; Отклонение от нормали θz: {theta_deg:.1f}°, азимут φ: {phase_xy:.1f}°"
                full_content = f"{header_text}\n{result_line}\n" + "=" * 80 + "\n"

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
                            "Момент": amplitude,
                            "Угол": theta_deg,
                            "Азимут": phase_xy,
                            "Магнит": header_text
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
                amplitude, theta_deg, phase_xy = final_results
                self.lbl_finalResult.setText(
                    f"Полный момент: {amplitude:.3e} [Вб⋅м]; Отклонение от нормали θz: {theta_deg:.2f}°, азимут φ: {phase_xy:.2f}°"
                )
                self.show_status_message(f"Цикл измерений завершён! Полный момент: {amplitude:.4e}; Отклонение θz: {theta_deg:.2f}°, азимут φ: {phase_xy:.2f}°")
            self.save_data()
                
    def on_serial_error(self, error_msg):
        """Обработка ошибок последовательного порта"""
        self.update_buttons_state(True)
        self.set_wait_cursor(False)
        self.progressBar.setVisible(False)
        QMessageBox.critical(self, "Error", error_msg)
        print(f"Error: {error_msg}")

    def update_graph(self):
        """Обновление графика текущими данными"""
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "График", "Нет завершённых измерений для отрисовки графика.")
            return
        x = self.df.deg
        y = self.df.detrend

        self.plot_widget.clear()
        self.plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=3), name="Магнитный поток")
        self.plot_widget.autoRange()

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