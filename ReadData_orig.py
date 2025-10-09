from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
from PyQt5.uic import loadUi
# from PyQt5.QtCore import Qt
# from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPainter
import sys, datetime
import serial.tools.list_ports
# import minimalmodbus
# from icecream import ic
import time, os
import pandas as pd
from scipy import signal
import numpy as np

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        loadUi("ReadData.ui", self)
        self.InitGraph()

        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.cBox_MotorPort.addItem(port.device)
            self.cBox_SensorPort.addItem(port.device)

        ''' Привязка кнопок '''
        self.pBtn_Read.clicked.connect(self.ReadSensor)
        self.pBtn_Get.clicked.connect(self.GetData)
        self.pBtn_Int.clicked.connect(self.IntegrateData)
        self.pBtn_Motor.clicked.connect(self.runMotor)
        self.pBtn_MotorStart.clicked.connect(self.startMotor)
        self.pBtn_Show.clicked.connect(self.ShowData)
        
    def ReadSensor(self) -> None:
        '''
        
        '''
        port = self.cBox_SensorPort.currentText()
        with (serial.Serial(port, baudrate=921600, bytesize=8, stopbits=1, timeout=11)) as self.serialData:

            # Read data from coils and encoder
            command = 'R'

            # Send the command to the DataPort
            self.serialData.write(command.encode())
        time.sleep(11)
        print('Sensor read ok, now get it!')

    def GetData(self) -> None:
        port = self.cBox_SensorPort.currentText()
        with (serial.Serial(port, baudrate=921600, bytesize=8, stopbits=1, timeout=47)) as self.serialData:

            # Get data from coils and encoder
            command = 'S'

            # Send the command to the DataPort
            self.serialData.write(command.encode())
            # Read data
            rawdata = self.serialData.read(4194305)

        data = []
        for num in range(0, 2097152, 2):
            hi_byte = rawdata[num]
            hi_byte = hi_byte if hi_byte < 128 else hi_byte-256
            lo_byte = rawdata[num+1]    
            data.append(hi_byte*256+lo_byte)

        encoder = []
        for num in range(2097152, len(rawdata)-1, 2):
            hi_byte = rawdata[num]
            hi_byte = hi_byte if hi_byte < 128 else hi_byte-256
            lo_byte = rawdata[num+1]    
            encoder.append(hi_byte*256+lo_byte)
        self.df = pd.DataFrame({'encoder' : encoder, 'data' : data})
        print('Data obtained! Time to integrate!')

    def InitGraph(self) -> None:
        # Создаем график
        self.chart = QChart()
        self.chart.setTitle("Числовые данные")
        self.chart.legend().setVisible(False)
        
        # Создаем серию данных
        self.series = QLineSeries()

        # # Добавляем точки данных
        # for index, data in enumerate(range(10)):
        #     self.series << QPointF(index, data)

        # self.series << QPointF(11, 1) << QPointF(12, 3) << QPointF(13, 6) 


        # Добавляем серию на график
        self.chart.addSeries(self.series)
        self.chart.createDefaultAxes()
        
        # Настраиваем оси
        axis_x = self.chart.axisX()
        axis_x.setTitleText("Сигнал")
        axis_x.setLabelFormat("%d")
        # self.chart.addAxis(axis_x, Qt.AlignBottom)
        # self.series.attachAxis(axis_x)
        
        axis_y = self.chart.axisY()
        axis_y.setTitleText("Индекс")
        axis_y.setLabelFormat("%d")
        # self.chart.addAxis(axis_y, Qt.AlignLeft)
        # self.series.attachAxis(axis_y)
        
        # Создаем виджет для отображения графика
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        
        # Устанавливаем центральный виджет
        self.Layout_3h.addWidget(self.chart_view)
    
    def ShowData(self) -> None:
        if not hasattr(self, 'df'):
            self.df = pd.read_csv(os.path.join('data','data1.csv'), index_col=0)
        
        # Применяем медианный фильтр из scipy
        # window_size = 3
        # df_filtered = pd.DataFrame(columns=['encoder', 'data'])
        # df_filtered.data = signal.medfilt(self.df.data, kernel_size=window_size)
        # df_filtered.encoder = signal.medfilt(self.df.encoder, kernel_size=window_size)
        
        self.series = QLineSeries()

        # Добавляем точки данных
        # for index, row in self.df.iterrows():
        #     self.series.append(index, row['data'])
        for row in self.df.itertuples():
            self.series.append(row.Index, row.data)

        # self.chart.zoomReset()
        
        # Добавляем серию на график
        self.chart.removeAllSeries()
        self.chart.addSeries(self.series)

        # Настраиваем оси
        # axis_x = self.chart.axisX()
        # axis_x.setTitleText("Сигнал")
        # axis_x.setLabelFormat("%d")
        # self.chart.addAxis(axis_x, Qt.AlignBottom)
        # self.series.attachAxis(axis_x)
        
        # axis_y = self.chart.axisY()
        # axis_y.setTitleText("Значение")
        # axis_y.setLabelFormat("%d")
        # self.chart.addAxis(axis_y, Qt.AlignLeft)
        # self.series.attachAxis(axis_y)

    def IntegrateData(self) -> None:
        if not hasattr(self, 'df'):
            self.df = pd.read_csv(os.path.join('data','data1.csv'), index_col=0)

        # Применяем медианный фильтр из scipy
        window_size = 3
        df = pd.DataFrame(columns=['encoder', 'data'])
        df.data = signal.medfilt(self.df.data, kernel_size=window_size)
        df.encoder = signal.medfilt(self.df.encoder, kernel_size=window_size)

        # Находим индексы, где происходит переход с 9999 на 0
        split_points = df.index[(df['encoder'].shift(1) == 9999) & (df['encoder'] == 0)]

        # Создаем список разделенных датафреймов с группировкой по encoder
        datasets = []
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            dataset_part = df.iloc[start_idx:end_idx].copy()
            # print(dataset_part.max())
            # Группировка по encoder и усреднение data
            datasets.append(dataset_part.groupby('encoder', as_index=False)['data'].mean())


        # Шаг интегрирования по encoder
        step = 20
        # Создание интервалов
        bins = range(0, 9999 + step, step)

        df_integral = pd.DataFrame(index=range(len(bins)-1))
        
        for i, dataset in enumerate(datasets):
            # Добавление столбца с номером интервала
            dataset['bin'] = pd.cut(dataset['encoder'], bins=bins, right=False, labels=bins[:-1])
            # Группировка по интервалам и усреднение данных
            grouped = dataset.groupby('bin', observed=False)['data'].mean().reset_index()
            # grouped['bin'] = grouped['bin'].astype(int)
            # Последовательное интегрирование (накопленная сумма)
            # grouped['integral'] = (grouped['data'] * step).cumsum()
            # print(grouped)

            # Вводим линейный корректирующий множитель
            line = (grouped['data'] * step).cumsum()
            # Вычисляем разницу
            difference = line[0] - line[len(line)-1]
            # Создаем корректирующий множитель (линейный)
            n = len(line)
            corrected_data = []
            
            for j, value in enumerate(line):
                correction = (difference * j) / (n - 1)
                corrected_data.append(value + correction)

            # Добавляем интегральные данные в датафрейм
            df_integral = df_integral.join(pd.DataFrame({f'integral_data_{i+1}': corrected_data}))

        self.df = pd.DataFrame(columns = ['encoder', 'data'], index=range(500))
        self.df.encoder = range(500)
        self.df.data = df_integral.mean(axis=1)

    def runMotor(self, checked=False, revolutions:int=1, distance:int=int(1000/9), speed:int=50) -> None:
        portMotor = self.cBox_MotorPort.currentText()
        with (serial.Serial(portMotor, baudrate=57600, bytesize=8, parity='N', stopbits=1, timeout=0)) as serialData:
            command = f'ON\rMOVE L({int(revolutions*distance)})F({int(speed)})\rOFF\r'
            print(serialData.write(command.encode(encoding="utf-8")))

    def startMotor(self, checked=False, direction:int=1, speed:int=50) -> None:
        portMotor = self.cBox_MotorPort.currentText()
        with (serial.Serial(portMotor, baudrate=57600, bytesize=8, parity='N', stopbits=1, timeout=0)) as serialData:
            command = f'ON\rMOVE C(inp1)D({direction})F({speed})\rOFF\r'
            print(serialData.write(command.encode(encoding="utf-8")))

    def stopMotor(self, checked=False, direction:int=1, speed:int=10) -> None:
        pass
        '''
        portMotor = self.cBox_MotorPort.currentText()
        with (serial.Serial(portMotor, baudrate=57600, bytesize=8, parity='N', stopbits=1, timeout=0)) as serialData:
            command = f'SET inp1, 0\r'
            serialData.write(command.encode(encoding="utf-8"))
            command = f'SHOW inp1\r'
            print(serialData.write(command.encode(encoding="utf-8")))
            print(serialData.read())
        '''

if __name__ == '__main__':
    # app = QApplication(sys.argv)
    app = QApplication([])
    
    coils = MainUI()
    coils.show()
    
    # app.exec_()
    sys.exit(app.exec_())