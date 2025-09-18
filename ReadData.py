from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
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

        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.cBox_COMPort.addItem(port.device)

        ''' Привязка кнопок '''
        self.pBtn1_Read.clicked.connect(self.ReadData)
        self.pBtn2_Get.clicked.connect(self.GetData)
        self.pBtn3_Show.clicked.connect(self.ShowData)
        
    def ReadData(self) -> None:
        port = self.cBox_COMPort.currentText()
        with (serial.Serial(port, baudrate=921600, bytesize=8, stopbits=1, timeout=11)) as self.serialData:

            # Read data from coils and encoder
            command = 'R'

            # Send the command to the DataPort
            self.serialData.write(command.encode())
        time.sleep(11)
        print('Ok')

    def GetData(self) -> None:
        port = self.cBox_COMPort.currentText()
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

    def ShowData(self) -> None:
        if hasattr(self, 'chart_view'):
            self.chart_view.deleteLater()
        if not hasattr(self, 'df'):
            self.df = pd.read_csv(os.path.join('data','data1.csv'))
        # Создаем график
        self.chart = QChart()
        self.chart.setTitle("График числовых данных")
        self.chart.legend().setVisible(False)
        
        # Создаем серию данных
        series = QLineSeries()
        
        # Добавляем точки данных
        window_size = 3
        data = signal.medfilt(self.df.data, kernel_size=window_size)
        
        for i, value in enumerate(data):
            series.append(i, value)
        
        # Добавляем серию на график
        self.chart.addSeries(series)
        
        # Настраиваем оси
        axis_x = QValueAxis()
        axis_x.setTitleText("Индекс")
        axis_x.setLabelFormat("%d")
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setTitleText("Значение")
        axis_y.setLabelFormat("%d")
        self.chart.addAxis(axis_y, Qt.AlignLeft)
        
        series.attachAxis(axis_y)
        
        # Создаем виджет для отображения графика
        self.chart_view = QChartView(self.chart)
        # chart_view.setRenderHint(QChartView.Antialiasing)
        
        # Устанавливаем центральный виджет
        self.hLayout.addWidget(self.chart_view)
                
        
if __name__ == '__main__':
    # app = QApplication(sys.argv)
    app = QApplication([])
    
    coils = MainUI()
    coils.show()
    
    # app.exec_()
    sys.exit(app.exec_())