from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
import sys, datetime
import serial.tools.list_ports
# import minimalmodbus
# from icecream import ic
import time

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        loadUi("ReadData.ui", self)

        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.cBox_COMPort.addItem(port.device)

        ''' Привязка кнопок '''
        # Движение ротора
        self.pBtn1_Read.clicked.connect(self.ReadData)
        self.pBtn2_Get.clicked.connect(self.GetData)
        self.pBtn3_Show.clicked.connect(self.ShowData)
        
    def ReadData(self) -> None:
        port = self.cBox_COMPort.currentText()
        with (serial.Serial(port, baudrate=921600)) as self.serialData:

            # Read data from COM port
            command = 'R'

            # Send the command to the DataPort
            self.serialData.write(command.encode())
        time.sleep(3)

    def GetData(self) -> None:
        port = self.cBox_COMPort.currentText()
        with (serial.Serial(port, baudrate=921600, timeout=3)) as self.serialData:

            # Read data from COM port
            command = 'S'

            # Send the command to the DataPort
            self.serialData.write(command.encode())
            # line2 = self.serialData.readline()
            # line1 = self.serialData.read(262144)
            line = self.serialData.read(262144)
        print(len(line))
        self.line = []
        for num in range(0, len(line), 2):
            hi_byte = line[num]
            hi_byte = hi_byte if hi_byte < 128 else hi_byte-256
            lo_byte = line[num+1]    
            self.line.append(hi_byte*256+lo_byte)
        print(len(self.line))

        with open("data.txt", "w") as f:
            f.write(str(self.line))

    def ShowData(self) -> None:
        if hasattr(self, 'chart_view'):
            self.chart_view.deleteLater()
        if not hasattr(self, 'line'):
            self.line = [10114, 11174, 11395, 11176, 11394, 11203, 11397, 11180, 11395, 11172, 10114, 11174, 11395, 11176, 11394, 11203, 11397, 11180, 11395, 11172]
        # Создаем график
        self.chart = QChart()
        self.chart.setTitle("График числовых данных")
        self.chart.legend().setVisible(False)
        
        # Создаем серию данных
        series = QLineSeries()
        
        # Добавляем точки данных
        for i, value in enumerate(self.line[6:]):
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
        self.vFrameLayout.addWidget(self.chart_view)

    def ShowData(self) -> None:
        if hasattr(self, 'chart_view'):
            self.chart_view.deleteLater()
        if not hasattr(self, 'line'):
            self.line = [10114, 11174, 11395, 11176, 11394, 11203, 11397, 11180, 11395, 11172, 10114, 11174, 11395, 11176, 11394, 11203, 11397, 11180, 11395, 11172]
        # Создаем график
        self.chart = QChart()
        self.chart.setTitle("График числовых данных")
        self.chart.legend().setVisible(False)
        
        # Создаем серию данных
        series = QLineSeries()
        
        # Добавляем точки данных
        for i, value in enumerate(self.line[6:]):
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
        self.vFrameLayout.addWidget(self.chart_view)

                
        
if __name__ == '__main__':
    # app = QApplication(sys.argv)
    app = QApplication([])
    
    coils = MainUI()
    coils.show()
    
    # app.exec_()
    sys.exit(app.exec_())