from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
import sys, datetime
import serial.tools.list_ports
# import minimalmodbus
# from icecream import ic
import time

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        loadUi("ReadData.ui", self)

        print(list(serial.tools.list_ports.comports(include_links=False)))
        # print(serial.tools.list_ports.ListPortInfo())

        ''' Привязка кнопок '''
        # Движение ротора
        self.pBtn1_Read.clicked.connect(self.ReadData)
        self.pBtn2_Get.clicked.connect(self.GetData)
        
       
    def ReadData(self) -> None:
        with (serial.Serial('COM5', baudrate=921600)) as self.serialData:

            # Read data from COM port
            command = 'R'

            # Send the command to the DataPort
            self.serialData.write(command.encode())

    def GetData(self) -> None:
        line = 0
        with (serial.Serial('COM5', baudrate=921600, timeout=2.9)) as self.serialData:

            # Read data from COM port
            command = 'S'

            # Send the command to the DataPort
            self.serialData.write(command.encode())
            # line2 = self.serialData.readline()
            # line1 = self.serialData.read(262144)
            line1 = self.serialData.read(300000)

        # if line1 == line2:
        #     print('ok')
        print(len(line1))
        # print(len(line2))

    def GetData2(self) -> list:
        try:
            with (serial.Serial('COM5', baudrate=921600)) as self.serialData:

                # Read data from COM port
                command = 'S'

                # Send the command to the DataPort
                self.serialData.read(command.encode())
                line = str(self.serialData.readline().strip()) # Строка вида 

            print(len(line))


            return line

        except ValueError as ve:             print("Error:", str(ve))
        except serial.SerialException as se: print("Serial port error:", str(se))
        except Exception as e:               print("An error occurred:", str(e))

if __name__ == '__main__':
    # app = QApplication(sys.argv)
    app = QApplication([])
    
    coils = MainUI()
    coils.show()
    
    # app.exec_()
    sys.exit(app.exec_())