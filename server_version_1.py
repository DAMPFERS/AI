
import socket, threading
import csv_test

LOCALHOST = '192.168.43.98'
PORT = 1984
start = False
download = False
move = 1


def processingCommand(command): 
    global start
    if command == 0xff:
        print("Download ....")
        start = True
        return 0
    if start:
        if command == 0x01:
            return 1
        elif command == 0x02:
            return 2
        elif command == 0x03:
            print("Download completed")
            global download
            download = True
            start = False
            return 0
        else:
            return 0
            
    


def start_server_one(list_bytes):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((LOCALHOST, PORT))
        print("ОН ВОССТАЛ !!!")
        count = 0
        a = 0
        server.listen(4)
        while(True):
            client_socket, address = server.accept()
            data = client_socket.recv(1024).decode('utf-8')
            print(data)
            count = processingCommand(int(data))
            #client_socket.send("abc".encode('ASCII'))
            if start:
                for i in list_bytes[count]:
                    client_socket.send(i)
            if download:
                #global move
                if a == 0: a = int(input("Command: "))
                str_move = "S" + str(move)
                for i in str_move:
                    client_socket.send(i.encode('ASCII'))
            client_socket.shutdown(socket.SHUT_WR)
    except KeyboardInterrupt:
        server.close()
        print('Finish')






if __name__ == '__main__':
    a = [None]*3
    a[0] = csv_test.csv_read("C:\\PROGRAMS\\NTO\\csv\\Karno.csv", 1)
    a[1] = csv_test.csv_read("C:\\PROGRAMS\\NTO\\csv\\Karno1.csv", 2)
    a[2] = csv_test.csv_read("C:\\PROGRAMS\\NTO\\csv\\Karno2.csv", 3)
    start_server_one(a)
    #bytes([a])# преобразует int в байт, до 255, по скорости  2
    # 'import struct',   struct.pack(">H", 200) преобразует int в байт, (строка формата, длина байта, знак, порядок следования байтов), "B" до 256, ">H" 2 байта в прямом напрвлении и т.д
    #(258).to_bytes(2, byteorder="big", signed=True) преобразует int в bytes, (кол-во байт, "big" или "litle" endian,signed определяет, используется ли дополнение двоих), по скорости 3