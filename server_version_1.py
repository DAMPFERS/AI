
import socket, threading
import csv_test

LOCALHOST = '192.168.43.98'
PORT = 1984


def start_server_one(list_bytes):
    
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((LOCALHOST, PORT))
        print("ОН ВОССТАЛ !!!")
        server.listen(4)
        while(True):
            client_socket, address = server.accept()
            data = client_socket.recv(1024).decode('utf-8')
            print(data)
            for i in list_bytes:
                client_socket.send(i)
            client_socket.shutdown(socket.SHUT_WR)
    except KeyboardInterrupt:
        server.close()
        print('Finish')
#def load_page_from_get_request(request_data):
 #   HDRS = 'HTTP/1.1 200 OK\r\nContent_Type: text/html; charset=utf-8\r\n\r\n'
 #   return HDRS.encode('utf-8') + request_data.encode('utf-8')





if __name__ == '__main__':
    a = csv_test.csv_read("C:\\PROGRAMS\\NTO\\csv\\Karno.csv")
    start_server_one(a)
    #bytes([a])# преобразует int в байт, до 255, по скорости  2
    # 'import struct',   struct.pack(">H", 200) преобразует int в байт, (строка формата, длина байта, знак, порядок следования байтов), "B" до 256, ">H" 2 байта в прямом напрвлении и т.д
    #(258).to_bytes(2, byteorder="big", signed=True) преобразует int в bytes, (кол-во байт, "big" или "litle" endian,signed определяет, используется ли дополнение двоих), по скорости 3