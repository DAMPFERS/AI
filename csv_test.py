import csv
import struct

def scv_write():#Запись в сsv файл для windows
    name_1 = "Anna"
    name_2 = "Antony"
    data_names = ["Anna","Anna"]
    users_data = [
        ("user name","user address"),
        ["user1","address1"],
        ["user2","address2"],
        ["user3","address3"],
        ]

    with open("C:\PROGRAMS\Python\Server\Тест\CSV\data.csv","w",encoding="cp1251", newline = "") as file:  
        writer = csv.writer(file, delimiter = ";")
        writer.writerows(users_data)
        # newline = "" если строки записываются через 1
        # encoding="cp1251" для поддержания кириллици
        #delimiter = ";"  разделитель м/у столбцами, априоре "," но в экселе windows по другому


    '''for user in users_data:
        with open("C:\PROGRAMS\Python\Server\Тест\CSV\data.csv","a") as file:
            writer = csv.writer(file)
            writer.writerow(
                user
            )
            '''

def csv_read(way):
    table = [None] * 8
    with open(way,"r") as file:
        reader = csv.reader(file,delimiter = ";")
        count = 0
        for line in reader:
            byte = 0
            for i in line:
                if i == "1":
                    byte += 1
                elif i == "0":
                    pass
                else:
                    print("ERROR csv Karno")
                    return -1
                
                byte = byte << 1
            byte = byte >> 1
            table[count] = struct.pack("B", byte)
            count += 1
        return table

if __name__ == "__main__":
    bytes_csv = csv_read("C:\\PROGRAMS\\NTO\\csv\\Karno.csv")
    print(bytes_csv)

    