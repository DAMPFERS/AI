#include<SoftwareSerial.h>

#define RX 10
#define TX 11

int csvStr(byte table);
int csvStrLong(byte half1, byte half2);

SoftwareSerial mySerial(RX,TX);

bool byte_flag = false;
byte v = 0b11001100;
byte bytes[8];

void setup() {
  mySerial.begin(9600);
  Serial.begin(9600);
  Serial.println("Start");
}

void loop() {
  //if (Serial.available() > 0){
    //byte_str = Serial.read();    
    //delay(100);  
  //}
  if (mySerial.available() > 0){
    byte packet[9];
    int size_bytes = mySerial.readBytes(packet,9);
    Serial.println(packet[0]);
    for (int i = 1; i < 9; i++)
        bytes[i-1] = packet[i];
    byte_flag = true;   
    delay(100);  
  }
  if (byte_flag){
    for (int i = 0; i < 8; i++)
      csvStr(bytes[i]);
    byte_flag = false;  
  }
}

int csvStr(byte table){
    Serial.print("| ");
    for (int i = 7; i >= 0; i--){
      Serial.print(bitRead(table,i));
      Serial.print(" ");
    }
    Serial.println("|");
    return 0;
    }
int csvStrLong(byte half1, byte half2){
    Serial.print("| ");
    for (int i = 7; i >= 0; i--){
      Serial.print(bitRead(half1,i));
      Serial.print(" ");
    }
    for (int i = 7; i >= 0; i--){
      Serial.print(bitRead(half2,i));
      Serial.print(" ");
    }
    Serial.println("|");
    return 0;
    }
    
    
