#include <WiFi.h>

#define START 0xff

const char* host = "192.168.43.98";
//const char* host = "192.168.1.107";
uint16_t port = 1984;
const char* ssid     = "Honor 8C";
const char* password = "89243536319";
//const char* ssid     = "SumОм";
//const char* password = "sumomsumom";
//byte feedback = 0;
byte feed_back;

void setup() {
  feed_back = START;
  Serial.begin(9600);
  WiFi.begin(ssid, password);
  delay(200);
  //Serial.println("Connecting... ");
  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print(".");
    delay(500);
  }
  delay(1000);
}

void loop() {
  WiFiClient client;
  if (!client.connect(host,port))
    delay(3000);
  if(client.connected())
  {
    client.print(feed_back);  
    delay(500);
    if (client.available() > 0){
      //byte packageS[9];
      byte count = client.available();
      byte packageS[count];
      int packageS_size = client.readBytes(packageS,count);
      feed_back = packageS[0];
      for (int i = 0; i < count; i++)
        Serial.write(packageS[i]);
        //Serial.println(packageS[i],BIN);
      //Serial.println("---------------------------");
    }
    //char w[15];
    //int count = client.available();
    //for(int i = 0; i < count; i++)
     // w[i] = client.read();
    //for(int i = 0; i < count; i++)
      //Serial.print(w[i]);
    //client.stop();
  }
}
