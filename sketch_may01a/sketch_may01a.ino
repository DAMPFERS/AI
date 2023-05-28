#include <WiFi.h>

const char* host = "192.168.43.98";
//const char* host = "192.168.1.107";
uint16_t port = 1984;
const char* ssid     = "Honor 8C";
const char* password = "89243536319";
//const char* ssid     = "SumОм";
//const char* password = "sumomsumom";



void setup() {
  Serial.begin(9600);
  WiFi.begin(ssid, password);
  delay(200);
  //Serial.println("Connecting... ");
  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print(".");
    delay(500);
  }
  //Serial.print("Connected to ");
  //Serial.println(ssid);
  //Serial.println("\nStarting connection to server...");
  delay(1000);
}

void loop() {
  WiFiClient client;
  if (!client.connect(host,port))
    delay(3000);
  if (client.connected())
  {
    client.print("12");  
    delay(500);
    if (client.available() > 0){
      byte packageS[8];
      int packageS_size = client.readBytes(packageS,8);
      for (int i = 0; i < 8; i++)
        Serial.write(packageS[i]);
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
