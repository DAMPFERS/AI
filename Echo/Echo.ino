long int timer = 0;
void setup() {
  Serial.begin(9600);
}

void loop() {
  if (millis() - timer > 2000){
    //Serial.write("a");
    Serial.print(4);
    timer = millis();
  }

}
