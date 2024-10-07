#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <DHT.h>
#include <Wire.h>

Adafruit_BMP280 bmp;

int mq135 = A0;
int data = 0;

int Fire = D0;

DHT dht2(D3, DHT11);

int Firedata = digitalRead(Fire);

void setup() {
  Serial.begin(9600);
  if (!bmp.begin(0x76)) {
    Serial.println("BMP280 Error!");
    while (1);
  }

  pinMode(Fire, INPUT);
}

void loop() {
  float temperature = bmp.readTemperature();
  data = analogRead(mq135);
  float hum = dht2.readHumidity();
  float sea_level = 1013.25;
  float pressure = bmp.readPressure() / 100.0F;
  float altitude2 = 44330.0 * (1.0 - pow((pressure/sea_level), 0.1903));
  
  //Fire,RoomTemp,AirQ,Humidity,Altitude,AirPressure
  Serial.print("Fire: ");
  Serial.print(Firedata);
  Serial.print(" ");
  Serial.print("RoomTemp: ");
  Serial.print(temperature);
  Serial.print(" ");
  Serial.print("AirQuality: ");
  Serial.print(data);
  Serial.print(" ");
  Serial.print("Humidity: ");
  Serial.print(hum);
  Serial.print(" ");
  Serial.print("Altitude: ");
  Serial.print(altitude2);
  Serial.print(" ");
  Serial.print("AirPressure: ");
  Serial.println(pressure);
  delay(1000);
}
