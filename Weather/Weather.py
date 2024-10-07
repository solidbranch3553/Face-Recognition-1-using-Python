#Libraries
import cv2
import matplotlib.pyplot as plt

#Give an Input
data = "Fire: False RoomTemp: 30C AirQuality: 150 Humidity: 40 Altitude: 100 AirPressure: 950pah"

#Main
if data:
    spl = data.split()
    fire = str(spl[1])
    roomTemp = str(spl[3])
    airQ = str(spl[5])
    hum = str(spl[7])
    alt = str(spl[9])
    airP = str(spl[11])
    print(f"Fire Stat: {fire}\nRoom Temperature: {roomTemp}\nAir Quality: {airQ}\nHumidity: {hum}\nAltitude: {alt}\nAir Pressure: {airP}")
    
    WeatherImg = cv2.imread('./Final.png')
    WeatherImg = cv2.cvtColor(WeatherImg, cv2.COLOR_BGR2RGB)
    
    cv2.putText(WeatherImg, fire, (227,520), cv2.FONT_HERSHEY_PLAIN, 4.4, (255,255,255), 7)
    cv2.putText(WeatherImg, roomTemp, (881,520), cv2.FONT_HERSHEY_PLAIN, 4.4, (255,0,0), 7)
    cv2.putText(WeatherImg, airQ, (1614,517), cv2.FONT_HERSHEY_PLAIN, 4.4, (0,0,0), 7)
    cv2.putText(WeatherImg, hum, (273,961), cv2.FONT_HERSHEY_PLAIN, 4.4, (255,0,0), 7)
    cv2.putText(WeatherImg, alt, (947,956), cv2.FONT_HERSHEY_PLAIN, 4.4, (255,255,255), 7)
    cv2.putText(WeatherImg, airP, (1633,949), cv2.FONT_HERSHEY_PLAIN, 4, (255,30,0), 7)
    
    plt.imshow(WeatherImg)
    plt.show()