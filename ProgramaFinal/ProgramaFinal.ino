int medidorPin = A0;     //pin A0 del Arduino
int medicion;            //valor obtenido del Pin A0
float valorConvertido;   //valor convertido a tension
const int botonPin = 2;  // Define el pin al que está conectado el botón
const int botonVid = 3;
volatile bool iniciarMedicion = false;  // Variable volatile para indicar si la medicion esta activa

const int frecuenciaMuestreo = 10;      //establece la frecuencia de muestreo

unsigned long tiempoInicial = 0;        // Variable para almacenar el tiempo de la última lectura
float valorResistencia;
int contador = 0;
bool contadorIniciar = false;

int duracionVideo = frecuenciaMuestreo*(21*60);

// Valores de resistencias utilizados
float R1 = 15600;
float R2 = 8200;
float R3 = 3300;
float R4 = 470000.0;
float R5 = 200000.0;
float Rp = 100000.0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);               //baudrate para la transmision de datos en forma serial

  pinMode(LED_BUILTIN, OUTPUT);      
  
  pinMode(botonPin, INPUT_PULLUP);  // Configura el pin del botón como entrada con resistencia pull-up
  //pinMode(botonVid, INPUT_PULLUP);                                            // Establece el modo del pin
  attachInterrupt(digitalPinToInterrupt(botonPin), botonPulsado, FALLING);       // Configura la interrupción en el flanco de bajada
  //attachInterrupt(digitalPinToInterrupt(botonVid), botonPulsadoVideo, FALLING);  // Configura la interrupción en el flanco de bajada
  
}

void loop() {
  // put your main code here, to run repeatedly:

  if (iniciarMedicion) {
    unsigned long tiempoActual = millis();
    if (tiempoActual - tiempoInicial >= 1000 / frecuenciaMuestreo) {
      medicion = analogRead(medidorPin);                        // realizar la lectura
      valorConvertido = mapFloat(medicion, 0, 1023, 0.0, 5.0);  // cambiar escala a 0.0 - 5.0 (voltios)
      convertirResistencia();

      Serial.print(valorResistencia);
      Serial.print("\n");

      tiempoInicial = tiempoActual;+
      contador++;
     //EL VALOR 100 CORRESPONDE A 10S, ARREGLAR CON EL t VIDEO
      if(contador == (300 + duracionVideo + 300))
      {
        iniciarMedicion = false;
        contador = 0;
        digitalWrite(LED_BUILTIN, LOW);
      }
    }
  }
}

void botonPulsado() {
  // Esta función se ejecutará cuando se produzca la interrupción (cuando se presione el botón)

  iniciarMedicion = !iniciarMedicion;  // Establece la bandera indicando que inicia la medicion
  if (iniciarMedicion) {
    digitalWrite(LED_BUILTIN, HIGH);  //Prende el led
    tiempoInicial = millis();         //"Inicia" el timer
  } else {
    digitalWrite(LED_BUILTIN, LOW);
    contador = 0;
  }
}
/*
void botonPulsadoVideo() {
  // Esta función se ejecutará cuando se produzca la interrupción (cuando se presione el botón)
  contadorIniciar = true;
  Serial.print("*****");
  Serial.print("\n");
}*/

void convertirResistencia() {
  //Esta funcion se encarga de transformar el voltaje obtenido a resistencia
  float zp  = R1 * (-1 + ((valorConvertido + 5 * (R3 / (R2 + R3) * (R5 / R4))) / ((1 + R5 / R4) * 0.5)));
  valorResistencia = (Rp*zp)/(Rp-zp);
}

float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
