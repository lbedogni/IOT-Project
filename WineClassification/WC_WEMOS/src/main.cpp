#include <Arduino.h>
#include <EloquentTinyML.h>
#include "WineClassification.h"

#define NUMBER_OF_INPUTS 13
#define NUMBER_OF_OUTPUTS 3
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> model;


bool isTest = true;
int nIterTest = 100;


void setup() {

    Serial.begin(9600);

    // Dal file Header (WineClassification.h):
    // modello tensorflow -> WineClassificationModel

    if ( ! model.begin(WineClassificationModel) ) {
        Serial.println("Cannot load model!\n");
        delay(60000);
    } else {
        Serial.print("Model loaded!\n");
    }

}

void loop() {

    if( isTest ){

        if( nIterTest > 0 ){

            float x_rand_test[ NUMBER_OF_INPUTS ];

            // Array di output per la previsione del modello
            float y_rand_pred[ NUMBER_OF_OUTPUTS ] = {0};

            // Inizializzo array con valore random che stano tra 0 e 1
            for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
                x_rand_test[i] = static_cast<float>(rand()) / RAND_MAX;
            }

            // Calcolo quanto inpiega la previsione
            uint32_t start = micros();

            model.predict(x_rand_test, y_rand_pred);

            uint32_t timePred = micros() - start;

            Serial.print(timePred);

            nIterTest -= 1;

            Serial.print(nIterTest == 0 ? "\n" : ", ");

        }

    } else {

        // Dal file Header (WineClassification.h):
        // un sample selezionato dal dataset -> x_test
        // valore reale del sample selezionato dal dataset -> y_test

        // Array di output per la previsione del modello
        float y_pred[ NUMBER_OF_OUTPUTS ] = {0};

        // Calcolo quanto impiega la previsione
        uint32_t start = micros();

        model.predict(x_test, y_pred);

        uint32_t timePred = micros() - start;

        Serial.print("\nIt took ");
        Serial.print(timePred);
        Serial.println(" micros to run inference !");

        // Stampo il valore reale del sample preso in considerazione
        Serial.print("Test output is: ");
        Serial.println(y_test);

        // Stampo il risultato del modello
        // Sono le probabilità in riferimento ad ogni classe
        Serial.print("Predicted proba are: ");
        for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
            Serial.print(y_pred[i]);
            Serial.print(i == NUMBER_OF_OUTPUTS-1 ? "\n" : ", ");
        }

        delay(60000);

    }

}
