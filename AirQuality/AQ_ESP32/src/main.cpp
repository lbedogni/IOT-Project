#include <Arduino.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "AirQuality.h"


#define NUMBER_OF_INPUTS 5
#define NUMBER_OF_OUTPUTS 1

bool isTest = true;
int nIterTest = 100;


tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 6*1024;
uint8_t tensor_arena[kTensorArenaSize];


void setup() {

    Serial.begin(9600);

    model = tflite::GetModel(AirQualityModel);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("Model provided is schema version not equal to supported!");
        return;
    } else {
        Serial.print("Model loaded!\n");
    }

    // Questo richiama tutte le implementazioni delle operazioni di cui abbiamo bisogno
    static tflite::AllOpsResolver resolver;

    // Creo un interprete con cui eseguire il modello
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Alloco la memoria del tensor_arena per i tensori del modello
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.print("AllocateTensors() failed");
        return;
    } else {
        Serial.print("AllocateTensors() done\n");
    }

    // Ottengo puntatori ai tensori di input e output del modello
    input = interpreter->input(0);
    output = interpreter->output(0);

}

void loop() {

    if( isTest ){

        if( nIterTest > 0 ){

            float x_rand_test[ NUMBER_OF_INPUTS ];

            // Inizializzo array con valore random che stano tra 0 e 1
            for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
                x_rand_test[i] = static_cast<float>(rand()) / RAND_MAX;
            }

            // Array di output per la previsione del modello
            float y_rand_pred[ NUMBER_OF_OUTPUTS ] = {0.0};

            // Calcolo il tempo che impiega per la previsione del modello
            uint32_t start = micros();

            // Posiziono l'input nel tensore di input del modello
            for(int x=0; x<NUMBER_OF_INPUTS; x++)
                input->data.f[x] = x_rand_test[x];

            // Eseguo l'inferenza e segnalo eventuali errori
            TfLiteStatus invoke_status = interpreter->Invoke();
            if (invoke_status != kTfLiteOk) {
                Serial.print("Invoke failed!");
                return;
            }

            // Ottengo l'output dal tensore di output del modello
            if (output != NULL) {
                for (int i = 0; i < NUMBER_OF_OUTPUTS; i++)
                    y_rand_pred[i] = output->data.f[i];
            }

            uint32_t timePred = micros() - start;

            Serial.print(timePred);

            nIterTest -= 1;

            Serial.print(nIterTest == 0 ? "\n" : ", ");

            // Reset dell'interpreter
            interpreter->Reset();

        }

    } else {

        // Dal file Header (AirQuality.h):
        // un sample selezionato dal dataset -> x_test
        // valore reale del sample selezionato dal dataset -> y_test

        // Array di output per la previsione del modello
        float y_pred[ NUMBER_OF_OUTPUTS ] = {0.0};

        // Calcolo il tempo che impiega per la previsione del modello
        uint32_t start = micros();

        // Posiziono l'input nel tensore di input del modello
        for(int x=0; x<NUMBER_OF_INPUTS; x++)
            input->data.f[x] = x_test[x];

        // Eseguo l'inferenza e segnalo eventuali errori
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            Serial.print("Invoke failed!");
            return;
        }

        // Ottengo l'output dal tensore di output del modello
        if (output != NULL) {
            for (int i = 0; i < NUMBER_OF_OUTPUTS; i++)
                y_pred[i] = output->data.f[i];
        }

        uint32_t timePred = micros() - start;

        Serial.print("\n\nIt took ");
        Serial.print(timePred);
        Serial.println(" micros to run inference");

        // Stampo il valore reale del sample preso in considerazione
        Serial.print("Test output is: ");
        Serial.println(y_test, 5);

        // Stampo il risultato del modello
        Serial.print("Predicted value is: ");
        Serial.print(y_pred[0], 5);

        // Reset dell'interpreter
        interpreter->Reset();  

        delay(60000);

    }

}
