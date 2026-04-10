/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "detection_responder.h"
#include "driver/gpio.h"
#include "esp_main.h"

// Broches GPIO pour les LEDs (vous pouvez les modifier selon votre câblage)
#define LED_RED_PIN GPIO_NUM_22
#define LED_BLUE_PIN GPIO_NUM_23

#if DISPLAY_SUPPORT
#include "esp_lcd.h"
#include "image_provider.h"
static QueueHandle_t xQueueLCDFrame = NULL;
#endif

void RespondToDetection(tflite::ErrorReporter *error_reporter,
                        float person_score, float no_person_score) {

  // Initialiser les broches LED la première fois que la fonction est appelée
  static bool leds_initialized = false;
  if (!leds_initialized) {
    gpio_reset_pin(LED_RED_PIN);
    gpio_set_direction(LED_RED_PIN, GPIO_MODE_OUTPUT);
    gpio_reset_pin(LED_BLUE_PIN);
    gpio_set_direction(LED_BLUE_PIN, GPIO_MODE_OUTPUT);
    leds_initialized = true;
  }

  int person_score_int = (person_score) * 100 + 0.5;

  // Contrôler les LEDs selon la détection (seuil de 60%)
  if (person_score_int >= 80) {
    gpio_set_level(LED_BLUE_PIN, 1); // Allumer Bleu = Personne
    gpio_set_level(LED_RED_PIN, 0);  // Eteindre Rouge
  } else {
    gpio_set_level(LED_BLUE_PIN, 0); // Eteindre Bleu
    gpio_set_level(LED_RED_PIN, 1);  // Allumer Rouge = Pas de personne
  }

#if DISPLAY_SUPPORT
  if (xQueueLCDFrame == NULL) {
    xQueueLCDFrame = xQueueCreate(2, sizeof(struct lcd_frame));
    register_lcd(xQueueLCDFrame, NULL, false);
  }

  int color = 0x1f << 6;       // red
  if (person_score_int < 80) { // treat score less than 60% as no person
    color = 0x3f;              // green
  }
  app_lcd_color_for_detection(color);

  // display frame (freed by lcd task)
  lcd_frame_t *frame = (lcd_frame_t *)malloc(sizeof(lcd_frame_t));
  frame->width = 96 * 2;
  frame->height = 96 * 2;
  frame->buf = image_provider_get_display_buf();
  xQueueSend(xQueueLCDFrame, &frame, portMAX_DELAY);
  (void)no_person_score;
#else
  TF_LITE_REPORT_ERROR(error_reporter,
                       "person score:%d%%, no person score %d%%",
                       person_score_int, 100 - person_score_int);
#endif
}
