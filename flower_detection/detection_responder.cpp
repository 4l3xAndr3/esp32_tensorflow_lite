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
#include <math.h>
#include "driver/gpio.h"
#include "esp_main.h"
#include "model_settings.h"

// Broches GPIO pour les LEDs (vous pouvez les modifier selon votre câblage)
#define LED_RED_PIN GPIO_NUM_22
#define LED_BLUE_PIN GPIO_NUM_23

#if DISPLAY_SUPPORT
#include "esp_lcd.h"
#include "image_provider.h"
static QueueHandle_t xQueueLCDFrame = NULL;
#endif

void RespondToDetection(tflite::ErrorReporter *error_reporter,
                        float* scores) {

  // Initialiser les broches LED la première fois que la fonction est appelée
  static bool leds_initialized = false;
  if (!leds_initialized) {
    gpio_reset_pin(LED_RED_PIN);
    gpio_set_direction(LED_RED_PIN, GPIO_MODE_OUTPUT);
    gpio_reset_pin(LED_BLUE_PIN);
    gpio_set_direction(LED_BLUE_PIN, GPIO_MODE_OUTPUT);
    leds_initialized = true;
  }

  // Le modèle tensorflow lite génère des logits.
  // On applique un Softmax pour obtenir de vraies probabilités entre 0 et 1.
  float max_logit = scores[0];
  for (int i = 1; i < kCategoryCount; i++) {
      if (scores[i] > max_logit) max_logit = scores[i];
  }
  float sum_exp = 0.0;
  for (int i = 0; i < kCategoryCount; i++) {
      sum_exp += exp(scores[i] - max_logit);
  }
  
  int best_index = 0;
  float best_prob = 0.0;
  for (int i = 0; i < kCategoryCount; i++) {
      float prob = exp(scores[i] - max_logit) / sum_exp;
      if (prob > best_prob) {
          best_prob = prob;
          best_index = i;
      }
  }

  int best_score_int = (best_prob) * 100 + 0.5;

  // S'il n'y a pas de certitude (>50%), on allume rouge
  if (best_score_int >= 50) {
    gpio_set_level(LED_BLUE_PIN, 1); 
    gpio_set_level(LED_RED_PIN, 0);  
  } else {
    gpio_set_level(LED_BLUE_PIN, 0); 
    gpio_set_level(LED_RED_PIN, 1);  
  }

#if DISPLAY_SUPPORT
  if (xQueueLCDFrame == NULL) {
    xQueueLCDFrame = xQueueCreate(2, sizeof(struct lcd_frame));
    register_lcd(xQueueLCDFrame, NULL, false);
  }

  int color = 0x1f << 6;       // red
  if (best_score_int >= 50) { 
    color = 0x3f;              // green
  }
  app_lcd_color_for_detection(color);

  // display frame (freed by lcd task)
  lcd_frame_t *frame = (lcd_frame_t *)malloc(sizeof(lcd_frame_t));
  frame->width = 96 * 2;
  frame->height = 96 * 2;
  frame->buf = image_provider_get_display_buf();
  xQueueSend(xQueueLCDFrame, &frame, portMAX_DELAY);
#else
  TF_LITE_REPORT_ERROR(error_reporter,
                       "top_flower:%s score:%d%%",
                       kCategoryLabels[best_index], best_score_int);
#endif
}
