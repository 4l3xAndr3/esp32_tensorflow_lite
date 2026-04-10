// Copyright 2020-2021 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <driver/uart.h>
#include <esp_console.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>
#include <stdio.h>
#include <string.h>

#include "esp_cli.h"
#include "esp_main.h"
#include "esp_timer.h"

#define SERIAL_IMAGE_SIZE (96 * 96) // 9216 bytes for 96x96 grayscale

static uint8_t
    serial_image_buf[SERIAL_IMAGE_SIZE]; // buffer for serial-received images

static int stop;
static const char *TAG = "[esp_cli]";

static int mem_dump_cli_handler(int argc, char *argv[]) {
  /* Just to go to the next line */
  printf("\n");
  printf("\tDescription\tInternal\tSPIRAM\n");
  printf("Current Free Memory\t%d\t\t%d\n",
         heap_caps_get_free_size(MALLOC_CAP_8BIT) -
             heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
         heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  printf(
      "Largest Free Block\t%d\t\t%d\n",
      heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL),
      heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
  printf("Min. Ever Free Size\t%d\t\t%d\n",
         heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL),
         heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
  return 0;
}

/**
 * detect_serial: Receive a 96x96 grayscale image over UART from the PC,
 * run TFLite inference, and send back the result.
 *
 * Protocol:
 *   ESP32 -> PC : "READY\n"
 *   PC -> ESP32 : 9216 raw bytes (96x96 grayscale image)
 *   ESP32 -> PC : "FRAME_OK:<time_ms>\n"
 *   (loops until no more data or timeout)
 */
static int detect_serial_cli_handler(int argc, char *argv[]) {
  printf("\n%s: Starting serial detection mode. Send 96x96 grayscale frames.\n",
         TAG);

  int uart_num = UART_NUM_0; // default UART used by CLI
  int total_bytes = SERIAL_IMAGE_SIZE;
  int frames_processed = 0;

  while (1) {
    // Signal to the PC that we are ready for the next frame
    const char *ready_msg = "READY\n";
    uart_write_bytes(uart_num, ready_msg, strlen(ready_msg));

    // Receive the image data
    int bytes_received = 0;
    int timeout_count = 0;
    const int MAX_TIMEOUTS = 100; // 100 * 100ms = 10s max wait

    while (bytes_received < total_bytes) {
      int len =
          uart_read_bytes(uart_num, &serial_image_buf[bytes_received],
                          total_bytes - bytes_received, pdMS_TO_TICKS(100));
      if (len > 0) {
        bytes_received += len;
        timeout_count = 0; // reset timeout on data received
      } else {
        timeout_count++;
        if (timeout_count >= MAX_TIMEOUTS) {
          printf("%s: Timeout waiting for image data (got %d/%d bytes)\n", TAG,
                 bytes_received, total_bytes);
          goto exit_serial;
        }
      }
    }

    // Run inference on the received image
    uint32_t detect_time = esp_timer_get_time();
    run_inference((void *)serial_image_buf);
    detect_time = (esp_timer_get_time() - detect_time) / 1000;

    frames_processed++;
    ESP_LOGI(TAG, "Frame #%d processed in %d ms", frames_processed,
             detect_time);

    // Send back a parseable result line
    char result_msg[64];
    snprintf(result_msg, sizeof(result_msg), "FRAME_OK:%d\n", detect_time);
    uart_write_bytes(uart_num, result_msg, strlen(result_msg));

    vTaskDelay(pdMS_TO_TICKS(10)); // small delay to avoid watchdog
  }

exit_serial:
  printf("%s: Serial detection stopped. %d frames processed.\n", TAG,
         frames_processed);
  return 0;
}

static esp_console_cmd_t diag_cmds[] = {
    {
        .command = "mem-dump",
        .help = "",
        .func = mem_dump_cli_handler,
    },
    {
        .command = "detect_serial",
        .help = "Receive 96x96 grayscale images from PC webcam via serial and "
                "run inference",
        .func = detect_serial_cli_handler,
    },
};

static void esp_cli_task(void *arg) {
  int uart_num = (int)arg;
  uint8_t linebuf[256];
  int i, cmd_ret;
  esp_err_t ret;
  QueueHandle_t uart_queue;
  uart_event_t event;

  ESP_LOGI(TAG, "Initialising UART on port %d", uart_num);
  uart_driver_install(uart_num, 10240, 0, 8, &uart_queue, 0);
  /* Initialize the console */
  esp_console_config_t console_config = {
      .max_cmdline_args = 8,
      .max_cmdline_length = 256,
  };

  esp_console_init(&console_config);
  esp_console_register_help_command();

  while (!stop) {
    uart_write_bytes(uart_num, "\n>> ", 4);
    memset(linebuf, 0, sizeof(linebuf));
    i = 0;
    do {
      ret = xQueueReceive(uart_queue, (void *)&event,
                          (portTickType)portMAX_DELAY);
      if (ret != pdPASS) {
        if (stop == 1) {
          break;
        } else {
          continue;
        }
      }
      if (event.type == UART_DATA) {
        while (uart_read_bytes(uart_num, (uint8_t *)&linebuf[i], 1, 0)) {
          if (linebuf[i] == '\r') {
            uart_write_bytes(uart_num, "\r\n", 2);
          } else {
            uart_write_bytes(uart_num, (char *)&linebuf[i], 1);
          }
          i++;
        }
      }
    } while ((i < 255) && linebuf[i - 1] != '\r');
    if (stop) {
      break;
    }
    /* Remove the truncating \r\n */
    linebuf[strlen((char *)linebuf) - 1] = '\0';
    ret = esp_console_run((char *)linebuf, &cmd_ret);
    if (ret < 0) {
      printf("%s: Console dispatcher error\n", TAG);
      break;
    }
  }
  ESP_LOGE(TAG, "Stopped CLI");
  vTaskDelete(NULL);
}

int esp_cli_register_cmds() {
  int cmds_num = sizeof(diag_cmds) / sizeof(esp_console_cmd_t);
  int i;
  for (i = 0; i < cmds_num; i++) {
    ESP_LOGI(TAG, "Registering command: %s", diag_cmds[i].command);
    esp_console_cmd_register(&diag_cmds[i]);
  }
  return 0;
}

int esp_cli_init() {
  static int cli_started;
  if (cli_started) {
    return 0;
  }
#define ESP_CLI_STACK_SIZE (4 * 1024)
  if (pdPASS != xTaskCreate(&esp_cli_task, "cli_task", ESP_CLI_STACK_SIZE, NULL,
                            4, NULL)) {
    ESP_LOGE(TAG, "Couldn't create task");
    return -1;
  }
  cli_started = 1;
  return 0;
}
