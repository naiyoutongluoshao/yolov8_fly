#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

const char* ssid = "nihao";
const char* password = "123456789";

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

esp_err_t index_handler(httpd_req_t *req);
esp_err_t stream_handler(httpd_req_t *req);
void startCameraServer();

void setup() {
  Serial.begin(115200);
  Serial.println("Start...");  // <-- 一定会打印！

  // 先连 WIFI，再初始化摄像头（解决卡死）
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi...");
  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 20) {
    delay(500);
    Serial.print(".");
    timeout++;
  }

  Serial.println("");
  Serial.println("WiFi OK!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());  // <-- 现在一定能看到！

  // 初始化摄像头
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.println("Camera init failed, but WiFi still work!");
  } else {
    Serial.println("Camera OK!");
    startCameraServer();
  }
}

void loop() {
}

esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  httpd_resp_send(req, "<img src=/stream>", HTTPD_RESP_USE_STRLEN);
  return ESP_OK;
}

esp_err_t stream_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  while (1) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) break;
    char buf[64];
    snprintf(buf, 64, "--frame\r\nContent-Type: image/jpeg\r\n\r\n");
    httpd_resp_send_chunk(req, buf, strlen(buf));
    httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    httpd_resp_send_chunk(req, "\r\n", 2);
    esp_camera_fb_return(fb);
  }
  return ESP_OK;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  httpd_handle_t server = NULL;
  httpd_uri_t index_uri = {"/", HTTP_GET, index_handler, NULL};
  httpd_uri_t stream_uri = {"/stream", HTTP_GET, stream_handler, NULL};
  httpd_start(&server, &config);
  httpd_register_uri_handler(server, &index_uri);
  httpd_register_uri_handler(server, &stream_uri);
}