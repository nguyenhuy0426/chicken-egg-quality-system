// ============================================
// NM03 - Modbus RTU Master đọc cảm biến nhiệt độ qua RS485
// UART2: TX=GPIO16, RX=GPIO4, DE/RE=GPIO5 (theo tài liệu phần cứng NM03)
// ============================================
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include <string.h>

#define RS485_UART      UART_NUM_2
#define RS485_TX_PIN    16
#define RS485_RX_PIN    4
#define RS485_DE_PIN    5
#define RS485_BAUD      9600     // SỬA theo baud rate của cảm biến (xem datasheet)

#define TEMP_SLAVE_ADDR 0x01     // SỬA theo Slave ID của cảm biến
#define TEMP_REG_ADDR   0x0000   // SỬA theo địa chỉ register (chú ý offset 0-based)
#define TEMP_REG_QTY    1        // số register cần đọc

static const char *TAG = "MODBUS_TEMP";

// ---- Khởi tạo UART2 cho RS485 half-duplex ----
void rs485_init(void)
{
    uart_config_t cfg = {
        .baud_rate = RS485_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_APB,
    };
    uart_param_config(RS485_UART, &cfg);
    uart_set_pin(RS485_UART, RS485_TX_PIN, RS485_RX_PIN,
                 RS485_DE_PIN, UART_PIN_NO_CHANGE);
    uart_driver_install(RS485_UART, 256, 256, 0, NULL, 0);
    // ESP-IDF tự kéo DE (GPIO5) HIGH khi gửi, LOW khi nhận
    uart_set_mode(RS485_UART, UART_MODE_RS485_HALF_DUPLEX);
}

// ---- CRC16 chuẩn Modbus ----
static uint16_t modbus_crc16(const uint8_t *buf, int len)
{
    uint16_t crc = 0xFFFF;
    for (int pos = 0; pos < len; pos++) {
        crc ^= (uint16_t)buf[pos];
        for (int i = 8; i != 0; i--) {
            if (crc & 0x0001) {
                crc >>= 1;
                crc ^= 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

// ---- Đọc Holding Register (function code 0x03) ----
// slave_addr : địa chỉ slave cảm biến
// reg_addr   : địa chỉ register bắt đầu (0-based)
// qty        : số register cần đọc
// out        : buffer nhận giá trị, phải cấp đủ qty phần tử uint16_t
esp_err_t modbus_read_holding_reg(uint8_t slave_addr, uint16_t reg_addr,
                                   uint16_t qty, uint16_t *out)
{
    uint8_t req[8];
    req[0] = slave_addr;
    req[1] = 0x03;                 // Function 03 - Read Holding Registers
    req[2] = (reg_addr >> 8) & 0xFF;
    req[3] = reg_addr & 0xFF;
    req[4] = (qty >> 8) & 0xFF;
    req[5] = qty & 0xFF;
    uint16_t crc = modbus_crc16(req, 6);
    req[6] = crc & 0xFF;
    req[7] = (crc >> 8) & 0xFF;

    uart_flush_input(RS485_UART);
    uart_write_bytes(RS485_UART, (const char *)req, 8);

    // Khung trả lời: [addr][func][byte_count][data...][crc_lo][crc_hi]
    uint8_t resp[64];
    int resp_len = 5 + qty * 2;
    int read_len = uart_read_bytes(RS485_UART, resp, resp_len,
                                    pdMS_TO_TICKS(500)); // timeout 500ms

    if (read_len != resp_len) {
        ESP_LOGW(TAG, "Timeout hoặc thiếu byte: nhận %d/%d", read_len, resp_len);
        return ESP_ERR_TIMEOUT;
    }

    // Báo lỗi exception Modbus (func code có bit 0x80)
    if (resp[1] & 0x80) {
        ESP_LOGW(TAG, "Slave trả exception code: 0x%02X", resp[2]);
        return ESP_ERR_INVALID_RESPONSE;
    }

    uint16_t recv_crc = resp[resp_len - 2] | (resp[resp_len - 1] << 8);
    uint16_t calc_crc = modbus_crc16(resp, resp_len - 2);
    if (recv_crc != calc_crc) {
        ESP_LOGW(TAG, "CRC sai - dữ liệu nhiễu hoặc đấu dây lỗi");
        return ESP_ERR_INVALID_CRC;
    }

    for (int i = 0; i < qty; i++) {
        out[i] = (resp[3 + i * 2] << 8) | resp[4 + i * 2];
    }
    return ESP_OK;
}

// ---- Task đọc nhiệt độ định kỳ ----
void temp_read_task(void *arg)
{
    uint16_t raw[TEMP_REG_QTY];
    while (1) {
        esp_err_t err = modbus_read_holding_reg(TEMP_SLAVE_ADDR, TEMP_REG_ADDR,
                                                  TEMP_REG_QTY, raw);
        if (err == ESP_OK) {
            float temp_c = raw[0] / 10.0f;  // SỬA hệ số chia theo datasheet cảm biến
            ESP_LOGI(TAG, "Nhiệt độ: %.1f °C (raw=%u)", temp_c, raw[0]);
        } else {
            ESP_LOGW(TAG, "Đọc Modbus lỗi: %s", esp_err_to_name(err));
        }
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}

// ---- Gọi trong app_main() ----
// rs485_init();
// xTaskCreate(temp_read_task, "temp_read_task", 4096, NULL, 5, NULL);
