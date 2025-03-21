/*
 * 修改后的固件示例：
 */

 #include <cstring>
 #include <vector>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 #include "app.h"
 #include "board.h"
 #include "fsl_device_registers.h"
 #include "fsl_debug_console.h"
 
 #include "ethosu_driver.h"
 #include "ethosu_core_interface.h"
 #include "inference_process.hpp"
 #include "conv2d_model.hpp"
 //#include "depthwise_conv2d_model.hpp"
 //#include "fullconnected_model.hpp"
 //#include "relu_model.hpp"
 //#include "softmax_model.hpp"
 
 #define FAST_MEMORY_ADDRESS      0x20480000
 #define FAST_MEMORY_SIZE         0x60000   // 384KB
 
 // 预留 16KB 存放模型数据，其余 368KB 用作 tensor arena
 #define MODEL_REGION_SIZE        (16 * 1024)
 #define TENSOR_ARENA_SIZE        (FAST_MEMORY_SIZE - MODEL_REGION_SIZE)
 
 #define ETHOSU_BASE_ADDRESS      0x4A900000
 #define ETHOSU_IRQ               178
 
 #define DDR_MEMORY_ADDRESS       0x80000000
 #define OCRAM_MEMORY_ADDRESS     0x20480000
 
 #if (!defined(__ICCARM__))
 using namespace std;
 using namespace InferenceProcess;
 #endif
 
 // 不再静态分配 tensor arena，改为运行时动态获取
 uint8_t *inferenceProcessTensorArena;
 
 using namespace EthosU;
 
 struct ethosu_driver ethosu_drv;
 
 int main(void)
 {
     BOARD_InitHardware();
 
     PRINTF("Initialize Arm Ethos-U\r\n");
     if (ethosu_init(&ethosu_drv, (void *)ETHOSU_BASE_ADDRESS,
                     (void *)FAST_MEMORY_ADDRESS, FAST_MEMORY_SIZE, 0, 0)) {
         PRINTF("Failed to initialize Arm Ethos-U\n");
         return 1;
     }
 
     NVIC_SetVector((IRQn_Type)ETHOSU_IRQ, (uint32_t)&ethosu_irq_handler);
     NVIC_EnableIRQ((IRQn_Type)ETHOSU_IRQ);
 
     // 复制模型数据到 OCRAM 的前 MODEL_REGION_SIZE 区域
     memcpy((void *)FAST_MEMORY_ADDRESS, (void *)model_data, sizeof(model_data));
     // 构造模型数据的 DataPtr，使用预留区域的起始地址和大小（可以使用 sizeof(model_data)）
     InferenceProcess::DataPtr networkModel((void *)FAST_MEMORY_ADDRESS, sizeof(model_data));
 
     std::vector<InferenceProcess::DataPtr> ifm, ofm, expectedOutput;
     // 注意这里 input_data 应该放到 DDR 或其他内存中（这里保持不变）
     ifm.push_back(InferenceProcess::DataPtr((void *)input_data, sizeof(input_data)));
 
//     PRINTF("I Run Here0!\r\n");
     std::vector<uint8_t> pmuEventConfig(ETHOSU_CORE_PMU_MAX);
 
//     PRINTF("I Run Here1!\r\n");
     InferenceProcess::InferenceJob job("job", networkModel, ifm, ofm, expectedOutput,
                                        pmuEventConfig, 0, &ethosu_drv, 0, nullptr, false);
     job.invalidate();
 
//     PRINTF("I Run Here2\r\n");
     // 使用 OCRAM 中 MODEL_REGION_SIZE 后面的区域作为 tensor arena
     inferenceProcessTensorArena = (uint8_t *)(FAST_MEMORY_ADDRESS + MODEL_REGION_SIZE);
     // 清零 tensor arena 区域
     memset(inferenceProcessTensorArena, 0, TENSOR_ARENA_SIZE);
 
     // 检查 tensor arena 是否清零成功
     bool cleared = true;
     for (uint32_t i = 0; i < TENSOR_ARENA_SIZE; i++) {
         if (inferenceProcessTensorArena[i] != 0) {
             cleared = false;
             break;
         }
     }
     if (cleared) {
         PRINTF("Tensor arena region cleared successfully.\r\n");
     } else {
         PRINTF("Failed to clear tensor arena region.\r\n");
         return 1;
     }
 
     InferenceProcess::InferenceProcess inferenceprocess(inferenceProcessTensorArena, TENSOR_ARENA_SIZE);
 
//     PRINTF("I Run Here2.5\r\n");
     bool failed = inferenceprocess.runJob(job);
//     PRINTF("I Run Here2.9\r\n");
     job.clean();
//     PRINTF("I Run Here2.91\r\n");
 
//     PRINTF("I Run Here3\r\n");
     if (failed == true)
         PRINTF("Inference status: failed\r\n");
     else
         PRINTF("Inference status: success\r\n");
 
     return 0;
 }
 