/*
 * Copyright (c) 2015, Freescale Semiconductor, Inc.
 * Copyright 2016-2023 NXP
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "fsl_debug_console.h"
#include "fsl_silicon_id.h"
#include "fsl_enet.h"
#include "fsl_phy.h"
#include "pin_mux.h"
#include "board.h"

#include "fsl_phyrtl8211f.h"
/*******************************************************************************
 * Definitions
 ******************************************************************************/
extern phy_rtl8211f_resource_t g_phy_resource;
#define EXAMPLE_ENET                ENET
#define EXAMPLE_PHY_ADDRESS         (0x02U)
#define EXAMPLE_PHY_INTERFACE_RGMII 1
#define EXAMPLE_PHY_OPS             &phyrtl8211f_ops
#define EXAMPLE_PHY_RESOURCE        &g_phy_resource

/* ENET clock frequency. */
#define EXAMPLE_CLOCK_ROOT kCLOCK_Root_WakeupAxi
#define EXAMPLE_CLOCK_FREQ CLOCK_GetIpFreq(EXAMPLE_CLOCK_ROOT)
#define ENETREF_CLOCK_ROOT kCLOCK_Root_EnetRef
#define ENET_CLOCK_GATE    kCLOCK_Enet1
#define PTP_CLOCK_ROOT     kCLOCK_Root_EnetTimer1
#define PTP_CLOCK_GATE     kCLOCK_Tstmr1
#define PTP_CLK_FREQ       CLOCK_GetIpFreq(kCLOCK_Root_EnetTimer1)
#define ENET_RXBD_NUM          (4)
#define ENET_TXBD_NUM          (4)
#define ENET_RXBUFF_SIZE       (ENET_FRAME_MAX_FRAMELEN)
#define ENET_TXBUFF_SIZE       (ENET_FRAME_MAX_FRAMELEN)
#define ENET_DATA_LENGTH       (1000)
#define ENET_TRANSMIT_DATA_NUM (20)
#define ENET_PTP_SYNC_MSG      0x00U
#ifndef APP_ENET_BUFF_ALIGNMENT
#define APP_ENET_BUFF_ALIGNMENT ENET_BUFF_ALIGNMENT
#endif
#ifndef PHY_AUTONEGO_TIMEOUT_COUNT
#define PHY_AUTONEGO_TIMEOUT_COUNT (300000)
#endif
#ifndef PHY_STABILITY_DELAY_US
#define PHY_STABILITY_DELAY_US (0U)
#endif

/* @TEST_ANCHOR */

#ifndef MAC_ADDRESS
#define MAC_ADDRESS                        \
    {                                      \
        0x54, 0x27, 0x8d, 0x00, 0x00, 0x00 \
    }
#else
#define USER_DEFINED_MAC_ADDRESS
#endif

/*******************************************************************************
 * Prototypes
 ******************************************************************************/

/*! @brief Build ENET PTP event frame. */
static void ENET_BuildPtpEventFrame(void);

/*******************************************************************************
 * Variables
 ******************************************************************************/
phy_rtl8211f_resource_t g_phy_resource;
/*! @brief Buffer descriptors should be in non-cacheable region and should be align to "ENET_BUFF_ALIGNMENT". */
AT_NONCACHEABLE_SECTION_ALIGN(static enet_rx_bd_struct_t g_rxBuffDescrip[ENET_RXBD_NUM], ENET_BUFF_ALIGNMENT);
AT_NONCACHEABLE_SECTION_ALIGN(static enet_tx_bd_struct_t g_txBuffDescrip[ENET_TXBD_NUM], ENET_BUFF_ALIGNMENT);
/*! @brief The data buffers can be in cacheable region or in non-cacheable region.
 * If use cacheable region, the alignment size should be the maximum size of "CACHE LINE SIZE" and "ENET_BUFF_ALIGNMENT"
 * If use non-cache region, the alignment size is the "ENET_BUFF_ALIGNMENT".
 */
SDK_ALIGN(static uint8_t g_rxDataBuff[ENET_RXBD_NUM][SDK_SIZEALIGN(ENET_RXBUFF_SIZE, APP_ENET_BUFF_ALIGNMENT)],
          APP_ENET_BUFF_ALIGNMENT);
SDK_ALIGN(static uint8_t g_txDataBuff[ENET_TXBD_NUM][SDK_SIZEALIGN(ENET_TXBUFF_SIZE, APP_ENET_BUFF_ALIGNMENT)],
          APP_ENET_BUFF_ALIGNMENT);

static enet_handle_t g_handle;
static uint8_t g_frame[ENET_DATA_LENGTH + 14];
static uint32_t g_testTxNum = 0;
static enet_frame_info_t txFrameInfoArray[ENET_TXBD_NUM];

/* The unicast MAC address for ENET device. */
static uint8_t g_macAddr[6] = MAC_ADDRESS;

/* The multicast MAC address. */
static uint8_t multicastAddr[6] = {0x01, 0x00, 0x5e, 0x01, 0x01, 0x1};

static volatile bool tx_frame_over   = false;
static enet_frame_info_t txFrameInfo = {0};

/*! @brief Enet PHY and MDIO interface handler. */
static phy_handle_t phyHandle;

/*******************************************************************************
 * Code
 ******************************************************************************/
static void MDIO_Init(void)
{
    (void)CLOCK_EnableClock(s_enetClock[ENET_GetInstance(EXAMPLE_ENET)]);
    ENET_SetSMI(EXAMPLE_ENET, EXAMPLE_CLOCK_FREQ, false);
}

static status_t MDIO_Write(uint8_t phyAddr, uint8_t regAddr, uint16_t data)
{
    return ENET_MDIOWrite(EXAMPLE_ENET, phyAddr, regAddr, data);
}

static status_t MDIO_Read(uint8_t phyAddr, uint8_t regAddr, uint16_t *pData)
{
    return ENET_MDIORead(EXAMPLE_ENET, phyAddr, regAddr, pData);
}

/*! @brief Build Frame for transmit. */
static void ENET_BuildPtpEventFrame(void)
{
    /* Build for PTP event message frame. */
    memcpy(&g_frame[0], &multicastAddr[0], 6);
    /* The six-byte source MAC address. */
    memcpy(&g_frame[6], &g_macAddr[0], 6);
    /* The type/length: if data length is used make sure it's smaller than 1500 */
    g_frame[12]    = 0x08U;
    g_frame[13]    = 0x00U;
    g_frame[0x0EU] = 0x40;
    /* Set the UDP PTP event port number. */
    g_frame[0x24U] = (kENET_PtpEventPort >> 8) & 0xFFU;
    g_frame[0x25U] = kENET_PtpEventPort & 0xFFU;
    /* Set the UDP protocol. */
    g_frame[0x17U] = 0x11U;
    /* Add ptp event message type: sync message. */
    g_frame[0x2AU] = ENET_PTP_SYNC_MSG;
    /* Add sequence id. */
    g_frame[0x48U]     = 0;
    g_frame[0x48U + 1] = 0;
}

static void enetCallback(ENET_Type *base,
                         enet_handle_t *handle,
#if FSL_FEATURE_ENET_QUEUE > 1
                         uint32_t ringId,
#endif /* FSL_FEATURE_ENET_QUEUE > 1 */
                         enet_event_t event,
                         enet_frame_info_t *frameInfo,
                         void *userData)
{
    /* Get frame info after whole frame transmits out */
    if ((event == kENET_TxEvent) && (frameInfo != NULL))
    {
        txFrameInfo   = *frameInfo;
        tx_frame_over = true;
    }
}

/*!
 * @brief Main function
 */
int main(void)
{
    phy_config_t phyConfig = {0};
    uint32_t length        = 0;
    uint32_t count         = 0;
    bool link              = false;
    bool autonego          = false;
    uint32_t txnumber      = 0;
    enet_config_t config;
    phy_speed_t speed;
    phy_duplex_t duplex;
    status_t result;

    enet_ptp_time_t ptpTime;
    status_t status;

    /* Hardware Initialization. */
    /* clang-format off */

    /* enetClk 250MHz */
    const clock_root_config_t enetClkCfg = {
        .clockOff = false,
	.mux = kCLOCK_WAKEUPAXI_ClockRoot_MuxSysPll1Pfd0, // 1000MHz
	.div = 4
    };

    /* enetRefClk 250MHz (For 125MHz TX_CLK ) */
    const clock_root_config_t enetRefClkCfg = {
        .clockOff = false,
	.mux = kCLOCK_ENETREF_ClockRoot_MuxSysPll1Pfd0Div2, // 500MHz
	.div = 2
    };

    /* enetPtpClk 100MHz */
    const clock_root_config_t enetPtpClkCfg = {
        .clockOff = false,
	.mux = kCLOCK_ENETTSTMR1_ClockRoot_MuxSysPll1Pfd1Div2, // 400MHz
	.div = 4
    };

    const clock_root_config_t lpi2cClkCfg = {
        .clockOff = false,
	.mux = 0, // 24MHz oscillator source
	.div = 1
    };
    /* clang-format on */

    BOARD_InitBootPins();
    BOARD_BootClockRUN();
    BOARD_InitDebugConsole();

    CLOCK_SetRootClock(EXAMPLE_CLOCK_ROOT, &enetClkCfg);
    CLOCK_SetRootClock(ENETREF_CLOCK_ROOT, &enetRefClkCfg);
    CLOCK_EnableClock(ENET_CLOCK_GATE);
    CLOCK_SetRootClock(PTP_CLOCK_ROOT, &enetPtpClkCfg);
    CLOCK_EnableClock(PTP_CLOCK_GATE);
    CLOCK_SetRootClock(BOARD_PCAL6524_I2C_CLOCK_ROOT, &lpi2cClkCfg);
    CLOCK_EnableClock(BOARD_PCAL6524_I2C_CLOCK_GATE);

    /* For a complete PHY reset of RTL8211FDI-CG, this pin must be asserted low for at least 10ms. And
     * wait for a further 30ms(for internal circuits settling time) before accessing the PHY register */
    pcal6524_handle_t handle;
    BOARD_InitPCAL6524(&handle);
    PCAL6524_SetDirection(&handle, (1 << BOARD_PCAL6524_ENET2_NRST), kPCAL6524_Output);
    PCAL6524_ClearPins(&handle, (1 << BOARD_PCAL6524_ENET2_NRST));
    SDK_DelayAtLeastUs(10000, SDK_DEVICE_MAXIMUM_CPU_CLOCK_FREQUENCY);
    PCAL6524_SetPins(&handle, (1 << BOARD_PCAL6524_ENET2_NRST));

    MDIO_Init();
    g_phy_resource.read  = MDIO_Read;
    g_phy_resource.write = MDIO_Write;

    PRINTF("\r\nENET PTP 1588 example start.\r\n");

    /* prepare the buffer configuration. */
    enet_buffer_config_t buffConfig[] = {{
        ENET_RXBD_NUM,
        ENET_TXBD_NUM,
        SDK_SIZEALIGN(ENET_RXBUFF_SIZE, APP_ENET_BUFF_ALIGNMENT),
        SDK_SIZEALIGN(ENET_TXBUFF_SIZE, APP_ENET_BUFF_ALIGNMENT),
        &g_rxBuffDescrip[0],
        &g_txBuffDescrip[0],
        &g_rxDataBuff[0][0],
        &g_txDataBuff[0][0],
        true,
        true,
        &txFrameInfoArray[0],
    }};

    /* Prepare the PTP configure */
    enet_ptp_config_t ptpConfig = {
        kENET_PtpTimerChannel1,
        PTP_CLK_FREQ,
    };

    /*
     * config.miiMode = kENET_RmiiMode;
     * config.miiSpeed = kENET_MiiSpeed100M;
     * config.miiDuplex = kENET_MiiFullDuplex;
     * config.rxMaxFrameLen = ENET_FRAME_MAX_FRAMELEN;
     */
    ENET_GetDefaultConfig(&config);
    config.callback  = enetCallback;

    /* The miiMode should be set according to the different PHY interfaces. */
#ifdef EXAMPLE_PHY_INTERFACE_RGMII
    config.miiMode = kENET_RgmiiMode;
#else
    config.miiMode = kENET_RmiiMode;
#endif
    phyConfig.phyAddr  = EXAMPLE_PHY_ADDRESS;
    phyConfig.ops      = EXAMPLE_PHY_OPS;
    phyConfig.resource = EXAMPLE_PHY_RESOURCE;
    phyConfig.autoNeg  = true;

    /* Initialize PHY and wait auto-negotiation over. */
    PRINTF("Wait for PHY init...\r\n");
    do
    {
        status = PHY_Init(&phyHandle, &phyConfig);
        if (status == kStatus_Success)
        {
            PRINTF("Wait for PHY link up...\r\n");
            /* Wait for auto-negotiation success and link up */
            count = PHY_AUTONEGO_TIMEOUT_COUNT;
            do
            {
                PHY_GetAutoNegotiationStatus(&phyHandle, &autonego);
                PHY_GetLinkStatus(&phyHandle, &link);
                if (autonego && link)
                {
                    break;
                }
            } while (--count);
            if (!autonego)
            {
                PRINTF("PHY Auto-negotiation failed. Please check the cable connection and link partner setting.\r\n");
            }
        }
    } while (!(link && autonego));

#if PHY_STABILITY_DELAY_US
    /* Wait a moment for PHY status to be stable. */
    SDK_DelayAtLeastUs(PHY_STABILITY_DELAY_US, SDK_DEVICE_MAXIMUM_CPU_CLOCK_FREQUENCY);
#endif

    /* Change the MII speed and duplex for actual link status. */
    PHY_GetLinkSpeedDuplex(&phyHandle, &speed, &duplex);
    config.miiSpeed  = (enet_mii_speed_t)speed;
    config.miiDuplex = (enet_mii_duplex_t)duplex;

#ifndef USER_DEFINED_MAC_ADDRESS
    /* Set special address for each chip. */
    SILICONID_ConvertToMacAddr(&g_macAddr);
#endif

    /* Initialize ENET. */
    ENET_Init(EXAMPLE_ENET, &g_handle, &config, &buffConfig[0], &g_macAddr[0], EXAMPLE_CLOCK_FREQ);
    /* Configure PTP */
    ENET_Ptp1588Configure(EXAMPLE_ENET, &g_handle, &ptpConfig);
    /* Add to multicast group to receive ptp multicast frame. */
    ENET_AddMulticastGroup(EXAMPLE_ENET, &multicastAddr[0]);
    /* Active ENET read. */
    ENET_ActiveRead(EXAMPLE_ENET);

    ENET_BuildPtpEventFrame();

    /* Enable Tx Reclaim and set callback to get timestamp */
    ENET_SetTxReclaim(&g_handle, true, 0);

    /* Check if the timestamp is running */
    for (count = 1; count <= 10; count++)
    {
        ENET_Ptp1588GetTimer(EXAMPLE_ENET, &g_handle, &ptpTime);
        PRINTF(" Get the %d-th time", count);
        PRINTF(" %d second,", (uint32_t)ptpTime.second);
        PRINTF(" %d nanosecond  \r\n", ptpTime.nanosecond);
        SDK_DelayAtLeastUs(200000, SDK_DEVICE_MAXIMUM_CPU_CLOCK_FREQUENCY);
    }

    while (1)
    {
        /* Get the Frame size */
        result = ENET_GetRxFrameSize(&g_handle, &length, 0);
        /* Call ENET_ReadFrame when there is a received frame. */
        if (length != 0)
        {
            /* Received valid frame. Deliver the rx buffer with the size equal to length. */
            uint8_t *data = (uint8_t *)malloc(length);
            uint32_t ts;
            enet_ptp_time_t rxPtpTime;
            result = ENET_ReadFrame(EXAMPLE_ENET, &g_handle, data, length, 0, &ts);
            if (result == kStatus_Success)
            {
                ENET_Ptp1588GetTimer(EXAMPLE_ENET, &g_handle, &rxPtpTime);
                /* If latest timestamp reloads after getting from Rx BD, then second - 1 to make sure the actual Rx
                 * timestamp is accurate */
                if (rxPtpTime.nanosecond < ts)
                {
                    rxPtpTime.second--;
                }

                PRINTF(" A frame received. the length %d ", length);
                PRINTF(" the timestamp is %d second,", (uint32_t)rxPtpTime.second);
                PRINTF(" %d nanosecond  \r\n", ts);
                PRINTF(" Dest Address %02x:%02x:%02x:%02x:%02x:%02x Src Address %02x:%02x:%02x:%02x:%02x:%02x \r\n",
                       data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9],
                       data[10], data[11]);
            }
            free(data);
        }
        else if (result == kStatus_ENET_RxFrameError)
        {
            /* Update the received buffer when error happened. */
            ENET_ReadFrame(EXAMPLE_ENET, &g_handle, NULL, 0, 0, NULL);
        }

        if (g_testTxNum < ENET_TRANSMIT_DATA_NUM)
        {
            /* Send a multicast frame when the PHY is link up. */
            if (kStatus_Success == PHY_GetLinkStatus(&phyHandle, &link))
            {
                if (link)
                {
                    g_testTxNum++;
                    tx_frame_over = false;
                    if (kStatus_Success ==
                        ENET_SendFrame(EXAMPLE_ENET, &g_handle, &g_frame[0], ENET_DATA_LENGTH, 0, true, NULL))
                    {
                        txnumber++;
                        /* Wait for Tx over then check timestamp */
                        while (!tx_frame_over)
                        {
                        }
                        PRINTF("The %d frame transmitted success!", txnumber);
                        if (txFrameInfo.isTsAvail)
                        {
                            PRINTF(" the timestamp is %d second,", (uint32_t)txFrameInfo.timeStamp.second);
                            PRINTF(" %d nanosecond  \r\n", txFrameInfo.timeStamp.nanosecond);
                        }
                        else
                        {
                            PRINTF(" Get transmit timestamp failed! \r\n");
                        }
                    }
                    else
                    {
                        PRINTF(" \r\nTransmit frame failed!\r\n");
                    }
                }
            }
        }
    }
}
