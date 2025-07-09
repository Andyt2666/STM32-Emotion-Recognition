#include "sys.h"
#include "usart.h"
#include "delay.h"
#include "led.h"
#include "serial.h"
#include "stdlib.h"

u16 t;  
u16 len;
u16 times=0; //不需要

extern int coords[2];
extern int emotion_code; // 新增：引用在main.c中定义的情感代码全局变量

void recieveData(void)
{
    // 使用状态机来解析，更清晰、更健壮
    u8 parse_state = 0; // 0:等待帧头#, 1:解析X, 2:解析Y, 3:解析Emotion
    char strX[4] = {0}; // 初始化为0
    char strY[4] = {0};
    char strEmotion[2] = {0}; // 用于存储情绪代码的字符串
    u8 cnt_x = 0;
    u8 cnt_y = 0;
    u8 cnt_emotion = 0;

    if(USART_RX_STA & 0x8000) // 如果接收完一帧数据
    {	
        LED1=!LED1; // 数据接收指示灯闪烁 (根据sys.h的定义，会操作PC13)

        len = USART_RX_STA & 0x3fff;

        for(t = 0; t < len; t++)
        {
            u8 ch = USART_RX_BUF[t]; // 用u8类型来接收

            // 根据分隔符改变状态
            if(ch == '#') { parse_state = 1; continue; }
            if(ch == '$') { parse_state = 2; continue; }
            if(ch == '&') { parse_state = 3; continue; } // 新增对'&'的判断

            // 根据当前状态，填充不同的字符串数组
            if(ch >= '0' && ch <= '9')
            {
                if(parse_state == 1 && cnt_x < 3) { strX[cnt_x++] = ch; }
                else if(parse_state == 2 && cnt_y < 3) { strY[cnt_y++] = ch; }
                else if(parse_state == 3 && cnt_emotion < 1) { strEmotion[cnt_emotion++] = ch; } // 新增解析Emotion
            }
        }

        // 转换字符串为整型
        coords[0] = atoi(strX);
        coords[1] = atoi(strY);
        emotion_code = atoi(strEmotion); // 新增：存储情感代码

        USART_RX_STA = 0; // 清空标志位
    }
    else
    {
        times++;
        if(times%30==0)LED0=!LED0;// 系统心跳灯
        delay_ms(10); 
    }
}
