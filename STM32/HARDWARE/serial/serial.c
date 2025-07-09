#include "sys.h"
#include "usart.h"
#include "delay.h"
#include "led.h"
#include "serial.h"
#include "stdlib.h"

u16 t;  
u16 len;
u16 times=0; //����Ҫ

extern int coords[2];
extern int emotion_code; // ������������main.c�ж������д���ȫ�ֱ���

void recieveData(void)
{
    // ʹ��״̬����������������������׳
    u8 parse_state = 0; // 0:�ȴ�֡ͷ#, 1:����X, 2:����Y, 3:����Emotion
    char strX[4] = {0}; // ��ʼ��Ϊ0
    char strY[4] = {0};
    char strEmotion[2] = {0}; // ���ڴ洢����������ַ���
    u8 cnt_x = 0;
    u8 cnt_y = 0;
    u8 cnt_emotion = 0;

    if(USART_RX_STA & 0x8000) // ���������һ֡����
    {	
        LED1=!LED1; // ���ݽ���ָʾ����˸ (����sys.h�Ķ��壬�����PC13)

        len = USART_RX_STA & 0x3fff;

        for(t = 0; t < len; t++)
        {
            u8 ch = USART_RX_BUF[t]; // ��u8����������

            // ���ݷָ����ı�״̬
            if(ch == '#') { parse_state = 1; continue; }
            if(ch == '$') { parse_state = 2; continue; }
            if(ch == '&') { parse_state = 3; continue; } // ������'&'���ж�

            // ���ݵ�ǰ״̬����䲻ͬ���ַ�������
            if(ch >= '0' && ch <= '9')
            {
                if(parse_state == 1 && cnt_x < 3) { strX[cnt_x++] = ch; }
                else if(parse_state == 2 && cnt_y < 3) { strY[cnt_y++] = ch; }
                else if(parse_state == 3 && cnt_emotion < 1) { strEmotion[cnt_emotion++] = ch; } // ��������Emotion
            }
        }

        // ת���ַ���Ϊ����
        coords[0] = atoi(strX);
        coords[1] = atoi(strY);
        emotion_code = atoi(strEmotion); // �������洢��д���

        USART_RX_STA = 0; // ��ձ�־λ
    }
    else
    {
        times++;
        if(times%30==0)LED0=!LED0;// ϵͳ������
        delay_ms(10); 
    }
}
