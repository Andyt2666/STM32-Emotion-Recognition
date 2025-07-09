#include "stm32f10x.h"
#include "delay.h"
#include "sys.h"
#include "led.h"
#include "key.h"
#include "usart.h"
#include "serial.h"
#include "timer.h"
#include "pid.h"
#include "OLED.h" // 1. ����OLEDͷ�ļ�

// ================== ȫ�ֱ��������� ==================
PID_TypeDef PID_x, PID_y; // ��������PID�ṹ��

int coords[2];            // ���ڴ洢��PC���յ��ĵ�ǰ���� (x, y)
int emotion_code = 0;     // ���ڴ洢��PC���յ�����д��� (0=Ĭ��, 1=����, 2=����, 3=����, 4=����)

u16 targetX = 320;        // Ŀ��X���� (��Ļ����)
u16 targetY = 240;        // Ŀ��Y���� (��Ļ����)

// ��������������OLED.c�ж�����ⲿ����ͼ������
// ����main��������ʹ����Щ����
extern const uint8_t Happy_Pattern[];
extern const uint8_t Sad_Pattern[];
extern const uint8_t Surprised_Pattern[];
extern const uint8_t Angry_Pattern[];


int main(void)
{
	// --- ����ֲ����� ---
	u16 pwmval_x, pwmval_y; // ���ڴ洢�������PWMռ�ձ�(CCR�Ĵ���ֵ)

	// --- ϵͳ�������ʼ�� ---
	delay_init();	    	 // ��ʱ������ʼ��	  
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2); // ����NVIC�жϷ���2
	uart_init(115200);	     // ���ڳ�ʼ��Ϊ115200������
 	LED_Init();			     // LED�˿ڳ�ʼ��
	KEY_Init();              // ������ʼ�� (��ʹ���ã������Է����뱨��)
	TIM3_PWM_Init(9999,143); // PWM��ʼ��, Ƶ��Ϊ: 72M / (9999+1) / (143+1) = 50Hz
	OLED_Init();             // ��������OLED��ʼ��

	// --- PID������ʼ�� ---
	// ��Щ�������������Ը���ʵ��Ч������΢��
	pid_init(0.03, 0, 0.30, &PID_x); // ��ʼ��X���PD������
	pid_init(0.03, 0, 0.30, &PID_y); // ��ʼ��Y���PD������
	
	// --- ������ʼֵ�趨 ---
	coords[0] = 320; // Ĭ����������Ļ����
	coords[1] = 240;
	
	// ��ʼ������Ƕȵ��м�λ�� (��Լ90��)
	// �����ʼֵ(650-750֮��)��Ҫ�����ݶ��ʵ�����΢����ȷ���ϵ�ʱ��������м�
	pwmval_x = 730;
	pwmval_y = 730;
	
	// --- ��ѭ�� ---
 	while(1)
	{
		// 1. �Ӵ��ڶ�ȡ���ݣ����� coords �� emotion_code ȫ�ֱ���
		recieveData();
		
		// 2. ʹ��PID������PWMռ�ձ�
		// ����˼·: pwmval ����һ�εĻ����ϣ����ϱ���PID������ġ�������
		pwmval_x = pwmval_x + pid(coords[0], targetX, &PID_x);
		pwmval_y = pwmval_y - pid(coords[1], targetY, &PID_y); // Y�᷽���෴���ü���

		// 3. PWMֵ�޷�����ֹ������������޶���ת
		if(pwmval_x < 300) pwmval_x = 300;
		if(pwmval_x > 1200) pwmval_x = 1200;
		if(pwmval_y < 400) pwmval_y = 400;
		if(pwmval_y > 900) pwmval_y = 900;
		
		// 4. �����������PWMֵд�붨ʱ���Ĵ������������ת��
		TIM_SetCompare1(TIM3, pwmval_x); // PA6, ������
		TIM_SetCompare2(TIM3, pwmval_y); // PA7, ������
		
		// 5. �������������������룬��OLED����ʾ��Ӧ�ı���
		// �� main.c �� while(1) ѭ����
switch(emotion_code)
{
    case 1: // Happy
        OLED_ShowPattern(48, 2, Happy_Pattern, 32, 32); // ���ӿ��32, �߶�32
        break;
    case 2: // Sad
        OLED_ShowPattern(48, 2, Sad_Pattern, 32, 32);
        break;
    case 3: // Surprised
        OLED_ShowPattern(48, 2, Surprised_Pattern, 32, 32);
        break;
    case 4: // Angry
        OLED_ShowPattern(48, 2, Angry_Pattern, 32, 32);
        break;
    default:
        OLED_Clear();
        break;
}
		// ע�⣺Ϊ�˱���OLED��������Ƶ��������˸��
		// ֻ�������������Ϊ0ʱ�����磬Python����û��⵽��ʱ��������0����
		// ��ִ��һ��OLED_Clear()���Ǹ��õ����顣
		// Ŀǰ���߼��ǣ�һ����ʾ�˱��飬����һֱ������Ļ�ϣ�ֱ����һ������ָ�����
	}	 
}