#include "led.h"

    
//LED IO��ʼ��
void LED_Init(void)
{
    GPIO_InitTypeDef  GPIO_InitStructure;
    
    // ֻʹ��GPIOCʱ��
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC, ENABLE);
    
    // ��ʼ������LED -> PC.13
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_13;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP; // �������
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOC, &GPIO_InitStructure);
    
    // ע�⣺PC13ͨ���ǵ͵�ƽ����������Ĭ�Ͽ�������Ϊ�ߵ�ƽ��Ϩ��
    GPIO_SetBits(GPIOC, GPIO_Pin_13);
}
 
