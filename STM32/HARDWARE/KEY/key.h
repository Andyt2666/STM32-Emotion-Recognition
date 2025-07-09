#ifndef __KEY_H
#define __KEY_H	 
#include "sys.h" 	 

// ���궨��ָ���µ�����
#define KEY0  GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_12) // ��ȡ����0 (PB12)
#define KEY1  GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_13) // ��ȡ����1 (PB13)
#define WK_UP GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_0)  // ��ȡ����3 (PA0), ���в���

#define KEY0_PRES 	1	//KEY0����
#define KEY1_PRES	  2	//KEY1����
#define WKUP_PRES   3	//KEY_UP����(��WK_UP/KEY_UP)

void KEY_Init(void);//IO��ʼ��
u8 KEY_Scan(u8);  	//����ɨ�躯��					  
#endif
