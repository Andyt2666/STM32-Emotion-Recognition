#ifndef __KEY_H
#define __KEY_H	 
#include "sys.h" 	 

// 将宏定义指向新的引脚
#define KEY0  GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_12) // 读取按键0 (PB12)
#define KEY1  GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_13) // 读取按键1 (PB13)
#define WK_UP GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_0)  // 读取按键3 (PA0), 此行不变

#define KEY0_PRES 	1	//KEY0按下
#define KEY1_PRES	  2	//KEY1按下
#define WKUP_PRES   3	//KEY_UP按下(即WK_UP/KEY_UP)

void KEY_Init(void);//IO初始化
u8 KEY_Scan(u8);  	//按键扫描函数					  
#endif
