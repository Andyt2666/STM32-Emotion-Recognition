#include "stm32f10x.h"
#include "delay.h"
#include "sys.h"
#include "led.h"
#include "key.h"
#include "usart.h"
#include "serial.h"
#include "timer.h"
#include "pid.h"
#include "OLED.h" // 1. 包含OLED头文件

// ================== 全局变量定义区 ==================
PID_TypeDef PID_x, PID_y; // 定义两个PID结构体

int coords[2];            // 用于存储从PC接收到的当前坐标 (x, y)
int emotion_code = 0;     // 用于存储从PC接收到的情感代码 (0=默认, 1=开心, 2=伤心, 3=惊讶, 4=生气)

u16 targetX = 320;        // 目标X坐标 (屏幕中心)
u16 targetY = 240;        // 目标Y坐标 (屏幕中心)

// 【新增】引用在OLED.c中定义的外部表情图案数据
// 这样main函数才能使用这些数组
extern const uint8_t Happy_Pattern[];
extern const uint8_t Sad_Pattern[];
extern const uint8_t Surprised_Pattern[];
extern const uint8_t Angry_Pattern[];


int main(void)
{
	// --- 定义局部变量 ---
	u16 pwmval_x, pwmval_y; // 用于存储计算出的PWM占空比(CCR寄存器值)

	// --- 系统与外设初始化 ---
	delay_init();	    	 // 延时函数初始化	  
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2); // 设置NVIC中断分组2
	uart_init(115200);	     // 串口初始化为115200波特率
 	LED_Init();			     // LED端口初始化
	KEY_Init();              // 按键初始化 (即使不用，保留以防编译报错)
	TIM3_PWM_Init(9999,143); // PWM初始化, 频率为: 72M / (9999+1) / (143+1) = 50Hz
	OLED_Init();             // 【新增】OLED初始化

	// --- PID参数初始化 ---
	// 这些参数您后续可以根据实际效果进行微调
	pid_init(0.03, 0, 0.30, &PID_x); // 初始化X轴的PD控制器
	pid_init(0.03, 0, 0.30, &PID_y); // 初始化Y轴的PD控制器
	
	// --- 变量初始值设定 ---
	coords[0] = 320; // 默认坐标在屏幕中心
	coords[1] = 240;
	
	// 初始化舵机角度到中间位置 (大约90度)
	// 这个初始值(650-750之间)需要您根据舵机实际情况微调，确保上电时舵机在正中间
	pwmval_x = 730;
	pwmval_y = 730;
	
	// --- 主循环 ---
 	while(1)
	{
		// 1. 从串口读取数据，更新 coords 和 emotion_code 全局变量
		recieveData();
		
		// 2. 使用PID计算舵机PWM占空比
		// 控制思路: pwmval 在上一次的基础上，加上本次PID计算出的“增量”
		pwmval_x = pwmval_x + pid(coords[0], targetX, &PID_x);
		pwmval_y = pwmval_y - pid(coords[1], targetY, &PID_y); // Y轴方向相反，用减法

		// 3. PWM值限幅，防止舵机超出物理极限而堵转
		if(pwmval_x < 300) pwmval_x = 300;
		if(pwmval_x > 1200) pwmval_x = 1200;
		if(pwmval_y < 400) pwmval_y = 400;
		if(pwmval_y > 900) pwmval_y = 900;
		
		// 4. 将计算出的新PWM值写入定时器寄存器，驱动舵机转动
		TIM_SetCompare1(TIM3, pwmval_x); // PA6, 横向舵机
		TIM_SetCompare2(TIM3, pwmval_y); // PA7, 竖向舵机
		
		// 5. 【新增】根据情绪代码，在OLED上显示对应的表情
		// 在 main.c 的 while(1) 循环中
switch(emotion_code)
{
    case 1: // Happy
        OLED_ShowPattern(48, 2, Happy_Pattern, 32, 32); // 增加宽度32, 高度32
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
		// 注意：为了避免OLED清屏过于频繁导致闪烁，
		// 只有在情绪代码变为0时（例如，Python端在没检测到脸时发送情绪0），
		// 才执行一次OLED_Clear()会是更好的体验。
		// 目前的逻辑是，一旦显示了表情，它会一直留在屏幕上，直到下一个表情指令到来。
	}	 
}