# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 18:03:22 2018

@author: James Wu
"""

import cv2
import numpy as np
import serial
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from tensorflow.keras.models import load_model

# 设置matplotlib使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 检查TensorFlow版本
import tensorflow as tf
print(f"\nTensorFlow版本: {tf.__version__}")

# 使用新的模型路径
MODEL_PATH = r"E:\Study\mode\modelv2.h5"  # 使用原始字符串避免转义问题
print(f"模型路径: {MODEL_PATH}")

# 加载情绪检测模型
try:
    print(f"\n正在加载情绪检测模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    
    # 检查模型文件
    import h5py
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            print("\n模型文件内容:")
            def print_attrs(name, obj):
                print(f"- 找到对象: {name}")
                for key, val in obj.attrs.items():
                    print(f"  属性: {key} = {val}")
            f.visititems(print_attrs)
    except Exception as e:
        print(f"\n警告：模型文件格式检查失败")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
    
    # 尝试加载模型
    print("\n尝试加载模型...")
    # 使用自定义对象作用域加载模型
    with tf.keras.utils.custom_object_scope({}):
        emotion_model = load_model(MODEL_PATH, compile=False)
    print("模型加载成功")
    
    # 打印模型结构
    print("\n模型结构:")
    emotion_model.summary()
    
    # 原模型的7种情绪标签
    ALL_EMOTIONS = ['生气', '厌恶', '害怕', '开心', '悲伤', '惊讶', '正常']
    # 我们需要的4种情绪标签
    EMOTIONS = ["生气", "开心", "难过", "惊讶"]
    print(f"\n成功加载情绪检测模型: {MODEL_PATH}")
except Exception as e:
    print(f"\n警告：无法加载情绪检测模型({MODEL_PATH})")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {str(e)}")
    import traceback
    print("详细错误信息:")
    traceback.print_exc()
    print("\n将只进行人脸跟踪")
    emotion_model = None

def cv2AddChineseText(img, text, position, textColor=(0, 0, 255), textSize=30):
    """
    给图片添加中文文字
    :param img: opencv图片
    :param text: 中文文字
    :param position: 文字位置
    :param textColor: 文字颜色
    :param textSize: 文字大小
    :return: 已添加文字的图片
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # 设置字体，请确保系统中已安装了该字体
    fontStyle = ImageFont.truetype("simhei.ttf", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 创建保存数据的目录
save_dir = 'experiment_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(os.path.join(save_dir, 'gray_images')):
    os.makedirs(os.path.join(save_dir, 'gray_images'))
if not os.path.exists(os.path.join(save_dir, 'detection_images')):
    os.makedirs(os.path.join(save_dir, 'detection_images'))

# 目标中心点坐标（期望值）
TARGET_X = 320
TARGET_Y = 240

# 用于记录坐标变化
coord_history = {
    'x': [], 'y': [], 'time': [],
    'error_x': [], 'error_y': [],  # 误差
    'abs_error_x': [], 'abs_error_y': []  # 绝对误差
}
start_time = time.time()
last_save_time = start_time  # 添加最后保存时间记录

# 性能指标
performance_metrics = {
    'max_overshoot_x': 0,  # X方向最大超调量
    'max_overshoot_y': 0,  # Y方向最大超调量
    'settling_time_x': 0,  # X方向调节时间
    'settling_time_y': 0,  # Y方向调节时间
    'steady_state_error_x': 0,  # X方向稳态误差
    'steady_state_error_y': 0,  # Y方向稳态误差
}

# 暂停标志
is_paused = False

# 添加全局变量用于控制程序退出
running = True

# 修正后的路径
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def save_all_data():
    """保存所有数据，包括性能指标、轨迹图和响应分析图"""
    # 计算性能指标
    calculate_performance_metrics()
    
    # 保存轨迹图和响应分析图
    save_trajectory_plot()
    save_response_plots()
    
    # 保存性能指标到文本文件
    with open(os.path.join(save_dir, 'performance_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("系统性能指标分析报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("X方向性能指标：\n")
        f.write(f"最大超调量: {performance_metrics['max_overshoot_x']:.2f}%\n")
        f.write(f"调节时间: {performance_metrics['settling_time_x']:.2f}秒\n")
        f.write(f"稳态误差: {performance_metrics['steady_state_error_x']:.2f}像素\n\n")
        f.write("Y方向性能指标：\n")
        f.write(f"最大超调量: {performance_metrics['max_overshoot_y']:.2f}%\n")
        f.write(f"调节时间: {performance_metrics['settling_time_y']:.2f}秒\n")
        f.write(f"稳态误差: {performance_metrics['steady_state_error_y']:.2f}像素\n")

def calculate_performance_metrics():
    """计算系统性能指标"""
    if len(coord_history['time']) < 2:
        return
    
    # 计算误差
    errors_x = np.array(coord_history['error_x'])
    errors_y = np.array(coord_history['error_y'])
    abs_errors_x = np.array(coord_history['abs_error_x'])
    abs_errors_y = np.array(coord_history['abs_error_y'])
    times = np.array(coord_history['time'])
    
    # 计算超调量
    if len(errors_x) > 0:
        max_error_x = np.max(np.abs(errors_x))
        max_error_y = np.max(np.abs(errors_y))
        performance_metrics['max_overshoot_x'] = (max_error_x / TARGET_X) * 100  # 百分比
        performance_metrics['max_overshoot_y'] = (max_error_y / TARGET_Y) * 100  # 百分比
    
    # 计算调节时间（误差进入并保持在±5%范围内所需时间）
    error_threshold = 0.05  # 5%误差带
    x_threshold = TARGET_X * error_threshold
    y_threshold = TARGET_Y * error_threshold
    
    # X方向调节时间
    for i in range(len(abs_errors_x)):
        if i > 10 and all(err <= x_threshold for err in abs_errors_x[i-10:i+1]):
            performance_metrics['settling_time_x'] = times[i] - times[0]
            break
    
    # Y方向调节时间
    for i in range(len(abs_errors_y)):
        if i > 10 and all(err <= y_threshold for err in abs_errors_y[i-10:i+1]):
            performance_metrics['settling_time_y'] = times[i] - times[0]
            break
    
    # 计算稳态误差（最后30个点的平均误差）
    if len(errors_x) > 30:
        performance_metrics['steady_state_error_x'] = np.mean(np.abs(errors_x[-30:]))
        performance_metrics['steady_state_error_y'] = np.mean(np.abs(errors_y[-30:]))

def save_response_plots():
    """保存响应曲线"""
    try:
        # 创建一个新的图形，包含三个子图
        plt.figure(figsize=(15, 12))
        
        # 1. X方向误差响应曲线
        plt.subplot(3, 1, 1)
        plt.plot(coord_history['time'], coord_history['error_x'], 'r-', label='X方向误差')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # 零线
        plt.title('X方向误差响应曲线')
        plt.xlabel('时间 (秒)')
        plt.ylabel('误差 (像素)')
        plt.grid(True)
        plt.legend()
        
        # 2. Y方向误差响应曲线
        plt.subplot(3, 1, 2)
        plt.plot(coord_history['time'], coord_history['error_y'], 'b-', label='Y方向误差')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # 零线
        plt.title('Y方向误差响应曲线')
        plt.xlabel('时间 (秒)')
        plt.ylabel('误差 (像素)')
        plt.grid(True)
        plt.legend()
        
        # 3. 误差的相平面图（X误差 vs Y误差）
        plt.subplot(3, 1, 3)
        plt.plot(coord_history['error_x'], coord_history['error_y'], 'g-', label='误差轨迹')
        plt.plot(coord_history['error_x'][0], coord_history['error_y'][0], 'ro', label='起始点')
        plt.plot(coord_history['error_x'][-1], coord_history['error_y'][-1], 'bo', label='终止点')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.title('误差相平面图')
        plt.xlabel('X方向误差 (像素)')
        plt.ylabel('Y方向误差 (像素)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'response_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"保存响应曲线时出错: {str(e)}")

class EmotionAnalyzer:
    def __init__(self):
        # 创建保存目录
        self.analysis_dir = 'emotion_analysis'
        self.confidence_history = []
        self.emotion_history = []
        self.time_points = []
        self.start_time = time.time()
        self.frame_count = 0
        self.last_save_time = time.time()  # 添加最后保存时间记录
        
        # 确保目录存在
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
            
        # 子目录
        self.prob_dir = os.path.join(self.analysis_dir, 'probability_plots')
        self.conf_dir = os.path.join(self.analysis_dir, 'confidence_plots')
        self.trend_dir = os.path.join(self.analysis_dir, 'trend_analysis')
        
        for dir_path in [self.prob_dir, self.conf_dir, self.trend_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def save_emotion_probabilities(self, predictions, frame_count):
        """保存情绪概率分布图"""
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(ALL_EMOTIONS, predictions)
            plt.title(f'情绪预测概率分布 (帧 {frame_count})')
            plt.xlabel('情绪类别')
            plt.ylabel('概率')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.prob_dir, f'emotion_prob_{frame_count}.png'))
            plt.close()
        except Exception as e:
            print(f"保存概率分布图出错: {str(e)}")
    
    def update_confidence_history(self, confidence, emotion):
        """更新置信度历史"""
        try:
            current_time = time.time() - self.start_time
            self.time_points.append(current_time)
            self.confidence_history.append(confidence)
            self.emotion_history.append(emotion)
            
            # 每30帧保存一次趋势图（约1秒）
            if len(self.confidence_history) % 30 == 0:
                self.save_trend_analysis()
                
            # 每5秒保存一次报告
            if current_time - self.last_save_time >= 5:
                self.save_analysis_report()
                self.last_save_time = current_time
                
        except Exception as e:
            print(f"更新历史数据时出错: {str(e)}")
    
    def save_trend_analysis(self):
        """保存趋势分析图"""
        if not self.confidence_history:
            return
            
        try:
            # 1. 置信度随时间变化
            plt.figure(figsize=(12, 6))
            plt.plot(self.time_points, self.confidence_history, 'b-', label='置信度')
            plt.title('情绪识别置信度随时间变化')
            plt.xlabel('时间 (秒)')
            plt.ylabel('置信度')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.conf_dir, 'confidence_trend.png'))
            plt.close()
            
            # 2. 情绪分布统计
            plt.figure(figsize=(10, 6))
            emotion_counts = {}
            for emotion in self.emotion_history:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            plt.bar(emotions, counts)
            plt.title('情绪分布统计')
            plt.xlabel('情绪类别')
            plt.ylabel('出现次数')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.trend_dir, 'emotion_distribution.png'))
            plt.close()
            
            # 3. 情绪变化热图
            plt.figure(figsize=(12, 6))
            emotion_types = list(set(self.emotion_history))
            emotion_indices = [emotion_types.index(e) for e in self.emotion_history]
            plt.scatter(self.time_points, emotion_indices, c=self.confidence_history, 
                       cmap='viridis', alpha=0.6)
            plt.colorbar(label='置信度')
            plt.yticks(range(len(emotion_types)), emotion_types)
            plt.title('情绪变化热图')
            plt.xlabel('时间 (秒)')
            plt.ylabel('情绪类别')
            plt.savefig(os.path.join(self.trend_dir, 'emotion_heatmap.png'))
            plt.close()
            
        except Exception as e:
            print(f"保存趋势分析图出错: {str(e)}")
    
    def save_analysis_report(self):
        """保存分析报告"""
        try:
            report_path = os.path.join(self.analysis_dir, 'emotion_analysis_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("情绪识别分析报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"总帧数: {self.frame_count}\n")
                f.write(f"当前运行时间: {self.time_points[-1]:.2f} 秒\n\n")
                
                # 情绪统计
                emotion_counts = {}
                for emotion in self.emotion_history:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                f.write("情绪分布统计:\n")
                for emotion, count in emotion_counts.items():
                    percentage = (count / len(self.emotion_history)) * 100
                    f.write(f"{emotion}: {count} 次 ({percentage:.2f}%)\n")
                
                if self.confidence_history:
                    f.write("\n置信度统计:\n")
                    f.write(f"平均置信度: {np.mean(self.confidence_history):.3f}\n")
                    f.write(f"最高置信度: {np.max(self.confidence_history):.3f}\n")
                    f.write(f"最低置信度: {np.min(self.confidence_history):.3f}\n")
            
        except Exception as e:
            print(f"保存分析报告时出错: {str(e)}")
            
    def save_final_analysis(self):
        """程序退出时保存最终分析"""
        print("\n正在保存最终情绪分析数据...")
        try:
            self.save_trend_analysis()
            self.save_analysis_report()
            print("所有分析数据已保存完成")
        except Exception as e:
            print(f"保存最终分析数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()

# 创建分析器实例
emotion_analyzer = EmotionAnalyzer()

def detect_emotion(face_img):
    """
    检测人脸情绪
    :param face_img: 人脸图像区域
    :return: 情绪代码 (1: 开心, 2: 难过, 3: 惊讶, 4: 生气)
    """
    # 如果模型未加载成功，直接返回默认情绪
    if emotion_model is None:
        return 1, "开心"  # 默认为开心
        
    try:
        # 图像预处理和增强
        # 1. 调整大小，确保面部特征清晰
        face_img = cv2.resize(face_img, (48, 48))
        
        # 2. 转换为灰度图
        if len(face_img.shape) == 3:
            gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = face_img.copy()
            
        # 3. 图像增强
        # 3.1 直方图均衡化
        gray_img = cv2.equalizeHist(gray_img)
        
        # 3.2 高斯模糊去噪
        gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        
        # 3.3 对比度增强
        alpha = 1.5  # 对比度增强因子
        beta = 10    # 亮度增强因子
        gray_img = cv2.convertScaleAbs(gray_img, alpha=alpha, beta=beta)
        
        # 3.4 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_img = clahe.apply(gray_img)
        
        # 4. 归一化
        processed_img = gray_img.astype('float32') / 255.0
        
        # 5. 保存处理后的图像用于调试
        debug_img_path = os.path.join(emotion_analyzer.analysis_dir, 'debug_face.jpg')
        cv2.imwrite(debug_img_path, gray_img)
        
        # 6. 扩展维度
        processed_img = np.expand_dims(processed_img, -1)
        processed_img = np.expand_dims(processed_img, 0)
        
        # 预测情绪
        predictions = emotion_model.predict(processed_img, verbose=0)[0]
        
        # 获取前两个最高概率的情绪
        top_2_indices = np.argsort(predictions)[-2:][::-1]
        top_2_emotions = [(ALL_EMOTIONS[i], predictions[i]) for i in top_2_indices]
        
        print("\n情绪预测概率:")
        for emotion, prob in zip(ALL_EMOTIONS, predictions):
            print(f"{emotion}: {prob:.3f}")
            
        print("\n最可能的两种情绪:")
        for emotion, prob in top_2_emotions:
            print(f"{emotion}: {prob:.3f}")
        
        # 获取最高概率的情绪
        max_index = top_2_indices[0]
        original_emotion = ALL_EMOTIONS[max_index]
        max_probability = predictions[max_index]
        
        # 设置动态置信度阈值
        base_threshold = 0.25  # 基础阈值
        second_best_prob = predictions[top_2_indices[1]]
        prob_diff = max_probability - second_best_prob
        
        # 如果最高概率与第二高概率差距显著，降低阈值要求
        if prob_diff > 0.2:
            confidence_threshold = base_threshold * 0.8
        else:
            confidence_threshold = base_threshold
        
        print(f"\n置信度阈值: {confidence_threshold:.3f}")
        print(f"最高概率: {max_probability:.3f}")
        print(f"概率差: {prob_diff:.3f}")
        
        if max_probability < confidence_threshold:
            print(f"置信度({max_probability:.3f})低于阈值({confidence_threshold})，返回默认情绪")
            emotion_analyzer.update_confidence_history(max_probability, "开心")
            return 1, "开心"  # 默认为开心
        
        # 将7种情绪映射到4种情绪，优化映射规则
        emotion_mapping = {
            '生气': (4, "生气"),    # 生气 -> 生气(4)
            '开心': (1, "开心"),    # 开心 -> 开心(1)
            '悲伤': (2, "难过"),    # 悲伤 -> 难过(2)
            '惊讶': (3, "惊讶"),    # 惊讶 -> 惊讶(3)
            '厌恶': (4, "生气"),    # 厌恶 -> 生气(4)
            '害怕': (3, "惊讶"),    # 害怕 -> 惊讶(3)，修改映射
            '正常': (1, "开心"),    # 正常 -> 开心(1)
        }
        
        emotion_code, emotion_name = emotion_mapping[original_emotion]
        print(f"检测到情绪: {original_emotion} -> {emotion_name} (置信度: {max_probability:.3f})")
        
        # 更新分析数据
        emotion_analyzer.update_confidence_history(max_probability, emotion_name)
        
        return emotion_code, emotion_name
        
    except Exception as e:
        print(f"情绪检测出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1, "开心"  # 默认为开心

def Detection(frame, frame_count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #转换为灰度图
    
    # 保存灰度图
    if frame_count % 30 == 0:  # 每30帧保存一次
        try:
            gray_filename = os.path.join(save_dir, 'gray_images', f'gray_{frame_count}.jpg')
            cv2.imwrite(gray_filename, gray)
        except Exception as e:
            print(f"保存灰度图时出错: {str(e)}")
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #人脸检测
    
    emotion_code = 2  # 默认情绪代码：难过
    
    if len(faces)>0: 
        for (x,y,w,h) in faces:
            # 提取人脸区域进行情绪检测
            face_roi = frame[y:y+h, x:x+w]
            emotion_code, emotion_name = detect_emotion(face_roi)
            
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            X = x+w//2
            Y = y+h//2
            center_pt=(X,Y)   #各人脸中点坐标
            cv2.circle(frame, center_pt, 8, (0,0,255), -1)   #绘制各人脸中点
            
            # 在人脸上方显示情绪
            frame = cv2AddChineseText(frame, f"情绪: {emotion_name}", (x, y-30))
            
        centroid_X = int(np.mean(faces, axis=0)[0] + np.mean(faces, axis=0)[2]//2)
        centroid_Y = int(np.mean(faces, axis=0)[1] + np.mean(faces, axis=0)[3]//2)
        centroid_pt=(centroid_X,centroid_Y)
        cv2.circle(frame, centroid_pt, 8, (0,0,255), -1)
    else:
        centroid_X = TARGET_X
        centroid_Y = TARGET_Y
        emotion_code = 2  # 没有检测到人脸时，默认为难过
    
    # 计算误差
    error_x = centroid_X - TARGET_X
    error_y = centroid_Y - TARGET_Y
    
    # 记录坐标和误差历史
    current_time = time.time() - start_time
    coord_history['x'].append(centroid_X)
    coord_history['y'].append(centroid_Y)
    coord_history['time'].append(current_time)
    coord_history['error_x'].append(error_x)
    coord_history['error_y'].append(error_y)
    coord_history['abs_error_x'].append(abs(error_x))
    coord_history['abs_error_y'].append(abs(error_y))
    
    # 在画面上显示当前误差（使用中文）
    frame = cv2AddChineseText(frame, f"X轴误差: {error_x:+d}", (10, 30))
    frame = cv2AddChineseText(frame, f"Y轴误差: {error_y:+d}", (10, 60))
    
    if is_paused:
        frame = cv2AddChineseText(frame, "已暂停", (10, 90))
    
    #==========================================================================
    #     绘制参考线
    #==========================================================================
    x = 0;
    y = 0;
    w = TARGET_X;
    h = TARGET_Y;
    
    rectangle_pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32) #最小包围矩形各顶点
    cv2.polylines(frame, [rectangle_pts], True, (0,255,0), 2) #绘制最小包围矩形
    
    x2 = TARGET_X;
    y2 = TARGET_Y;
    rectangle_pts2 = np.array([[x2,y2],[x2+w,y2],[x2+w,y2+h],[x2,y2+h]], np.int32) #最小包围矩形各顶点
    cv2.polylines(frame, [rectangle_pts2], True, (0,255,0), 2) #绘制最小包围矩形

    # 保存检测结果图像
    if frame_count % 30 == 0:  # 每30帧保存一次
        try:
            detection_filename = os.path.join(save_dir, 'detection_images', f'detection_{frame_count}.jpg')
            cv2.imwrite(detection_filename, frame)
        except Exception as e:
            print(f"保存检测结果图像时出错: {str(e)}")
    
    #==========================================================================
    #     显示
    #==========================================================================
    cv2.imshow('frame',frame)
    
    return centroid_X, centroid_Y, emotion_code

def save_trajectory_plot():
    """保存轨迹图"""
    try:
        plt.figure(figsize=(12, 8))
        
        # 绘制X坐标随时间变化
        plt.subplot(2, 1, 1)
        plt.plot(coord_history['time'], coord_history['x'], 'r-', label='X坐标')
        plt.axhline(y=TARGET_X, color='k', linestyle='--', label='目标值')
        plt.title('人脸中心X坐标随时间变化')
        plt.xlabel('时间 (秒)')
        plt.ylabel('X坐标')
        plt.grid(True)
        plt.legend()
        
        # 绘制Y坐标随时间变化
        plt.subplot(2, 1, 2)
        plt.plot(coord_history['time'], coord_history['y'], 'b-', label='Y坐标')
        plt.axhline(y=TARGET_Y, color='k', linestyle='--', label='目标值')
        plt.title('人脸中心Y坐标随时间变化')
        plt.xlabel('时间 (秒)')
        plt.ylabel('Y坐标')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'trajectory_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"保存轨迹图时出错: {str(e)}")
                
#==============================================================================
#   ****************************主函数入口***********************************
#==============================================================================

# 设置串口参数
ser = serial.Serial()
ser.baudrate = 115200    # 设置比特率为115200bps
ser.port = 'COM5'      # 单片机接在哪个串口，就写哪个串口。这里默认接在"COM5"端口
ser.open()             # 打开串口

#先发送一个中心坐标使初始化时云台保持水平
data = '#'+str(TARGET_X)+'$'+str(TARGET_Y)+'\r\n'
ser.write(data.encode())        

cap = cv2.VideoCapture(0) #打开摄像头

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

# 读取一帧测试
ret, frame = cap.read()
if not ret:
    print("错误：无法从摄像头读取图像")
    exit()

frame_count = 0  # 添加帧计数器

def on_keyboard(key):
    """处理键盘事件"""
    global is_paused, running
    if key == 27:  # ESC键
        print("\n检测到ESC键，准备退出程序...")
        running = False
        return True
    elif key == 32:  # 空格键
        is_paused = not is_paused
        print("程序已" + ("暂停" if is_paused else "继续"))
    return False

try:
    print("\n=== 程序启动 ===")
    print("按空格键：暂停/继续")
    print("按ESC键：退出程序")
    print("按q键：退出程序")
    print("==================\n")
    
    while(cap.isOpened() and running):
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面，程序退出")
                break
                
            frame_count += 1
            emotion_analyzer.frame_count = frame_count  # 更新帧计数
            
            X, Y, emotion = Detection(frame, frame_count)
            
            if(X<10000):
                print('X = ')
                print(X)
                print('Y =')
                print(Y)
                print('Emotion =')
                print(emotion)
                data = '#'+str(X)+'$'+str(Y)+'&'+str(emotion)+'\r\n'
                ser.write(data.encode())
        
        # 处理键盘事件
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC键或q键
            print("\n检测到退出指令，准备结束程序...")
            break
        elif key == 32:  # 空格键
            is_paused = not is_paused
            print("程序已" + ("暂停" if is_paused else "继续"))

except KeyboardInterrupt:
    print("\n检测到键盘中断，准备结束程序...")
except Exception as e:
    print(f"\n程序运行时出错: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    print("\n开始保存数据...")
    
    # 保存情绪分析数据
    emotion_analyzer.save_final_analysis()
    
    # 保存系统响应数据
    try:
        save_all_data()
        print("系统响应数据保存完成")
    except Exception as e:
        print(f"保存系统响应数据时出错: {str(e)}")
    
    print("\n正在关闭设备...")
    if ser.is_open:
        ser.close()
        print("串口已关闭")
    if cap.isOpened():
        cap.release()
        print("摄像头已关闭")
    cv2.destroyAllWindows()
    print("所有窗口已关闭")
    
    print("\n=== 程序已安全退出 ===")
    
    # 强制退出
    import sys
    sys.exit(0)
