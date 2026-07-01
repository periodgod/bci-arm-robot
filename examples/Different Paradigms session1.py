import csv

import numpy as np
import pygame
import random
import time
import os
import math
from ctypes import windll

# 加载 inpoutx64.dll
P = windll.inpoutx64

def send_code(code):
    # 输出代码到 0x378 端口
    P.Out32(0x0378, code)
    # 等待 6 毫秒
    time.sleep(0.006)
    # 清除寄存器，发送 0
    P.Out32(0x0378, 0)
    # 等待 10 毫秒
    time.sleep(0.01)

# trial_num1 = 0
# numb1 = 0
# view_results = []
# auditory_results = []

# sub_numb = 1

def process_events():
    # 处理事件，如退出事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()


# SSVEP范式  5s
def ssvep_experiment(win,targets, frequencies,stim_duration=1, flash_duration=4, initial_phases=None):  #trigger_queue,
    """
    运行SSVEP刺激实验
    :param targets: 一个包含1-4之间整数的列表，每个整数表示一次实验的目标
    :param frequencies: 一个包含4个频率值的列表，定义每个方块的闪烁频率
    :param bg_color: 背景颜色，默认灰色 (90, 90, 90)
    :param stim_duration: 刺激显示持续时间，默认 1 秒
    :param flash_duration: 闪烁持续时间，默认 4 秒
    :param initial_phases: 一个包含4个初始相位值的列表，定义每个方块的初始相位
    """
    target_code_mapping = {
        1: 21,
        2: 22,
        3: 23,
        4: 24
    }
    black = (0, 0, 0)
    grey = (155, 155, 155)
    # 时间管理
    clock = pygame.time.Clock()
    if initial_phases is None:
        phases_ratio = [i * 0.35 % 2 for i in range(4)]  # 每个方块的相位比例 [0, 0.35, 0.7, 1.05]
        initial_phases = [2 * math.pi * p for p in phases_ratio]  # 映射到 [0, 2π)

    # 准备4个方块，居中显示
    squares = []
    cols = 2  # 列数
    rows = 2  # 行数
    square_size = 200
    padding_x = (win.get_width() - (cols * square_size)) // (cols + 1)  # 水平间距
    padding_y = (win.get_height() - (rows * square_size)) // (rows + 1)  # 垂直间距
    for i in range(4):  # 创建4个方块
        x = (i % cols) * square_size + padding_x * (i % cols + 1)
        y = (i // cols) * square_size + padding_y * (i // cols + 1)
        squares.append(pygame.Rect(x, y, square_size, square_size))  # 创建方块
## 1 2
## 3 4
    send_code(target_code_mapping.get(targets[0]))
    for target in targets:
        # print(f"Target: {target} -> Square: {squares[target - 1]}")  # 输出目标方块的信息
        # 显示目标刺激
        start_ticks = pygame.time.get_ticks()
        while (pygame.time.get_ticks() - start_ticks) < stim_duration * 1000:
            process_events()
            win.fill(grey)
            for i, square in enumerate(squares):
                if i == target - 1:
                    pygame.draw.rect(win, (255, 0, 0), square)  # 绘制红色目标方块
                else:
                    pygame.draw.rect(win, black, square)  # 绘制黑色方块
            pygame.display.flip()  # 刷新显示
            clock.tick(60)  # 控制帧率

        # 闪烁方块
        flash_start_ticks = pygame.time.get_ticks()
        while (pygame.time.get_ticks() - flash_start_ticks) < flash_duration * 1000:
            time_now = (pygame.time.get_ticks() - flash_start_ticks) / 1000  # 获取当前时间
            process_events()  # 处理事件
            win.fill(grey)  # 填充背景色
            for i, square in enumerate(squares):
                fs = frequencies[i]  # 获取频率
                fi = initial_phases[i]  # 获取初始相位
                t = time_now  # 当前时间
                intensity = (math.sin(2 * math.pi * fs * t + fi) + 1) / 2  # 计算亮度
                color_value = int(255 * intensity)  # 转换为颜色值
                pygame.draw.rect(win, (color_value, color_value, color_value), square)  # 绘制闪烁方块
            pygame.display.flip()  # 刷新显示
            clock.tick(60)  # 控制帧率
    send_code(200)


# 运动想象(MI)范式 5s
def mi_experiment(trial_list,win):  #trigger_queue,
    """
    运行运动想象实验
    :param trial_list: 包含1, 2或3的列表，1表示左手，2表示右手，3表示踮脚
    :param scale: 缩放比例
    """
    MI_code_mapping = {
        1: 11,
        2: 12,
        3: 13,
    }
    scale1 = 0.35
    grey = (155, 155, 155)
    def load_and_scale_image(image_path, scale):
        img = pygame.image.load(image_path).convert_alpha()
        original_width, original_height = img.get_size()
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        return pygame.transform.smoothscale(img, (new_width, new_height))

    def display_images(images, win, win_size):
        total_width = sum(img.get_width() for img in images)
        max_height = max(img.get_height() for img in images)
        x_offset = (win_size[0] - total_width) // 2
        y_offset = (win_size[1] - max_height) // 2

        for img in images:
            win.blit(img, (x_offset, y_offset))
            x_offset += img.get_width()
    send_code(MI_code_mapping.get(trial_list[0]))
    for trial in trial_list:
        # 想象阶段
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < 5:  # 5秒想象阶段
            process_events()  # 处理事件
            win.fill(grey)
            images = []
            if trial == 1:
                images.append(load_and_scale_image("1.png", scale1))
            elif trial == 2:
                images.append(load_and_scale_image("2.png", scale1))
            else:
                images.append(load_and_scale_image("33.png", 0.35))
            display_images(images, win, (win.get_width(),win.get_height()))
            pygame.display.flip()
    send_code(200)



# P300 范式 5.2s
# 设置字体
font_path = os.path.join("C:\\Windows\\Fonts", "msyh.ttc")
if not os.path.exists(font_path):
    font_path = pygame.font.get_default_font()

def p300_experiment(win, targets, highlight_orders, stim_duration=0.1, interval_duration=0.075):
    """
    P300 实验函数
    :param win: Pygame 窗口对象
    :param target: 目标字符
    :param highlight_orders: 高亮顺序列表，例如 ['613245', '265431', '132654', '423165']
    :param stim_duration: 刺激显示时间（秒）
    :param interval_duration: 刺激间隔时间（秒）
    :param rest_duration: 每轮之间的休息时间（秒）
    """
    black = (0, 0, 0)
    white = (255, 255, 255)
    # global trial_num1
    # global view_results
    # trial_num1 += 1
    P300_code_mapping = {
        'A': 31,
        'B': 32,
        'C': 33,
        'D': 34,
        'E': 35,
        'R': 36,
        '1': 37,
        '2': 38,
        '3': 39
    }

    font_large1 = pygame.font.Font(font_path, 132)

    letters = [
        ['A', 'B', 'C'],
        ['D', 'E', 'R'],
        ['1', '2', '3']
    ]

    positions = {}
    cell_width = win.get_width() // 3
    cell_height = win.get_height() // 3
    for i in range(3):
        for j in range(3):
            positions[letters[i][j]] = (j * cell_width + cell_width // 2, i * cell_height + cell_height // 2)

    def draw_letters(highlight=None, target=None, highlight_all=False):
        win.fill(black)
        for i in range(3):
            for j in range(3):
                letter = letters[i][j]
                pos = positions[letter]

                # 全局高亮显示所有字符
                if highlight_all:
                    text = font_large1.render(letter, True, white)
                    text_rect = text.get_rect(center=pos)
                    win.blit(text, text_rect)

                elif highlight and (highlight == (i, None) or highlight == (None, j)):
                    # 行或列的单独高亮
                    shadow_offset = 25
                    text_highlight = pygame.font.Font(None, 278).render(letter, True, white)
                    text_highlight_rect = text_highlight.get_rect(center=(pos[0], pos[1]))
                    shadow = pygame.font.Font(None, 278).render(letter, True, (50, 50, 50))
                    shadow_rect = shadow.get_rect(center=(pos[0] + shadow_offset, pos[1] + shadow_offset))
                    win.blit(shadow, shadow_rect)
                    win.blit(text_highlight, text_highlight_rect)
                else:
                    # 普通字符显示
                    if letter == target:
                        text = font_large1.render(letter, True, (255, 0, 0))  # 标红目标字符
                    else:
                        text = font_large1.render(letter, True, (60, 60, 60))
                    text_rect = text.get_rect(center=pos)
                    win.blit(text, text_rect)

        pygame.display.update()

    stimulus_start_time = time.perf_counter()
    #发送标签
    send_code(P300_code_mapping.get(targets[0]))
    for trial_num, target in enumerate(targets, start=1):
        start_ticks = time.perf_counter()
        # 在实验开始的前1秒内标红目标字符
        while (time.perf_counter() - start_ticks) < 1:  # 1秒内标红目标字符
            process_events()
            draw_letters(target=target)

        # 开始高亮阶段
        for round_num, current_order in enumerate(highlight_orders, start=1):  # 遍历每轮高亮顺序
            # print(f"Round {round_num}: Target = {target}, Order = {current_order}")
            # trial_highlights = []  # 用于存储每轮的高亮信息
            # trial_highlights.append(current_order)  # 保存每轮高亮的顺序
            for digit in current_order:
                num = int(digit)
                if 1 <= num <= 3:  # 列高亮
                    col = num - 1
                    start_ticks_1 = time.perf_counter()
                    while (time.perf_counter() - start_ticks_1) < stim_duration:
                        process_events()
                        draw_letters(highlight=(None, col), target=None)  # 目标字符不再标红
                    start_ticks_2 = time.perf_counter()
                    while (time.perf_counter() - start_ticks_2) < interval_duration:
                        process_events()
                        draw_letters(target=None)

                elif 4 <= num <= 6:  # 行高亮
                    row = num - 4
                    start_ticks_1 = time.perf_counter()
                    while (time.perf_counter() - start_ticks_1) < stim_duration:
                        process_events()
                        draw_letters(highlight=(row, None), target=None)  # 目标字符不再标红
                    start_ticks_2 = time.perf_counter()
                    while (time.perf_counter() - start_ticks_2) < interval_duration:
                        process_events()
                        draw_letters(target=None)

                # print(f"Highlighting: {num}")  # 打印当前高亮的行或列编号
    send_code(200)
    stimulus_end_time = time.perf_counter() - stimulus_start_time
    print(f"P300 Stimulus Total Time: {stimulus_end_time:.2f} seconds")
    #
    # view_results.append({
    #     "numb": trial_num1,
    #     "目标字符": targets,
    # })
    #


# 听觉范式 4.9s
def auditory_experiment(win, trials, target_channel_sequence):
    # 初始化一些全局参数
    # global numb1, auditory_results
    grey = (155, 155, 155)
    scale = 0.25
    pygame.mixer.init(frequency=44100, size=-16, channels=2)

    # # 增加实验计数
    # numb1 += 1

    # 加载音频文件
    target_sound_left = pygame.mixer.Sound('target_stereo_left.wav')
    target_sound_right = pygame.mixer.Sound('target_stereo_right.wav')
    non_target_sound_left = pygame.mixer.Sound('non_target_stereo_left.wav')
    non_target_sound_right = pygame.mixer.Sound('non_target_stereo_right.wav')
    silent_sound = pygame.mixer.Sound('silent_60ms.wav')
    silent_sound.set_volume(0)

    # 加载图片
    img = pygame.image.load("heilaba.png").convert_alpha()
    original_width, original_height = img.get_size()
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    img = pygame.transform.smoothscale(img, (new_width, new_height))

    # 准备阶段
    start = time.perf_counter()
    send_code(41)
    start_ticks = time.perf_counter()
    while (time.perf_counter() - start_ticks) < 1:
        process_events()
        win.fill(grey)
        win.blit(img, ((win.get_width() - new_width) // 2, (win.get_height() - new_height) // 2))
        pygame.display.flip()

    # # 目标声道信息存储
    # target_channel_info = []

    # 对于每一个trial的实验
    for trial_idx, trial in enumerate(trials):
        round_order = []  # 每个trial的顺序

        # 获取当前trial的目标音声道序列
        if isinstance(target_channel_sequence, list):  # 确保是列表类型
            trial_target_sequence = target_channel_sequence[trial_idx % len(target_channel_sequence)]  # 选取当前trial的目标声道序列
        else:
            print(f"Error: target_channel_sequence is not a list at trial {trial_idx + 1}")
            return

        # 打印调试信息
        # print(f"Trial {trial_idx + 1} - Target Sequence: {trial_target_sequence}")
        non_target_sequence = [0, 0, 1, 1]  # 0表示左声道，1表示右声道
        random.shuffle(non_target_sequence)  # 打乱非目标声道顺序
        non_target_index = 0

        # 遍历当前trial的每个循环
        for round_idx, round_sequence in enumerate(trial):
            stimuli = []
            # 遍历当前轮的序列
            for i, stim in enumerate(round_sequence):
                if stim == '1':  # 目标音
                    channel = trial_target_sequence[i]  # 选择目标音的声道（左或右）
                    sound = target_sound_left if channel == 'L' else target_sound_right
                    # target_channel_info.append(f"Trial {trial_idx + 1}, Round {round_idx + 1}, Target Sound: {'Left' if channel == 'L' else 'Right'}")
                else:  # 非目标音
                    channel = non_target_sequence[non_target_index]  # 从预生成的序列中读取声道
                    non_target_index += 1
                    sound = non_target_sound_left if channel == 0 else non_target_sound_right

                stimuli.append((sound, channel))  # 添加到刺激列表
                # print(f"Stimulus {i}: {stim}, Channel: {channel}, Sound: {sound}")

            # 播放刺激并记录顺序
            for sound, channel in stimuli:

                process_events()
                pygame.mixer.Channel(0).stop()
                pygame.mixer.Channel(1).stop()

                if channel == 0:
                    pygame.mixer.Channel(0).play(sound)
                    pygame.mixer.Channel(1).play(silent_sound)
                else:
                    pygame.mixer.Channel(1).play(sound)
                    pygame.mixer.Channel(0).play(silent_sound)

                round_order.append('1' if sound == target_sound_left or sound == target_sound_right else '2')

                # 精确计时
                sound_start_time = time.perf_counter()
                while (time.perf_counter() - sound_start_time) < 0.06:  # 60ms的音频时长
                    process_events()

                interval_start_time = time.perf_counter()
                while (time.perf_counter() - interval_start_time) < 0.135:  # 135ms间隔
                    process_events()

        # 保存当前trial的顺序
        # experiment_order.append("".join(round_order))

    # 结束信号
    send_code(200)
    print("auditory",time.perf_counter()-start)
    # # 保存实验结果
    # auditory_results.append({
    #     "numb": numb1,
    #     "目标音声道信息": target_channel_info,  # 保存目标音的声道信息
    # })


def wait_for_duration(duration):
    start_ticks = time.perf_counter()  # 获取开始时的 ticks
    while (time.perf_counter() - start_ticks) < duration:  # 持续轮询直到达到指定的时间
        process_events()  # 你自己的事件处理函数
        pygame.display.update()  # 保持屏幕更新以避免卡顿


def prepare_for_experiment(win, win_size, line_shrink, grey, black, target_sound_left, target_sound_right):
    send_code(51)
    # 显示准备界面
    win.fill(grey)  # 填充背景
        # 画水平线
    pygame.draw.line(win, black,
                         (win_size[0] // 2 - line_shrink, win_size[1] // 2),
                         (win_size[0] // 2 + line_shrink, win_size[1] // 2), 15)
        # 画垂直线
    pygame.draw.line(win, black,
                         (win_size[0] // 2, win_size[1] // 2 - line_shrink),
                         (win_size[0] // 2, win_size[1] // 2 + line_shrink), 15)
    pygame.display.update()
    wait_for_duration(60)
    send_code(200)
    # 播放第一次目标音
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    target_sound = random.choice([target_sound_left, target_sound_right])
    target_sound.play()
    # 等待 3 秒钟
    wait_for_duration(3)
    send_code(52)
    # 等待 1 分钟后播放第二次目标音
    win.fill(grey)  # 填充背景
        # 画水平线
    pygame.draw.line(win, black,
                         (win_size[0] // 2 - line_shrink, win_size[1] // 2),
                         (win_size[0] // 2 + line_shrink, win_size[1] // 2), 15)
        # 画垂直线
    pygame.draw.line(win, black,
                         (win_size[0] // 2, win_size[1] // 2 - line_shrink),
                         (win_size[0] // 2, win_size[1] // 2 + line_shrink), 15)
    pygame.display.update()
    wait_for_duration(60)
    send_code(200)
    target_sound.play()  # 播放第二次目标音
    # 等待 3 秒钟
    wait_for_duration(3)

# 主实验函数
def main_experiment():  #多线程参数trigger_queue,stop_event
    # 设置线条的缩短量
    # global view_results
    # global auditory_results
    pygame.init()
    win_size = (pygame.display.Info().current_w, pygame.display.Info().current_h)
    win = pygame.display.set_mode(win_size, pygame.FULLSCREEN)
    pygame.display.set_caption("实验")  # 设置窗口标题



    font_path = os.path.join("C:\\Windows\\Fonts", "msyh.ttc")  # 字体路径
    # 设置字体
    if not os.path.exists(font_path):  # 检查字体文件是否存在
        font_path = pygame.font.get_default_font()  # 使用默认字体
    font_large = pygame.font.Font(font_path, 74)  # 设置大字体
    line_shrink = 50  # 缩短的像素值

    black = (0, 0, 0)
    grey = (155, 155, 155)
    trial_order = ['MI 左手'] * 30 + ['MI 右手'] * 30 + ['MI 踮脚'] * 30 + ['SSVEP'] * 40 + ['P300'] * 36 + ['auditory'] * 30

    # 设置P300顺序 列为1，2，3 行为4，5，6
    p300_targets_and_orders = [
        ('B', ['435261', '435612', '132654', '546123']),  # 目标字符 B 的列行为: (2, 4)
        ('1', ['213456', '265431', '156243', '231456']),  # 目标字符 1 的列行为: (1, 6)
        ('E', ['154623', '456123', '465321', '432156']),  # 目标字符 E 的列行为: (2, 5)
        ('3', ['435162', '543261', '154623', '543126']),  # 目标字符 3 的列行为: (3, 6)
        ('A', ['132645', '135426', '462135', '543261']),  # 目标字符 A 的列行为: (1, 4)
        ('C', ['246531', '132654', '453261', '642315']),  # 目标字符 C 的列行为: (3, 4)
        ('R', ['156243', '456123', '132645', '243165']),  # 目标字符 R 的列行为: (3, 5)
        ('1', ['156243', '134265', '265431', '356421']),  # 目标字符 1 的列行为: (1, 6)
        ('D', ['546123', '164532', '543261', '132654']),  # 目标字符 D 的列行为: (1, 5)
        ('R', ['432615', '132645', '654132', '321654']),  # 目标字符 R 的列行为: (3, 5)
        ('C', ['132645', '642135', '321654', '465321']),  # 目标字符 C 的列行为: (3, 4)
        ('D', ['312465', '213456', '612453', '456123']),  # 目标字符 D 的列行为: (1, 5)
        ('3', ['342165', '265431', '432156', '351264']),  # 目标字符 3 的列行为: (3, 6)
        ('E', ['342156', '132645', '645132', '624135']),  # 目标字符 E 的列行为: (2, 5)
        ('3', ['435126', '132645', '462135', '213456']),  # 目标字符 3 的列行为: (3, 6)
        ('D', ['312465', '465321', '312645', '635241']),  # 目标字符 D 的列行为: (1, 5)
        ('C', ['612453', '642135', '153264', '135642']),  # 目标字符 C 的列行为: (3, 4)
        ('2', ['564312', '324561', '432156', '423516']),  # 目标字符 2 的列行为: (2, 6)
        ('A', ['612453', '132654', '432651', '213456']),  # 目标字符 A 的列行为: (1, 4)
        ('E', ['164532', '453126', '312465', '432156']),  # 目标字符 E 的列行为: (2, 5)
        ('1', ['341256', '432651', '426513', '543126']),  # 目标字符 1 的列行为: (1, 6)
        ('A', ['312546', '321654', '264531', '653124']),  # 目标字符 A 的列行为: (1, 4)
        ('R', ['321654', '243165', '465123', '423165']),  # 目标字符 R 的列行为: (3, 5)
        ('R', ['154623', '132645', '234651', '213465']),  # 目标字符 R 的列行为: (3, 5)
        ('2', ['321465', '423165', '243516', '213456']),  # 目标字符 2 的列行为: (2, 6)
        ('B', ['431256', '436215', '512346', '215634']),  # 目标字符 B 的列行为: (2, 4)
        ('A', ['312645', '432156', '135426', '265431']),  # 目标字符 A 的列行为: (1, 4)
        ('2', ['653124', '321564', '432156', '546123']),  # 目标字符 2 的列行为: (2, 6)
        ('B', ['465321', '235461', '321654', '435612']),  # 目标字符 B 的列行为: (2, 4)
        ('C', ['462135', '456123', '623154', '164523']),  # 目标字符 C 的列行为: (3, 4)
        ('B', ['435612', '543612', '532614', '312654']),  # 目标字符 B 的列行为: (2, 4)
        ('E', ['321654', '321654', '162345', '351624']),  # 目标字符 E 的列行为: (2, 5)
        ('1', ['415362', '564312', '265431', '251436']),  # 目标字符 1 的列行为: (1, 6)
        ('D', ['456213', '243165', '341265', '235461']),  # 目标字符 D 的列行为: (1, 5)
        ('2', ['523461', '512346', '213465', '436512']),  # 目标字符 2 的列行为: (2, 6)
        ('3', ['435162', '531264', '564132', '264531']),  # 目标字符 3 的列行为: (3, 6)
    ]
    # 建立字符到label值的映射字典
    label_mapping = {
        'A': 31,
        'B': 32,
        'C': 33,
        'D': 34,
        'E': 35,
        'R': 36,
        '1': 37,
        '2': 38,
        '3': 39
    }
    SSVEP_label_mapping = {
        1: 21,
        2: 22,
        3: 23,
        4: 24,
    }
    #听觉范式1代表目标音，2代表非目标音
    p300_auditory = [
        ['22212', '22122', '22212', '22221'],
        ['22122', '22212', '22122', '21222'],
        ['21222', '12222', '22221', '22122'],
        ['22122', '21222', '12222', '22122'],
        ['12222', '22212', '21222', '22212'],
        ['21222', '12222', '22122', '21222'],
        ['22221', '22122', '21222', '22122'],
        ['12222', '22221', '22122', '22221'],
        ['22221', '22212', '22122', '22212'],
        ['22122', '21222', '22122', '21222'],
        ['22122', '21222', '12222', '21222'],
        ['12222', '22122', '12222', '21222'],
        ['22122', '22212', '22221', '22122'],
        ['12222', '22122', '12222', '21222'],
        ['22212', '22221', '22212', '22122'],
        ['21222', '12222', '22212', '22221'],
        ['12222', '22221', '22122', '22212'],
        ['22122', '12222', '22122', '12222'],
        ['22212', '22221', '22122', '22221'],
        ['22221', '22212', '21222', '22122'],
        ['22122', '22212', '21222', '22122'],
        ['12222', '21222', '22122', '22212'],
        ['22212', '22122', '22221', '12222'],
        ['12222', '22221', '22122', '21222'],
        ['12222', '22221', '22122', '22221'],
        ['12222', '21222', '22122', '22221'],
        ['22122', '12222', '21222', '22212'],
        ['22122', '22212', '22221', '22212'],
        ['22221', '22122', '22221', '22122'],
        ['22122', '12222', '22212', '22221'],
    ]
    #控制目标音通道
    target_channel_sequence = [
        ['R', 'L', 'L', 'R'],
        ['R', 'L', 'L', 'L'],
        ['L', 'R', 'R', 'R'],
        ['R', 'R', 'L', 'L'],
        ['R', 'L', 'L', 'R'],
        ['L', 'R', 'R', 'R'],
        ['L', 'L', 'L', 'L'],
        ['L', 'L', 'R', 'R'],
        ['L', 'L', 'L', 'L'],
        ['R', 'R', 'L', 'R'],
        ['L', 'L', 'L', 'R'],
        ['R', 'R', 'L', 'R'],
        ['R', 'R', 'R', 'L'],
        ['R', 'L', 'R', 'L'],
        ['L', 'L', 'R', 'L'],
        ['R', 'R', 'L', 'L'],
        ['L', 'R', 'R', 'R'],
        ['L', 'L', 'L', 'L'],
        ['R', 'R', 'R', 'R'],
        ['L', 'L', 'R', 'R'],
        ['L', 'R', 'L', 'L'],
        ['R', 'R', 'L', 'L'],
        ['R', 'L', 'L', 'L'],
        ['R', 'L', 'R', 'R'],
        ['R', 'L', 'R', 'L'],
        ['L', 'L', 'L', 'R'],
        ['R', 'R', 'R', 'L'],
        ['L', 'R', 'R', 'L'],
        ['L', 'R', 'L', 'R'],
        ['R', 'R', 'R', 'R'],
    ]
    SSVEP_Order = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
                   [2],[2],[2],[2],[2],[2],[2],[2],[2],[2],
                   [3],[3],[3],[3],[3],[3],[3],[3],[3],[3],
                   [4],[4],[4],[4],[4],[4],[4],[4],[4],[4],
                   ]

    random.shuffle(SSVEP_Order)
    # 设置P300顺序
    random.shuffle(trial_order)  # 随机打乱试验顺序
    # 生成均匀分布的准备时间并四舍五入到小数点后两位
    prepare_times = np.round(np.linspace(1.5, 3.5, num=166), 1)
    random.shuffle(prepare_times)

    # 记录 trial_order, prepare_times 和 labels
    # results = []
    # 显示准备界面和播放音频
    # 音频文件路径
    target_sound_left = pygame.mixer.Sound('target_stereo_left.wav')
    target_sound_right = pygame.mixer.Sound('target_stereo_right.wav')
    prepare_for_experiment(win, win_size, line_shrink, grey, black, target_sound_left, target_sound_right)
    # 时间5s wait
    start_task = time.perf_counter()
    send_code(99)
    for trial, prepare_time in zip(trial_order, prepare_times):
        # finish_event.set()  # 设置事件，通知其他进程启动 while 循环
        win.fill(grey)  # 填充白色背景
        # 画水平线
        pygame.draw.line(win, black,
                         (win_size[0] // 2 - line_shrink, win_size[1] // 2),
                         (win_size[0] // 2 + line_shrink, win_size[1] // 2), 15)
        # 画垂直线
        pygame.draw.line(win, black,
                         (win_size[0] // 2, win_size[1] // 2 - line_shrink),
                         (win_size[0] // 2, win_size[1] // 2 + line_shrink), 15)
        pygame.display.update()
        wait_for_duration(prepare_time)
        print(prepare_time)

        # 根据任务类型和目标设置标签
        if trial == 'MI 左手':
            # label = 11
            # task_duration = task_durations['MI']
            mi_experiment([1],win)  # 左手运行运动想象实验trigger_queue,
        elif trial == 'MI 右手':
            # label = 12
            mi_experiment([2],win)
        elif trial == 'MI 踮脚':
            # label = 13
            mi_experiment([3],win)
        elif trial == 'auditory':
            trials = p300_auditory.pop(0)
            target_channel = target_channel_sequence.pop(0)
            # label = 41
            auditory_experiment(win,trials=trials,target_channel_sequence=target_channel)
        elif trial == 'SSVEP':
            target = SSVEP_Order.pop(0)
            # label = SSVEP_label_mapping.get(target[0])
            ssvep_experiment(win,target, [12, 13, 16, 17])  # 运行SSVEP实验
        elif trial == 'P300':
            # 按顺序取出目标和高亮顺序
            target, highlight_order = p300_targets_and_orders.pop(0)
            # label = label_mapping.get(target)
            p300_experiment(win, [target], highlight_order)  # 按顺序传入目标和高亮顺序
        # results.append((trial, label))
    print(time.perf_counter()-start_task)
    win.fill(grey)
    end_text = font_large.render('实验结束', True, black)
    end_rect = end_text.get_rect(center=(win_size[0] // 2, win_size[1] // 2))
    win.blit(end_text, end_rect)
    pygame.display.update()
    time.sleep(5)
    send_code(100)
    # # 指定保存路径
    # folder_name = "D:/czc labels_Different Paradigms Session1"
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)  # 如果文件夹不存在则创建
    #
    # # 保存 results 到 CSV 文件 (Trial 和 Label)
    # with open(os.path.join(folder_name, f'Experiment_Results_Sub_{sub_numb}.csv'), mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Trial', 'Label'])  # 写入表头
    #     for row in results:
    #         writer.writerow(row)  # 写入数据

    # 保存 auditory_results 到 CSV 文件
    # with open(os.path.join(folder_name, 'auditory_experiment_results.csv'), mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['numb', '目标音声道信息'])  # 表头, '第二轮顺序', '第三轮顺序', '第四轮顺序'
    #
    #     for row in auditory_results:
    #         # 直接保存每一轮的顺序字符串
    #         # first_round = row['第一轮顺序']
    #         # second_round = row['第二轮顺序']
    #         # third_round = row['第三轮顺序']
    #         # fourth_round = row['第四轮顺序']
    #         frit = row['目标音声道信息']
    #         # 保存为一行
    #         writer.writerow([row['numb'],frit])# first_round, second_round, third_round, fourth_round,
    # 保存 view_results 到 CSV 文件
    # with open(os.path.join(folder_name, 'P300_experiment_results.csv'), mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['numb', '目标字符'])  # 表头
    #     for row in view_results:
    #         # 将每一轮顺序列表转换为字符串
    #         writer.writerow([row['numb'], str(row['目标字符'])])


"============================================================================="
"============================================================================="
"============================================================================="
"============================================================================="
"============================================================================="
"============================================================================="
"============================================================================="
"============================================================================="
#
# class Marker:
#     def __init__(self):
#         self.position = 0
#         self.points = 0
#         self.channel = -1
#         self.type = ""
#         self.description = ""
#
#
# # Helper function for receiving whole message
# def RecvData(socket, requestedSize):
#     returnStream = b''
#     while len(returnStream) < requestedSize:
#         databytes = socket.recv(requestedSize - len(returnStream))
#         if not databytes:
#             raise RuntimeError("Connection broken")
#         returnStream += databytes
#
#     return returnStream
#
#
# # Helper function for splitting a raw array of
# # zero-terminated strings (C) into an array of Python strings
# def SplitString(raw):
#     stringlist = []
#     s = b""
#     for i in range(len(raw)):
#         if raw[i] != 0:
#             s += bytes([raw[i]])
#         else:
#             stringlist.append(s.decode("utf-8"))
#             s = b""
#
#     return stringlist
#
#
# # Helper function for extracting EEG properties from a raw data array
# # read from TCP/IP socket
# def GetProperties(rawdata):
#     # Extract numerical data
#     (channelCount, samplingInterval) = unpack('<Ld', rawdata[:12])
#
#     # Extract resolutions
#     resolutions = []
#     for c in range(channelCount):
#         index = 12 + c * 8
#         restuple = unpack('<d', rawdata[index:index + 8])
#         resolutions.append(restuple[0])
#
#     # Extract channel names
#     channelNames = SplitString(rawdata[12 + 8 * channelCount:])
#
#     return (channelCount, samplingInterval, resolutions, channelNames)
#
#
# # Helper function for extracting EEG and marker data from a raw data array
# # read from TCP/IP socket
# def GetData(rawdata, channelCount):
#     # Extract numerical data
#     (block, points, markerCount) = unpack('<LLL', rawdata[:12])
#
#     # Extract EEG data as an array of floats
#     data = []
#     for i in range(points * channelCount):
#         index = 12 + 4 * i
#         value = unpack('<f', rawdata[index:index + 4])
#         data.append(value[0])
#
#     # Extract markers
#     markers = []
#     index = 12 + 4 * points * channelCount
#     for m in range(markerCount):
#         markersize = unpack('<L', rawdata[index:index + 4])
#
#         ma = Marker()
#         (ma.position, ma.points, ma.channel) = unpack('<LLl', rawdata[index + 4:index + 16])
#         typedesc = SplitString(rawdata[index + 16:index + markersize[0]])
#         ma.type = typedesc[0]
#         ma.description = typedesc[1]
#
#         markers.append(ma)
#         index = index + markersize[0]
#
#     return (block, points, markerCount, data, markers)
#
#
# # Placeholder for RDA module
# class RDA:
#     def __init__(self):
#         # self.eeg_data = []
#
#         # Main RDA routine
#         # Create a TCP/IP socket
#         self.con = socket(AF_INET, SOCK_STREAM)
#         # Connect to the recorder host via the 32-bit RDA port
#         # Adapt to your host; if the recorder is not running on the local machine,
#         # change the port to 51234 to connect to the 16-bit RDA port
#         self.con.connect(("localhost", 51244))
#
#         # Block counter to check overflows of the TCP/IP buffer
#         self.lastBlock = -1
#
#     def read_data(self):
#
#         # Get message header as a raw array of chars
#         rawhdr = RecvData(self.con, 24)
#
#         # Split array into useful information; id1 to id4 are constants
#         (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)
#
#         # Get the data part of the message, which is of variable size
#         rawdata = RecvData(self.con, msgsize - 24)
#
#         # Perform an action depending on the message type
#         if msgtype == 1:
#             # Start message, extract EEG properties and display them
#             (self.channelCount, samplingInterval, resolutions, channelNames) = GetProperties(rawdata)
#             # Reset the block counter
#             self.lastBlock = -1
#
#             # print("Start")
#             # print("Number of channels: " + str(self.channelCount))
#             # print("Sampling interval: " + str(samplingInterval))
#             # print("Resolutions: " + str(resolutions))
#             # print("Channel Names: " + str(channelNames))
#
#             info = {'Number of channels': str(self.channelCount),
#                     'Sampling interval': str(samplingInterval),
#                     'Resolutions': str(resolutions),
#                     'Channel Names': str(channelNames),
#                     }
#
#             return 'info', info
#
#         elif msgtype == 4:
#             # Data message, extract data and markers
#             (block, points, markerCount, packet_data, markers) = GetData(rawdata, self.channelCount)
#
#             # Check for overflow
#             if self.lastBlock != -1 and block > self.lastBlock + 1:
#                 print("*** Overflow with " + str(block - self.lastBlock) + " data blocks ***")
#             self.lastBlock = block
#
#             # Print markers if there are some in the actual block
#             # if markerCount > 0:
#             #     for m in range(markerCount):
#             #         print("Marker " + markers[m].description + " of type " + markers[m].type)
#
#             '''
#             (通道X 5)
#             data packet，
#
#             50个 packet_data  是 1s的data
#             采样频率 ： 250Hz
#             '''
#             packet_data = np.array(packet_data, dtype=np.int32)  # 采集的数据是 int类型
#             # packet_data = packet_data.reshape(self.channelCount,-1)
#
#             packet_data = packet_data.reshape(-1, self.channelCount).T
#
#             # print(packet_data.shape)
#             # Put data at the end of the actual buffer
#             return 'data', packet_data
#
#
#         elif msgtype == 3:
#             # Stop message, terminate the program
#             print("Stop")
#             finish = True
#
#         else:
#             return ['unkown', 'unkown']
#
#
# def rda_reader(eeg_queue, trigger_queue, stop_event):
#     """Function to read data and place it into the queue."""
#     rda = RDA()  # Initialize RDA object (assumed to be provided)
#     while not stop_event.is_set():
#         form, packet_data = rda.read_data()  # Read packet data
#         if form == 'info':
#             print('info')
#         elif form == 'data':
#             packet_trigger = np.zeros((1, packet_data.shape[-1]), np.int32)
#             if not trigger_queue.empty():  # put trigger # 这个频率高， trigger_queue频率低
#                 # packet_data.shape (n_channels, 5), 250Hz, 50个packet_data为1s的数据
#                 trigger = trigger_queue.get()
#                 packet_trigger[0, 0] = trigger
#                 # print('RDA trigger:{}'.format(trigger))
#             else:
#                 packet_trigger[0, :] = 0
#
#             data = np.concatenate([packet_data, packet_trigger], axis=0)
#             eeg_queue.put(data)
#
# def process_data(eeg_queue, stop_event ):
#     """Function to process and save data from the queue."""
#     # 初始化一个形状为(32, 5)的二维数组，所有元素都是0
#     data = None
#     i = 0
#     fs = 250  # Hz
#     interval = 5  # s 每隔多少s就保存一次
#     folder_name = "D:/czc"
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)  # 如果文件夹不存在则创建
#     data_root = 'D:\\czc'
#     start_time = datetime.now()
#     formatted_time = start_time.strftime('%Y%m%d_%H%M%S')
#     file_name = formatted_time + '_' + 'confusion.npy'
#     while not stop_event.is_set():
#         if not eeg_queue.empty():
#             packet_data = eeg_queue.get()  # Get data from the queue
#             # If data is None, initialize it with the first packet
#             if data is None:
#                 data = packet_data
#                 continue
#             # Concatenate new data to the existing data
#             data = np.concatenate((data, packet_data), axis=-1)  # (32, times)
#
#             # Save data every specified interval
#             if data.shape[-1] / fs % interval == 0:  # Every interval seconds
#                 i += 1
#                 np.save(os.path.join(data_root, file_name), data)
#                 saved_length = data.shape[-1] / fs  # Data length in seconds
#                 print(f"保存长度: 第{i}次, {saved_length}")

# Ensure that the experiment is only started when this script is run directly
if __name__ == '__main__':
    # Thread
    # main_experiment()
    # trigger_queue = mp.Queue()
    # eeg_queue = mp.Queue()
    # stop_event = mp.Event()  # 全局事件用于停止进程
    # #
    # reader_process = mp.Process(target=rda_reader, args=(eeg_queue, trigger_queue, stop_event))
    # save_data = mp.Process(target=process_data, args=(eeg_queue, stop_event))
    # main_process = mp.Process(target=main_experiment,args=(trigger_queue, stop_event))
    #
    # #
    # reader_process.start()
    # save_data.start()
    # main_process.start()
    #
    #
    # main_process.join()
    # reader_process.join()
    # save_data.join()
    #
    # pygame.quit()
    # print("退出")
    main_experiment()
    pygame.quit()
    print("退出")
# 一个显示，一个读取，一个写入
