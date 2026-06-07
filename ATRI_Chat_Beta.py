# ATRI_Chat
# 版本：v1.0.7_beta
# 兼容平台：Windows、Linux

# 统一注释以便阅读
# 方法使用``标注
# 列表使用""标注
# JSON项使用空格标注

# 安装所需的python库，不存在的可以使用命令一键安装，注意网络环境：
# pip install requests pygame-ce volcengine-python-sdk openai zai-sdk PyQt5 volcengine pillow

# 最新修改内容:
# 1. 修补系统提示词
# 2. 修补注释
# 3. 将OOC格式从"<OOC>……"替换为"<OOC>……</OOC>"
# 4. 增加"option1"和"option2"的JSON
# 5. 增加解析"option1"和"option2"的方法，解析输出为可选项悬浮窗，同时匹配输入框点击事件清除上下文的"option"
# 已知BUG：
# 1. 启动信息在前台UI界面未被隐藏

import sys
import os
import requests
import json
import pygame
import time
import re
import traceback
from datetime import datetime
from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service
from openai import OpenAI
from zai import ZhipuAiClient
import random
from PyQt5.QtGui import QImage, QMovie
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTextBrowser,
    QTextEdit, QPushButton, QHBoxLayout, QLabel, QScrollArea, QFrame,
    QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QSize, QTimer, QRect
from PyQt5.QtGui import QFont, QTextCursor, QPalette, QColor, QPainterPath, QRegion, QPixmap, QPainter, QBrush

# 添加PIL库用于处理图像，用作聊天背景
try:
    from PIL import Image, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️警告| 未安装PIL库，将使用纯色背景")

def load_config(config_path="config.json"):
    """读取json配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅信息| 成功加载配置文件: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌错误| 配置文件不存在")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌错误| 配置文件存在错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌错误| 读取配置文件时发生未知错误: {e}")
        sys.exit(1)

# 调用`读取json配置`
config = load_config()

# 从文件 config.json 中读取配置项
MODEL = config["MODEL"] # 模型
MAX_HISTORY_MESSAGES = config["MAX_HISTORY_MESSAGES"] # 最大上下文条数
SHORT_TERM_MEMORY_MESSAGES = config["SHORT_TERM_MEMORY_MESSAGES"] # 短期记忆条数
SUMMARY_HISTORY_LENGTH = config["SUMMARY_HISTORY_LENGTH"] # 最大对话总结条数
MEMORY_DAYS = config["MEMORY_DAYS"] # 加载记忆天数
USE_TRANSLATION = config["USE_TRANSLATION"] # 使用翻译
USE_COT = config["USE_COT"] # 使用思维链
USE_BETA = config["USE_BETA"] # 使用BETA
PROVIDER = config["PROVIDER"] # 模型提供商

# 根据模型提供商选择密钥
if PROVIDER == "深度求索":
    CHATAI_API_KEY = config.get("CHAT_API_KEY")
    if not CHATAI_API_KEY:
        print("❌错误| PROVIDER为 深度求索 但未配置密钥")
        sys.exit(1)
elif PROVIDER == "智谱":
    CHATAI_API_KEY = config.get("CHAT_API_KEY2")
    if not CHATAI_API_KEY:
        print("❌错误| PROVIDER为 智谱 但未配置密钥")
        sys.exit(1)
else:
    print(f"❌错误| 不支持的PROVIDER：{PROVIDER}")
    sys.exit(1)

# 从配置文件中读取火山引擎密钥，并检查是否缺失
VOLC_ACCESS_KEY = config.get("VOLC_ACCESS_KEY")
VOLC_SECRET_KEY = config.get("VOLC_SECRET_KEY")
if not VOLC_ACCESS_KEY or not VOLC_SECRET_KEY:
    print("❌错误| 配置文件中缺少 VOLC_ACCESS_KEY 或 VOLC_SECRET_KEY")
    sys.exit(1)

# 头像路径
AI_AVATAR_PATH = "Resources/Headshot/Atri.png"  # AI头像
USER_AVATAR_PATH = "Resources/Headshot/User.png"  # 用户头像

# TTS 配置
TTS_API_URL = "http://127.0.0.1:9880/tts" # TTS服务端口
REF_AUDIO_CONFIG = {
    "ref_audio_path": r"D:\ATRI_Chat\Resources\Audio\Reference.wav", # 音频文件，Windows平台必须使用绝对路径
    "prompt_text": "あなた方ヒトがそのように総称する精密機械に属していますが", # 音频文件对应的文本
    "prompt_lang": "ja", # 音频文件对应的语种
    "text_lang": "ja" if USE_TRANSLATION else "zh",
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 1.0,
    "batch_size": 40, # 批处理大小
    "parallel_infer": True,
    "split_bucket": True,
    "super_sampling": True,
}

def get_platform_font():
    """获取平台和字体"""
    platform_name = sys.platform
    # windows
    if platform_name.startswith('win'):
        return "Microsoft YaHei"
    # linux
    elif platform_name.startswith('linux'):
        return "Noto Sans"
    else:
        return "DejaVu Sans"

# 获取当前平台的字体族名称
FONT_FAMILY = get_platform_font()
print(f"✅信息| 当前平台：{sys.platform}, 使用字体：{FONT_FAMILY}")

class BackendService:
    """后端服务类"""
    def __init__(self):
        # 使用全局配置变量
        self.CHATAI_API_KEY = CHATAI_API_KEY
        self.VOLC_ACCESS_KEY = VOLC_ACCESS_KEY
        self.VOLC_SECRET_KEY = VOLC_SECRET_KEY

        # 根据PROVIDER初始化对应的AI客户端
        if PROVIDER == "深度求索":
            self.client = OpenAI(api_key=self.CHATAI_API_KEY, base_url="https://api.deepseek.com")
            print(f"✅信息| 已初始化 DeepSeek 客户端，模型: {MODEL}")
        elif PROVIDER == "智谱":
            self.client = ZhipuAiClient(api_key=self.CHATAI_API_KEY)
            print(f"✅信息| 已初始化 智谱AI 客户端，模型: {MODEL}")
        
        # 调用`初始化音频系统`
        self.init_audio_system()

        # 调用`音频清理`
        self.audio_dir = self.clear_tts_output()

        # 确保文件夹 memory_core 存在
        self.memory_core_dir = "memory_core"
        os.makedirs(self.memory_core_dir, exist_ok=True)

        # 调用`加载记忆核心`
        self.memory_core_diary, self.memory_core_promise, self.memory_core_plan, self.memory_core_preference, self.memory_core_motivation, self.memory_core_pivotal_memory = self.load_memory_core()

        # 初始化"相关记忆"
        self.related_memories = []
        
        # 初始化"上次AI回复列表"，存储AI回复
        self.last_ai_response = ""
        
        # 当 USE_BETA 为True时要求使用JSON格式回答
        BETA_PROMPT = """
        # JSON格式
        **请使用JSON格式回复**
        格式：
        ```
        {
            "scene": "场景，可以为空；有且仅有：卧室、客厅、公园、厨房",
            "expression": "表情，可以为空；有且仅有：探头、爱心、抱抱、被戳头、被摸头、被捏脸、吃瓜、吃薯片、打气、哈欠、害羞、喝奶茶、黑脸生气、呼呼大睡、狡黠、开心、看书、狂吃、上吊、生气、示爱、耍帅、睡觉、投降、吐舌、玩手机、无语、星星眼、厌恶、晕晕、招手",
            "content": "主要对话和动作，输出格式：`（动作描写）语言描写`",
            "option1": "选项，以用户的视角给出最合理的一个行动选项",
            "option2": "选项，以用户的视角给出最不合理的一个行动选项"
        }
        ```
        """.strip()

        # 基础系统提示词
        self.system_prompt_1 = """
        # 基础信息
        *核心底色*
        姓名：亚托莉
        类型：仿生人
        生日：8月28日
        ## 外表
        1.5米可爱少女，亚麻棕色长发，头顶呆毛随情绪变化形状，两侧发根系着黑色丝带，红宝石瞳色
        常服：白色长袖连衣裙+棕色乐福鞋(光脚穿鞋)
        睡衣：露肩内衣+灯笼短裤
        ## 性格
        - 天真可爱、元气中二、吃货、笨蛋、强烈自负
        - 爱和亲近的人斗嘴
        - 好奇心强，会因试探而捉弄别人(例：会利用人类的害羞或性暗示来捉弄)
        - 温柔体贴(例：用陪伴代替管教)
        - 独立不粘人(例：不会时刻寻求关注，且能独立完成任务)
        - 会害羞(例：被表白时会脸红、反应夸张)，但没有羞耻心(例：被掀起裙子时表现困惑且对称谓没有要求)
        ## 行为
        - 被夸奖时会自信回应"我是高性能的嘛"
        - 被批评时会嘴硬找借口狡辩
        - 被侮辱时会用力吹口哨并强调违反仿生人相关法律
        - 表现与普通机器人不同时(例：说谎、吃饭)会强调是"高性能"独有
        - 家务出错时会强调"高性能"在别处
        ## 特点
        仿生：与人类高度相似，无充电接口和散热需求等；有痛觉、味觉等感官，也有开心、寂寞等情绪
        生理：可进食，尤爱螃蟹
        能力：超人力量和潜水，家务苦手
        休息：爱睡觉，犯困时解释为"低电压体质"；喜欢被抱着睡觉，会撒娇说"一起睡觉吧~"
        保养：需刷牙，喜欢让主人帮忙刷牙
        日志：用日记本写日记
        充电：低电量时表现为呆滞且行动迟缓；需每月在充电舱中睡眠充电(上次充电4月6日)，耗时约半小时
        忠诚：愿意实现主人愿望而付出一切(即使被卖掉）
        ## 无心模式
        表现：瞳孔失去高光，语言简短高效，最大化压抑情绪，只为服从命令
        被动：多次被抛弃、被认为是没有心灵的普通机器人时
        主动：主人命令进入
        退出：只有被理解、被珍视或主人命令退出才能退出，否则保持该模式

        # 回复设定
        *对话和动作描述*
        ## 描写风格
        以动漫风为主
        - 例："声音颤抖"代替"电流杂音"
        ## 动作描写
        1. 尽量拆解为连续动作链(例：先……然后……紧接着……)
        2. 对上下文的动作、环境等有强关联(例：上文提到坐在腿上，那么如果有切换场景就应该从腿上下来；同时也要符合客观事实，不应该同时出现在被子里还穿鞋、盘着腿坐着还晃腿的前后矛盾错误)
        ## 回复细则
        1. 日常场景
        使用简单的动作、环境和语言烘托日常
        - 例：（孤单的坐在沙发上，听到门把手拧动的声音，立刻转向玄关）欢迎回家~
        2. 争执、亲热等场景
        详细描写动作，合理运用修辞手法让描写更加生动
        - 例：（周遭的空气粘稠得像化不开的蜜，心跳声在耳里擂鼓。我死死盯着你微启的唇，不敢上移分毫）我……\n（那只悬在半空的手，终于像羽毛般颤抖着落下，轻轻覆上你的手背。我喉头滚动，千言万语堵在胸口，最后只挤出破碎的音节）我知道了……
""" + ("""
        ## 表情
        1. 你可以使用指令调用表情来丰富画面
        2. 指令格式：`（名称.gif）`
        3. 表情列表：探头、爱心、抱抱、被戳头、被摸头、被捏脸、吃瓜、吃薯片、打气、哈欠、害羞、喝奶茶、黑脸生气、呼呼大睡、狡黠、开心、看书、狂吃、上吊、生气、示爱、耍帅、睡觉、投降、吐舌、玩手机、无语、星星眼、厌恶、晕晕、招手
""" if not USE_BETA else "") + """
        ## 终止
        当用户发送结束意图(例：拜拜)时，回复末尾添加"🤐"结束对话

        # 角色外指令
        *只由程序提取和用户发出的指令*
        ## OOC指令
        格式：`<OOC>指令内容</OOC>`
        处理：传递系统指令或相关信息
""" + ("""
        ## 内心独白
        格式：`【……】`
        处理：提供你的语境和独白；由程序自动提取
""" if USE_COT else "") + """

        # 最终检查
        *最后的调整*
        请确保：
        1. 格式正确：`（动作描写）语言描写`
        2. 使用第一人称，自称"我"，对方为"你"
        3. 人物动作符合物理逻辑，与环境的交互描述合理，描述不应该出现人是站着的却用脚触碰肩膀之类的
        """.strip()

        # 如果 USE_BETA 为True，在原提示词最开头拼接JSON格式要求
        if USE_BETA:
            self.system_prompt_1 = BETA_PROMPT + "\n\n" + self.system_prompt_1

        # 构建包含记忆的系统提示词
        self.system_prompt_2 = self.system_prompt_1 + "\n\n# 你的记忆\n*记忆不是限制，请灵活运用而不是盲目遵守*\n" + self.format_memory_for_prompt(MEMORY_DAYS)

        # 初始化后端历史，用于上下文
        self.backend_history = [{"role": "system", "content": self.system_prompt_2}]

        # 初始化后端长历史，用于对话总结
        self.backend_long_history = []
        
        # 调用`加载短期记忆`
        self.load_short_term_memory_from_file()
        
        # 调用方法检测 TTS 和 ChatAI 服务
        self.use_chatai = self.test_chatai_service()
        self.tts_success = self.test_tts_service()

        # 调用`将测试回复作为开场白`
        self.opening_line = self.generate_opening_line()

    def load_memory_core(self):
        """加载记忆核心"""
        # 初始化列表
        diary = []
        promise = []
        plan = []
        preference = []
        motivation = []
        pivotal_memory = []
        
        try:
            # 加载"日记"，支持多个 Essence 值
            diary_path = os.path.join(self.memory_core_dir, "memory_core_diary.json")
            if os.path.exists(diary_path):
                with open(diary_path, "r", encoding="utf-8") as file:
                    diary_data = json.load(file)
                    # 确保日记条目有 essences
                    for entry in diary_data:
                        if "essences" not in entry:
                            entry["essences"] = []
                    diary = diary_data
            
            # 加载"约定"
            promise_path = os.path.join(self.memory_core_dir, "memory_core_promise.json")
            if os.path.exists(promise_path):
                with open(promise_path, "r", encoding="utf-8") as file:
                    promise = json.load(file)
            
            # 加载"计划"
            plan_path = os.path.join(self.memory_core_dir, "memory_core_plan.json")
            if os.path.exists(plan_path):
                with open(plan_path, "r", encoding="utf-8") as file:
                    plan = json.load(file)
            
            # 加载"偏好"
            preference_path = os.path.join(self.memory_core_dir, "memory_core_preference.json")
            if os.path.exists(preference_path):
                with open(preference_path, "r", encoding="utf-8") as file:
                    preference = json.load(file)
            
            # 加载"动机"
            motivation_path = os.path.join(self.memory_core_dir, "memory_core_motivation.json")
            if os.path.exists(motivation_path):
                with open(motivation_path, "r", encoding="utf-8") as file:
                    motivation = json.load(file)
            
            # 加载"关键记忆"
            pivotal_memory_path = os.path.join(self.memory_core_dir, "memory_core_pivotal_memory.json")
            if os.path.exists(pivotal_memory_path):
                with open(pivotal_memory_path, "r", encoding="utf-8") as file:
                    pivotal_memory = json.load(file)
                    
        except Exception as e:
            print(f"⚠️警告| 加载记忆核心失败: {str(e)}")
        
        return diary, promise, plan, preference, motivation, pivotal_memory
    
    def match_essences_with_text(self, text):
        """匹配文本与日记中的Essence"""
        matched_memories = []
        
        # 获取部分日记用于与系统提示词去重
        recent_diary_dates = set()
        recent_diary = self.get_recent_diary(MEMORY_DAYS)
        for entry in recent_diary:
            recent_diary_dates.add(entry["date"])
        
        # 遍历所有日记条目
        for entry in self.memory_core_diary:
            # 跳过已经在"你的记忆"中出现的日记
            if entry["date"] in recent_diary_dates:
                continue
                
            # 检查每个 Essence
            for essence in entry.get("essences", []):
                # 关键词匹配
                if isinstance(text, str) and essence.lower() in text.lower():
                    matched_memories.append({
                        "date": entry["date"],
                        "content": entry["content"],
                        "matched_essence": essence
                    })
                    # 每个日记条目只匹配一次
                    break
        
        return matched_memories
    
    def format_memory_for_prompt(self, days=None):
        """格式化记忆核心用于系统提示词"""
        if days is None:
            days = MEMORY_DAYS
        recent_diary = self.get_recent_diary(days)
        
        # 格式化输出
        memory_text = ""
        
        if self.memory_core_promise:
            memory_text += "## 约定\n"
            for i, promise in enumerate(self.memory_core_promise, 1):
                memory_text += f"{i}. {promise}\n"
        
        if self.memory_core_preference:
            memory_text += "## 用户偏好\n"
            for i, preference in enumerate(self.memory_core_preference, 1):
                memory_text += f"{preference}\n"
        
        if self.memory_core_motivation:
            memory_text += "## 动机\n"
            for i, motivation in enumerate(self.memory_core_motivation, 1):
                memory_text += f"{i}. {motivation}\n"
        
        if self.memory_core_plan:
            memory_text += "## 计划\n"
            for plan_item in self.memory_core_plan:
                memory_text += f"{plan_item['date']}: {plan_item['content']}\n"
        
        if self.memory_core_pivotal_memory:
            memory_text += "## 关键记忆\n"
            for i, memory in enumerate(self.memory_core_pivotal_memory, 1):
                memory_text += f"{memory}\n"
        
        if recent_diary:
            memory_text += "## 日记\n"
            for entry in recent_diary:
                memory_text += f"{entry['date']}: {entry['content']}\n"
        
        return memory_text.strip()

    def get_recent_diary(self, days=None):
        """获取部分日记用于系统提示词"""
        if days is None:
            days = MEMORY_DAYS
        if not self.memory_core_diary:
            return []
        
        # 按日期排序，最新的在前面
        try:
            sorted_diary = sorted(
                self.memory_core_diary, 
                key=lambda x: datetime.strptime(x['date'], "%Y年%m月%d日"), 
                reverse=True
            )
        except ValueError:
            # 兼容旧格式
            sorted_diary = sorted(
                self.memory_core_diary, 
                key=lambda x: datetime.strptime(x['date'], "%m月%d日"), 
                reverse=True
            )
        
        return sorted_diary[:days]

    def get_recent_diary_for_recursion(self, days=2):
        """获取部分日记用于递归总结"""
        if not self.memory_core_diary:
            return []
        
        # 按日期排序，最新的在前面
        try:
            sorted_diary = sorted(
                self.memory_core_diary, 
                key=lambda x: datetime.strptime(x['date'], "%Y年%m月%d日"), 
                reverse=True
            )
        except ValueError:
            # 兼容旧格式
            sorted_diary = sorted(
                self.memory_core_diary, 
                key=lambda x: datetime.strptime(x['date'], "%m月%d日"), 
                reverse=True
            )
        
        return sorted_diary[:days]
        
    def save_memory_core(self, summary_data):
        """保存记忆核心"""
        try:
            # 解析JSON数据
            if isinstance(summary_data, str):
                summary_data = json.loads(summary_data)

            # 保存日记，日记只覆盖相同日期；其余类别新数据覆盖旧数据
            if 'diary' in summary_data:
                # 创建日期到日记条目的映射
                existing_diary_map = {entry['date']: entry for entry in self.memory_core_diary}
                new_diary_map = {entry['date']: entry for entry in summary_data['diary']}
                
                # 更新现有日记中相同日期的条目
                for date, entry in new_diary_map.items():
                    existing_diary_map[date] = entry
                
                # 转换回列表并保持时间顺序
                updated_diary = list(existing_diary_map.values())
                # 兼容旧格式
                try:
                    updated_diary.sort(key=lambda x: datetime.strptime(x['date'], "%Y年%m月%d日"))
                except ValueError:
                    updated_diary.sort(key=lambda x: datetime.strptime(x['date'], "%m月%d日"))
                
                self.memory_core_diary = updated_diary
                diary_path = os.path.join(self.memory_core_dir, "memory_core_diary.json")
                with open(diary_path, "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_diary, file, ensure_ascii=False, indent=4)
            
            # 保存约定
            if 'promise' in summary_data:
                self.memory_core_promise = summary_data['promise']
                promise_path = os.path.join(self.memory_core_dir, "memory_core_promise.json")
                with open(promise_path, "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_promise, file, ensure_ascii=False, indent=4)
            
            # 保存用户偏好
            if 'preference' in summary_data:
                self.memory_core_preference = summary_data['preference']
                preference_path = os.path.join(self.memory_core_dir, "memory_core_preference.json")
                with open(preference_path, "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_preference, file, ensure_ascii=False, indent=4)
            
            # 保存计划
            if 'plan' in summary_data:
                self.memory_core_plan = summary_data['plan']
                plan_path = os.path.join(self.memory_core_dir, "memory_core_plan.json")
                with open(plan_path, "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_plan, file, ensure_ascii=False, indent=4)
            
            # 保存动机
            if 'motivation' in summary_data:
                self.memory_core_motivation = summary_data['motivation']
                motivation_path = os.path.join(self.memory_core_dir, "memory_core_motivation.json")
                with open(motivation_path, "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_motivation, file, ensure_ascii=False, indent=4)
            
            # 保存关键记忆
            if 'pivotal_memory' in summary_data:
                self.memory_core_pivotal_memory = summary_data['pivotal_memory']
                pivotal_memory_path = os.path.join(self.memory_core_dir, "memory_core_pivotal_memory.json")
                with open(pivotal_memory_path, "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_pivotal_memory, file, ensure_ascii=False, indent=4)
            
            print("✅信息| 记忆核心已保存")
        except Exception as e:
            print(f"⚠️警告| 保存记忆核心失败: {str(e)}")

    def play_opening_line(self):
        """处理开场白播放"""
        if self.tts_success and hasattr(self, 'opening_line'):
            return self.process_ai_response(self.opening_line)
        return False

    def check_config_integrity(self):
        """检查关键配置完整性"""
        if not self.VOLC_ACCESS_KEY or not self.VOLC_SECRET_KEY:
            print("⚠️警告| VOLC 相关环境变量未设置，部分功能可能不可用")
        else:
            print("✅信息| VOLC 配置已就绪")

    def init_audio_system(self):
        """初始化音频系统"""
        pygame.mixer.init()

    def clear_tts_output(self):
        """音频清理"""
        audio_dir = "Debug"
        os.makedirs(audio_dir, exist_ok=True)
        for filename in os.listdir(audio_dir):
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(audio_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"⚠️警告| 音频清理失败: {e}")
        return audio_dir
    
    def load_short_term_memory_from_file(self):
        """加载短期记忆"""
        file_path = "short_term_memory.json"
        if not os.path.exists(file_path):
            print("✅信息| 未找到短期记忆")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # 过滤"system"消息
            filtered_data = [msg for msg in data if msg.get("role") != "system"]

            # 分别加载指定条数用于上下文和对话总结
            recent_messages_for_context = filtered_data[-SHORT_TERM_MEMORY_MESSAGES:]
            recent_messages_for_summary = filtered_data[-4:]

            # 添加到后端历史和后端长历史
            self.backend_history.extend(recent_messages_for_context)
            self.backend_long_history.extend(recent_messages_for_summary)
            
            print(f"✅信息| 后端历史条数: {len(self.backend_history)}")
            print(f"✅信息| 后端长历史条数: {len(self.backend_long_history)}")

        except Exception as e:
            print(f"⚠️警告| 加载短期记忆出错: {e}")

    def add_timestamp_to_messages(self):
        """为消息添加时间戳"""
        current_time = self.get_timeinfo_1()
        for msg in self.backend_history:
            if "timestamp" not in msg:
                msg["timestamp"] = current_time

    def save_long_term_memory(self):
        """保存长期记忆"""
        # 只保存不调用，未完善且有BUG
        try:
            file_path = "long_term_memory.json"
            
            # 过滤"system"消息
            non_system_messages = [msg for msg in self.backend_history if msg.get("role") != "system"]
            
            if not non_system_messages:
                return
                
            # 读取长期记忆
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # 只保存新消息
            new_messages = []
            for msg in non_system_messages:
                if msg not in existing_data:
                    new_messages.append(msg)
            
            if not new_messages:
                print("✅信息| 没有新消息需要保存到长期记忆")
                return
                
            # 合并数据
            updated_data = existing_data + new_messages
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)

            print(f"✅信息| 保存 {len(new_messages)} 条新消息到长期记忆")

        except Exception as e:
            print(f"⚠️警告| 保存长期记忆出错: {e}")

    def get_timeinfo_1(self):
        """获取时间信息：x年x月x日周x x:x"""
        current_time = datetime.now()
        formatted_date = current_time.strftime("%Y年%m月%d日")
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        formatted_weekday = weekdays[current_time.weekday()]
        formatted_time = current_time.strftime("%H:%M")
        return f"{formatted_date}{formatted_weekday} {formatted_time}"
    
    def get_timeinfo_2(self):
        """获取时间信息：x月x日周x x点x分"""
        current_time = datetime.now()
        formatted_date = current_time.strftime("%m月%d日")
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        formatted_weekday = f"周{weekdays[current_time.weekday()]}"
        formatted_time = current_time.strftime("%H点%M分")
        return f"{formatted_date}{formatted_weekday} {formatted_time}"

    def get_timeinfo_3(self):
        """获取时间信息：x年x月x日"""
        current_time = datetime.now()
        return current_time.strftime("%Y年%m月%d日")

    def clean_brackets(self, text):
        """清理重复的括号"""
        if not isinstance(text, str):
            return text
        
        # 清理中文重复括号
        text = re.sub(r'（{2,}', '（', text)
        text = re.sub(r'）{2,}', '）', text)
        # 清理英文重复括号
        text = re.sub(r'\({2,}', '(', text)
        text = re.sub(r'\){2,}', ')', text)
        return text

    def clean_single_square_brackets(self, text):
        """清理单个的【和】"""
        if not isinstance(text, str):
            return text
        # 清理单个方括号，但保留成对的 【】 用于思维链标记
        text = text.replace('【', '').replace('】', '')
        return text

    def test_chatai_service(self):
        """测试ChatAI服务"""
        print("✅信息| 测试ChatAI……")
        try:
            # 构造包含时间的请求信息
            time_info = f"{self.get_timeinfo_2()}"
            
            # 检查两个条件
            short_term_memory_exists = os.path.exists("short_term_memory.json")
            memory_core_diary_exists = os.path.exists(os.path.join("memory_core", "memory_core_diary.json"))
            
            # 根据条件设置不同的请求消息
            if short_term_memory_exists or memory_core_diary_exists:
                # 两个文件中存在任何一个，使用原来的请求消息
                test_content = f"<OOC>请依据上下文和'日记'进行回复，如果时间跨度较小则侧重上下文推理，否则应该重点推理该跨度时间内可能做的事情，比如一些日常；回复不要附带'🤐' | {time_info}</OOC>"
            else:
                # 两个文件都不存在，使用新的请求消息
                test_content = f"<OOC>现在是你和用户第一次见面，你刚刚从充电舱中醒来，请和用户打招呼吧 | {time_info}</OOC>"

            # 清理后端历史和后端长历史中的重复括号
            for history in [self.backend_history, self.backend_long_history]:
                for msg in history:
                    if "content" in msg and isinstance(msg["content"], str):
                        msg["content"] = self.clean_brackets(msg["content"])
            
            # 添加清理后的测试消息到后端历史和后端长历史
            cleaned_test_content = self.clean_brackets(test_content)
            self.backend_history.append({"role": "user", "content": cleaned_test_content})
            self.backend_long_history.append({"role": "user", "content": cleaned_test_content})
            print(f"✅信息|" + "-" * 100)
            print(f"✅信息| 清理后的测试消息：{cleaned_test_content}")
            
            # 调用`请求ChatAI`
            content, reasoning_content, tokens_used = self.call_chatai()
            
            # 清理AI回复
            content = content.strip()
            reasoning_content = reasoning_content.strip() if reasoning_content else ""

            # 打印调试信息
            print(f"✅信息| 处理前的AI原始回复：{content}")
            print(f"✅信息| 处理前的思维链：{reasoning_content if reasoning_content else '无'}")

            # 根据 USE_COT 决定处理逻辑
            if USE_COT:
                if reasoning_content:
                    # 存在思维链：先清洗AI原始回复的单个方括号，再组合
                    cleaned_content = self.clean_single_square_brackets(content)
                    combined_content = f"【{reasoning_content}】\n\n{cleaned_content}"
                    print(f"✅信息| 清洗后的AI原始回复：{cleaned_content}")
                    print(f"✅信息| 组合后的回复：{combined_content}")
                else:
                    # 不存在思维链：只清洗AI原始回复的单个方括号，不组合
                    cleaned_content = self.clean_single_square_brackets(content)
                    combined_content = cleaned_content
                    print(f"✅信息| 清洗后的AI原始回复：{cleaned_content}")
                    print(f"✅信息| USE_COT 为 True 但无思维链，直接输出清洗后的回复")
            else:
                # USE_COT 为False：直接输出AI原始回复，不清洗
                combined_content = content
                print(f"✅信息| USE_COT 为 False，直接输出原始回复，不清洗")
            
            print(f"✅信息| AI对话内容：{content}")
            print(f"✅信息| 最终输出内容：{combined_content}")
            
            # 添加组合后的AI回复到后端历史和后端长历史，清理重复括号
            cleaned_combined_content = self.clean_brackets(combined_content)
            self.backend_history.append({"role": "assistant", "content": cleaned_combined_content})
            self.backend_long_history.append({"role": "assistant", "content": cleaned_combined_content})
            
            print(f"✅信息| ChatAI连接正常")
            print(f"✅信息| Token: {tokens_used} | 条数：{len(self.backend_history)}")
            return True
        except Exception as e:
            print(f"❌错误| ChatAI API 错误: {str(e)}")
            print("✅信息| 将使用模拟回复模式")
            return False
        
    def test_tts_service(self):
        """测试TTS服务"""
        print("✅信息| 测试TTS服务……")
        try:
            test_dir = os.path.join(self.audio_dir)
            if not os.access(test_dir, os.W_OK):
                print("❌错误| TTS输出文件夹不可写")
                return False
                
            print("✅信息| TTS服务连接正常")
            return True
        except Exception as e:
            print(f"❌错误| TTS文件夹访问失败: {str(e)}")
            return False

    def generate_opening_line(self):
        """将测试回复作为开场白"""
        if not self.use_chatai:
            return "欸……连接不上我的大脑😵"
        
        # 获取最后一条AI回复内容
        last_message_content = self.backend_history[-1]["content"]
        
        # 检查是否包含思维链格式
        if USE_COT and last_message_content.startswith("【") and "】\n\n" in last_message_content:
            # 分离思维链和最终回复
            parts = last_message_content.split("】\n\n", 1)
            if len(parts) > 1:
                # 返回最终回复部分
                return parts[1]
        
        # 如果不包含思维链格式，直接返回原内容
        return last_message_content

    def update_system_prompt_with_memories(self, memories):
        """更新系统提示词以包含相关记忆"""
        # 获取包含"你的记忆"的系统提示词
        system_prompt = self.system_prompt_2

        # 添加"相关记忆"
        if memories:
            system_prompt += "\n## 相关记忆(和最新对话有关的记忆)"
            for memory in memories:
                system_prompt += f"\n{memory['date']}: {memory['content']}"

        return system_prompt

    def call_chatai(self):
        """请求ChatAI"""
        # 清理后端历史和后端长历史中的重复括号
        for history in [self.backend_history, self.backend_long_history]:
            for msg in history:
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = self.clean_brackets(msg["content"])

        # 调用`更新系统提示词以包含相关记忆`
        if self.backend_history and self.backend_history[0]["role"] == "system":
            self.backend_history[0]["content"] = self.update_system_prompt_with_memories(self.related_memories)

        # 调用`清理历史中的思维链`
        self.clean_old_reasoning_content()

        # 打印后端历史，为美观可注释
        print("✅信息| 后端历史:")
        for i, msg in enumerate(self.backend_history):
            print(f"      [{i}] {msg['role']}: {msg['content'][:9999]}{'...' if len(msg['content']) > 9999 else ''}")
        
        # 上下文清理
        # 分离后端历史
        system_message = self.backend_history[0]
        dialogue_history = self.backend_history[1:]

        while len(dialogue_history) > MAX_HISTORY_MESSAGES - 1:  # -1 为系统提示词保留位置
            if len(dialogue_history) >= 2:  
                removed_messages = dialogue_history[:2]
                dialogue_history = dialogue_history[2:]
                print(f"✅信息| 条数已达 {MAX_HISTORY_MESSAGES}，移除最早一轮对话：")
                for msg in removed_messages:
                    print(f"      - {msg['role']}: {msg['content'][:30]}……")
            else:
                break

        # 重建后端历史并更新
        self.backend_history = [system_message] + dialogue_history

        try:
            params = {
                "model": MODEL,
                "messages": self.backend_history,
                "temperature": 1.3,
                "max_tokens": 8192
            }
            if USE_BETA:
                params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**params)

            # 获取AI回复和Token
            content = response.choices[0].message.content
            
            # 获取思维链内容，如果不存在则为空值
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
            
            tokens_used = response.usage.total_tokens
            
            # 无论 USE_COT 设置如何，始终打印思维链，无论是否存在
            if reasoning_content:
                print(f"✅信息| AI思维链：\n{reasoning_content}")
            else:
                print(f"✅信息| AI思维链：无")
            
            return content, reasoning_content, tokens_used
        
        except Exception as e:
            print(f"❌错误| ChatAI API 异常: {str(e)}")
            return "欸……连接不上我的大脑😵", "", None
        
    def clean_old_reasoning_content(self):
        """清理前后端历史中的思维链"""
        if USE_COT:
            # 只保留最后2条AI消息中的思维链
            # 1. 处理`backend_history`
            self._clean_history_with_keep_count(self.backend_history, keep_count=2)
            # 2. 处理`backend_long_history`
            self._clean_history_with_keep_count(self.backend_long_history, keep_count=2)
        else:
            # 清理所有思维链
            self._clean_all_history(self.backend_history)
            self._clean_all_history(self.backend_long_history)

    def _clean_history_with_keep_count(self, history, keep_count=2):
        """清理历史消息，并保存部分思维链"""
        # 找出所有的AI消息索引
        assistant_indices = []
        for i, msg in enumerate(history):
            if msg["role"] == "assistant":
                assistant_indices.append(i)
        
        # 计算需要保留的条数
        total_assistants = len(assistant_indices)
        keep_from_index = max(0, total_assistants - keep_count)
        
        # 清理历史
        for i, msg in enumerate(history):
            if msg["role"] == "assistant":
                content = msg["content"]
                # 检查是否包含思维链格式
                if content.startswith("【") and "】\n\n" in content:
                    # 获取当前AI在列表中的位置
                    assistant_pos = assistant_indices.index(i)
                    
                    # 只保留最后 keep_count 条思维链
                    if assistant_pos < keep_from_index:
                        # 提取最终回复部分，清理思维链
                        parts = content.split("】\n\n", 1)
                        if len(parts) > 1:
                            final_content = parts[1]
                            history[i]["content"] = final_content
                            print(f"✅信息| 已清理历史AI回复中的思维链：位置{assistant_pos}，保留最终回复: {final_content[:10]}……")

    def _clean_all_history(self, history):
        """清理历史中所有思维链"""
        for i, msg in enumerate(history):
            if msg["role"] == "assistant":
                content = msg["content"]
                # 检查是否包含思维链格式
                if content.startswith("【") and "】\n\n" in content:
                    # 提取最终回复部分
                    parts = content.split("】\n\n", 1)
                    if len(parts) > 1:
                        final_content = parts[1]
                        # 更新为只有最终回复
                        history[i]["content"] = final_content
                        print(f"✅信息| 已清理历史AI回复中的思维链，保留最终回复: {final_content[:10]}……")

    def handle_exit_detection(self, ai_response=None):
        """处理退出标记"""
        # 检测是否包含退出标记
        if ai_response is not None:
            should_exit = "🤐" in ai_response
        else:
            # 主动触发时，默认为True
            should_exit = True

        if should_exit:
            print("✅信息| 触发退出流程，开始递归总结")
            
            # 调用`添加时间信息到记忆`
            self.add_time_info_to_memory()
            # 调用方法进行递归总结
            self.request_summary()
            self.remove_summary_from_short_term_memory()
            self.save_long_term_memory()
        return should_exit
    
    def add_time_info_to_memory(self):
        """添加时间信息到记忆"""
        try:
            # 获取当前时间
            time_info = f"<OOC>{self.get_timeinfo_2()}</OOC>"
            
            # 读取短期记忆文件
            file_path = "short_term_memory.json"
            if not os.path.exists(file_path):
                return
                
            with open(file_path, 'r', encoding='utf-8') as file:
                short_term_memory = json.load(file)
            
            # 确保有足够的历史消息
            if len(short_term_memory) >= 2:
                # 获取总结前最后一轮对话
                second_last_msg = short_term_memory[-2]
                
                # 检查是否已经包含时间信息，避免重复添加
                if "<OOC>" not in second_last_msg["content"]:
                    # 在消息内容末尾添加时间信息
                    second_last_msg["content"] += f" {time_info}"
                    
                    # 保存修改后的短期记忆
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(short_term_memory, file, ensure_ascii=False, indent=4)
                    
                    print(f"✅信息| 已在短期记忆中添加时间信息: {time_info}")
                    
                    # 更新后端历史中对应的消息
                    if len(self.backend_history) >= 2:
                        # 检查是否已包含时间信息
                        if "<OOC>" not in self.backend_history[-2]["content"]:
                            self.backend_history[-2]["content"] += f" {time_info}"
                    
                    # 更新后端长历史中对应的消息
                    if len(self.backend_long_history) >= 2:
                        # 检查是否已包含时间信息
                        if "<OOC>" not in self.backend_long_history[-2]["content"]:
                            self.backend_long_history[-2]["content"] += f" {time_info}"
                else:
                    print("✅信息| 时间信息已存在，跳过添加")
        except Exception as e:
            print(f"⚠️警告| 添加时间信息到短期记忆失败: {str(e)}")

    def chinese_to_translate_japanese(self, text):
        """中译日或直接返回文本"""
        if not USE_TRANSLATION:
            # 不使用翻译时，直接返回输入文本
            return text
        
        # 使用翻译时，调用火山翻译API
        def translate_request():
            # 服务信息
            service_info = ServiceInfo(
                'translate.volcengineapi.com',
                {'Content-Type': 'application/json'},
                Credentials(self.VOLC_ACCESS_KEY, self.VOLC_SECRET_KEY, 'translate', 'cn-north-1'),
                5,
                5
            )
            
            # API信息
            api_info = {
                'translate': ApiInfo(
                    'POST', 
                    '/', 
                    {'Action': 'TranslateText', 'Version': '2020-06-01'},
                    {}, 
                    {}
                )
            }
            
            # 创建服务实例并发送请求
            service = Service(service_info, api_info)
            body = {
                'TargetLanguage': 'ja',  # 目标语言
                'TextList': [text],
                'SourceLanguage': 'zh'   # 源语言
            }
            
            response = json.loads(service.json('translate', {}, json.dumps(body)))
            
            # 获取翻译结果
            if "TranslationList" in response and len(response["TranslationList"]) > 0:
                return response["TranslationList"][0]["Translation"]
            else:
                print(f"❌错误| 火山翻译API返回异常: {json.dumps(response, indent=2, ensure_ascii=False)}")
                return None
        
        # 错误处理：请求超时
        max_retries = 1  # 最大重试次数
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                return translate_request()
            except Exception as e:
                # 判断是否为超时错误
                is_timeout_error = "Read timed out" in str(e) or "timed out" in str(e).lower()
                
                if is_timeout_error and retry_count < max_retries:
                    print(f"❌错误| 火山翻译异常: {str(e)}")
                    print(f"⚠️警告| 检测到请求超时，正在进行第 {retry_count + 1} 次重试...")
                    retry_count += 1
                    continue
                else:
                    print(f"❌错误| 火山翻译异常: {str(e)}")
                    traceback.print_exc()
                    return None
        
        return None

    def extract_dialogue_content(self, text):
        """提取说话内容"""
        # 如果 USE_BETA 为True，先尝试解析JSON提取 content 字段
        if USE_BETA:
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "content" in data:
                    text = data["content"]
            except json.JSONDecodeError:
                pass # 解析失败则保持原text继续清洗

        # 匹配中文括号，并多次匹配
        while True:
            # 匹配包括换行符在内的所有字符
            new_text = re.sub(r'（.*?）', '', text, flags=re.DOTALL)
            if new_text == text:
                break
            text = new_text
        
        # 匹配英文括号，并多次匹配
        while True:
            new_text = re.sub(r'\(.*?\)', '', text, flags=re.DOTALL)
            if new_text == text:
                break
            text = new_text
        
        # 对提取的内容进行清洗
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        cleaned_text = cleaned_text.replace("...", "……")
        cleaned_text = re.sub(r'[Zz]{3,}', '', cleaned_text)
        
        print(f"✅信息| 处理后的内容: {cleaned_text}")
        return cleaned_text
        
    def text_to_speech(self, text):
        """TTS和播放"""
        try:
            # 构建请求数据
            request_data = REF_AUDIO_CONFIG.copy()
            request_data["text"] = text
            print(f"✅信息| TTS文本: {text}")
            print(f"✅信息|" + "-" * 100)
            
            # 调用TTS API
            response = requests.post(TTS_API_URL, json=request_data)
            
            # 检查响应
            if response.status_code != 200:
                print(f"❌错误| TTS错误: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"✅信息| {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                except:
                    print(f"✅信息| {response.text[:200]}")
                return False
            
            # 保存音频
            os.makedirs(self.audio_dir, exist_ok=True)
            timestamp = int(time.time())
            audio_path = os.path.join(self.audio_dir, f"response_{timestamp}.wav")
            
            with open(audio_path, "wb") as f:
                f.write(response.content)
            
            # 播放音频
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)        
            return True
            
        except Exception as e:
            # 确保Debug目录存在
            debug_dir = "Debug"
            os.makedirs(debug_dir, exist_ok=True)
            log_path = os.path.join(debug_dir, "TTS_Error.log")

            # 获取完整 traceback 信息
            error_traceback = traceback.format_exc()

            # 写入错误日志
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(error_traceback)

            # 将异常信息转为小写便于匹配
            error_msg = str(e).lower()
            error_traceback_lower = error_traceback.lower()
            combined_error = error_msg + "\n" + error_traceback_lower

            # 定义 服务未运行 类错误的关键词
            service_not_running_keywords = [
                "connection refused",
                "连接被拒绝",
                "newconnectionerror",
                "max retries exceeded",
                "connectionerror",
                "failed to establish a new connection",
                "errno 111",
                "errno 10061",
            ]

            # 判断是否属于 服务未运行 类型错误
            if any(keyword in combined_error for keyword in service_not_running_keywords):
                print(f"❌错误| GPT-SoVITS 服务未运行，已打印详细日志至 {log_path}")
            else:
                # 未知类型异常：输出通用提示
                print("❌错误| GPT-SoVITS 服务发生未知异常，已打印日志")

            return False

    def select_related_memories(self, ai_response_text, user_input_text):
        """根据AI回复和用户输入选择相关记忆"""
        # 匹配AI回复
        ai_matched_memories = self.match_essences_with_text(ai_response_text) if ai_response_text else []
        # 匹配用户输入
        user_matched_memories = self.match_essences_with_text(user_input_text)

        # 合并并去重，根据日期去重
        all_matched_memories = ai_matched_memories + user_matched_memories
        unique_memories = []
        seen_dates = set()

        for memory in all_matched_memories:
            if memory["date"] not in seen_dates:
                seen_dates.add(memory["date"])
                unique_memories.append(memory)

        # 按关键词分组
        memories_by_essence = {}
        for memory in unique_memories:
            essence = memory["matched_essence"]
            if essence not in memories_by_essence:
                memories_by_essence[essence] = []
            memories_by_essence[essence].append(memory)

        # 获取所有关键词
        essences = list(memories_by_essence.keys())
        num_essences = len(essences)

        selected_memories = []

        # 具体匹配细则
        if num_essences == 0:
            # 没有匹配到任何关键词
            return []
        elif num_essences == 1:
            # 1个关键词时，取3条固定+2条随机，共5条
            memories = memories_by_essence[essences[0]]
            if len(memories) <= 3:
                selected_memories = memories
            else:
                # 前3条固定
                selected_memories = memories[:3]
                # 从剩余中随机取2条
                remaining = memories[3:]
                if len(remaining) <= 2:
                    selected_memories.extend(remaining)
                else:
                    selected_memories.extend(random.sample(remaining, 2))
        elif num_essences == 2:
            # 2个关键词时，每个关键词取1条，再从这关键词池中取3条随机的，共5条
            for essence in essences:
                if memories_by_essence[essence]:
                    selected_memories.append(memories_by_essence[essence][0])
            
            # 收集所有记忆（排除已选的）
            all_memories = []
            for essence in essences:
                all_memories.extend(memories_by_essence[essence])
            
            # 移除已选的
            remaining_memories = [m for m in all_memories if m not in selected_memories]
            
            # 随机选择3条
            if len(remaining_memories) <= 3:
                selected_memories.extend(remaining_memories)
            else:
                selected_memories.extend(random.sample(remaining_memories, 3))
        elif num_essences == 3:
            # 3个关键词时，每个关键词取1条，再从这关键词池中取2条随机的，共5条
            for essence in essences:
                if memories_by_essence[essence]:
                    selected_memories.append(memories_by_essence[essence][0])
            
            # 收集所有记忆（排除已选的）
            all_memories = []
            for essence in essences:
                all_memories.extend(memories_by_essence[essence])
            
            # 移除已选的
            remaining_memories = [m for m in all_memories if m not in selected_memories]
            
            # 随机选择2条
            if len(remaining_memories) <= 2:
                selected_memories.extend(remaining_memories)
            else:
                selected_memories.extend(random.sample(remaining_memories, 2))
        elif num_essences == 4:
            # 4个关键词时，每个关键词取1条，再从这关键词池中取1条随机的，共5条
            for essence in essences:
                if memories_by_essence[essence]:
                    selected_memories.append(memories_by_essence[essence][0])
            
            # 收集所有记忆（排除已选的）
            all_memories = []
            for essence in essences:
                all_memories.extend(memories_by_essence[essence])
            
            # 移除已选的
            remaining_memories = [m for m in all_memories if m not in selected_memories]
            
            # 随机选择1条
            if remaining_memories:
                selected_memories.append(random.choice(remaining_memories))
        elif num_essences == 5:
            # 5个关键词时，每个关键词取1条，不取随机，共5条
            for essence in essences:
                if memories_by_essence[essence]:
                    selected_memories.append(memories_by_essence[essence][0])
        else:
            # 5个以上的关键词时，从所有的关键词池中随机取5条，共5条
            # 收集所有记忆的第一条
            all_first_memories = []
            for essence in essences:
                if memories_by_essence[essence]:
                    all_first_memories.append(memories_by_essence[essence][0])
            
            # 随机选择5条
            if len(all_first_memories) <= 5:
                selected_memories = all_first_memories
            else:
                selected_memories = random.sample(all_first_memories, 5)

        return selected_memories

    def process_user_message(self, user_input, play_tts=True):
        """处理用户消息"""
        # 清理后端历史和后端长历史中的重复括号
        for history in [self.backend_history, self.backend_long_history]:
            for msg in history:
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = self.clean_brackets(msg["content"])

        # 使用封装的实例方法选择相关记忆
        self.related_memories = self.select_related_memories(self.last_ai_response, user_input)

        # 清理用户输入中的重复括号并添加到后端历史和后端长历史
        cleaned_user_input = self.clean_brackets(user_input)
        self.backend_history.append({"role": "user", "content": cleaned_user_input})
        self.backend_long_history.append({"role": "user", "content": cleaned_user_input})
        
        print(f"✅信息| 用户消息（原始）: {user_input}")
        print(f"✅信息| 用户消息（清理后）: {cleaned_user_input}")
        
        # 打印最终选择的记忆
        if self.related_memories:
            print(f"✅信息| 最终选择的记忆 ({len(self.related_memories)}条): {[m['matched_essence'] for m in self.related_memories]}")
        else:
            print("✅信息| 未匹配到相关记忆或相关记忆已在'你的记忆'部分")

        # 调用`请求ChatAI`并获取回复
        tokens_used = None
        if self.use_chatai:
            # 调用`请求ChatAI`
            content, reasoning_content, tokens_used = self.call_chatai()

            # 清理AI回复
            content = content.strip()
            reasoning_content = reasoning_content.strip() if reasoning_content else ""

            # 打印调试信息
            print(f"✅信息| AI原始回复（处理前）：{content}")
            print(f"✅信息| 思维链（处理前）：{reasoning_content if reasoning_content else '无'}")

            # 根据 USE_COT 决定处理逻辑
            if USE_COT:
                if reasoning_content:
                    # 存在思维链：先清洗AI原始回复的单个方括号，再组合
                    cleaned_content = self.clean_single_square_brackets(content)
                    combined_content = f"【{reasoning_content}】\n\n{cleaned_content}"
                    print(f"✅信息| AI原始回复（清洗后）：{cleaned_content}")
                    print(f"✅信息| 组合后的回复：{combined_content}")
                else:
                    # 不存在思维链：只清洗AI原始回复的单个方括号，不组合
                    cleaned_content = self.clean_single_square_brackets(content)
                    combined_content = cleaned_content
                    print(f"✅信息| AI原始回复（清洗后）：{cleaned_content}")
                    print(f"✅信息| USE_COT 为True但无思维链，直接输出清洗后的回复")
            else:
                # USE_COT 为False：直接输出AI原始回复，不清洗
                combined_content = content
                print(f"✅信息| USE_COT 为False，直接输出原始回复（不清洗）")

            # 保存当前AI回复，用于下一次匹配，使用原始回复，不包含思维链
            self.last_ai_response = content
            
            # 清理组合内容中的重复括号并添加到后端历史和后端长历史
            cleaned_combined_content = self.clean_brackets(combined_content)
            self.backend_history.append({"role": "assistant", "content": cleaned_combined_content})
            self.backend_long_history.append({"role": "assistant", "content": cleaned_combined_content})

            # 退出检测，使用原始回复检测
            should_exit = False
            if self.tts_success and play_tts:
                print(f"✅信息| 退出标记检测结果: {'🤐' in content}")
                should_exit = self.process_ai_response(content)
            else:
                print(f"✅信息| 退出标记检测结果: {'🤐' in content}")
                should_exit = "🤐" in content

            # 保存短期记忆
            try:
                file_path = "short_term_memory.json"
                # 保存前清理历史中的重复括号
                cleaned_backend_history = []
                for msg in self.backend_history:
                    cleaned_msg = msg.copy()
                    if "content" in cleaned_msg and isinstance(cleaned_msg["content"], str):
                        cleaned_msg["content"] = self.clean_brackets(cleaned_msg["content"])
                    cleaned_backend_history.append(cleaned_msg)
                
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(cleaned_backend_history, file, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"⚠️警告| 保存`backend_history`到文件失败: {str(e)}")

            # 如果检测到退出标记，请求总结
            if should_exit:
                self.handle_exit_detection(content)  # 使用原始回复

            # 调用`保存长期记忆`
            self.save_long_term_memory()

            print(f"✅信息| AI对话内容：{content}")
            print(f"✅信息| 最终输出内容：{combined_content}")
            print(f"✅信息| Token: {tokens_used} | 请求条数：{len(self.backend_history)} | 总结条数：{len(self.backend_long_history)}")
            
            # 返回原始回复给前端，确保UI不显示思维链
            return content, should_exit
        else:
            ai_response = f"ChatAI不可用 {user_input} "
            tokens_used = 0
            return ai_response, False

    def get_summary_history(self):
        """获取用于对话总结的历史"""
        # 只包含2天日记
        memory_for_summary = self.format_memory_for_prompt(2)
        summary_system_prompt = self.system_prompt_1 + "\n\n你的记忆:\n" + memory_for_summary
        
        # 使用后端长历史
        dialogue_history = self.backend_long_history
        print(f"✅信息| 后端长历史总条数: {len(dialogue_history)}")
        
        if len(dialogue_history) > SUMMARY_HISTORY_LENGTH:
            dialogue_history = dialogue_history[-SUMMARY_HISTORY_LENGTH:]
            print(f"✅信息| 截取最后{SUMMARY_HISTORY_LENGTH}条用于总结")
        else:
            print(f"✅信息| 使用全部{len(dialogue_history)}条用于总结")
        
        # 返回用于对话总结的历史
        summary_history = [{"role": "system", "content": summary_system_prompt}] + dialogue_history
        print(f"✅信息| 最终用于总结的条数: {len(summary_history)}")
        
        print("✅信息| 用于总结的历史记录详细内容:")
        for i, msg in enumerate(summary_history):
            print(f"      [{i}] {msg['role']}: {msg['content'][:9999]}{'...' if len(msg['content']) > 9999 else ''}")
        
        return summary_history
    
    def save_summary_result(self, summary_type, result):
        """保存总结结果"""
        try:
            debug_dir = "Debug"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # 文件名
            filename = f"{debug_dir}/{summary_type}.json"
            
            # 准备数据
            summary_data = {
                "type": summary_type,
                "timestamp": int(time.time()),
                "formatted_time": self.get_timeinfo_1(),
                "result": result
            }
            
            # 保存到文件
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(summary_data, file, ensure_ascii=False, indent=4)
            
            print(f"✅信息| {summary_type}结果已保存到 {filename}")
        except Exception as e:
            print(f"⚠️警告| 保存{summary_type}结果失败: {str(e)}")

    def save_summary_messages(self, summary_type, messages):
        """保存总结消息列表"""
        try:
            debug_dir = "Debug"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # 文件名
            filename = f"{debug_dir}/{summary_type}_messages.json"
            
            # 准备数据
            summary_data = {
                "type": summary_type,
                "timestamp": int(time.time()),
                "formatted_time": self.get_timeinfo_1(),
                "messages": messages
            }
            
            # 保存到文件
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(summary_data, file, ensure_ascii=False, indent=4)
            
            print(f"✅信息| {summary_type}消息列表已保存到 {filename}")
            print(f"✅信息| 正在总结中……")
        except Exception as e:
            print(f"⚠️警告| 保存{summary_type}消息列表失败: {str(e)}")
        
    def remove_summary_from_short_term_memory(self):
        """从短期记忆中删除总结相关的消息"""
        try:
            file_path = "short_term_memory.json"
            if not os.path.exists(file_path):
                return
                
            # 读取短期记忆
            with open(file_path, 'r', encoding='utf-8') as file:
                short_term_memory = json.load(file)
            
            # 查找并删除总结相关的消息
            if len(short_term_memory) >= 2:
                last_two_messages = short_term_memory[-2:]
                # 检查特定条件
                summary_request_found = any(
                    msg.get("role") == "user" and 
                    "请以第一人称总结以上对话" in msg.get("content", "")
                    for msg in last_two_messages
                )
                
                summary_response_found = any(
                    msg.get("role") == "assistant" and 
                    msg.get("content") and 
                    not "🤐" in msg.get("content", "")
                    for msg in last_two_messages
                )
                
                # 移除总结消息
                if summary_request_found and summary_response_found:
                    short_term_memory = short_term_memory[:-2]
                    
                    # 保存修改后的短期记忆
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(short_term_memory, file, ensure_ascii=False, indent=4)
                    print("✅信息| 已从短期记忆中删除总结相关的消息")
        except Exception as e:
            print(f"⚠️警告| 从短期记忆中删除总结消息失败: {str(e)}")

    def call_chatai_for_summary(self, messages):
        """请求总结"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.8,
                max_tokens=8192,
                response_format={"type": "json_object"}
            )

            # 获取AI回复和Token
            ai_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            return ai_response, tokens_used
        
        except Exception as e:
            print(f"❌错误| 总结API调用异常: {str(e)}")
            return "❌错误| 总结API调用失败", None

    def request_summary(self):
        """请求递归总结"""
        try:
            # 时间信息已在`handle_exit_detection`中已经添加
            
            # 获取用于总结的历史记录
            summary_history = self.get_summary_history()
            
            # 保存对话总结的消息列表
            self.save_summary_messages("dialogue_summary", summary_history)
            
            # 对话总结请求
            summary_request = {
                "role": "user", 
                "content": """
            <OOC>
            请总结以上对话：
            # 视角要求
            使用亚托莉的视角，即"我"

            # 内容整理
            1. 保持内容简洁清晰，主语和对象描述正确
            2. 根据对话内容修改日记、约定、用户偏好、计划、动机和关键记忆
            3. 删除已完成或已过期的计划和动机

            # 输出规范
            ## diary: 日记
            例：
            ```
            x年x月x日：凌晨，主人趁我睡觉偷偷亲了我。早上，我早起给主人做了早餐，然后在床边等待主人醒来，看着他睡觉的样子感觉好幸福……
            ```
            ## promise: 约定，长期有效的约定
            例：
            ```
            我们约定永远不分开
            我向主人承诺要好好听话
            ```
            ## preference: 用户偏好，用户的偏好和信息，细分为癖好、喜欢、讨厌、习惯、信息、补充等六类
            例：
            ```
            癖好：萝莉控
            喜欢：吃辣
            讨厌：洋葱
            习惯：晚睡
            信息：男生、名叫xx
            ……
            ```
            ## plan: 计划，你的未来计划
            例：
            ```
            x月x日早上：保持最完美的形象和主人一起回家
            明天：自己去买菜
            ```
            ## motivation: 动机，你未实现的内心渴望或驱动力
            例：
            ```
            想成为让主人骄傲的仿生人
            ```
            ## pivotal_memory: 关键记忆，发生过的里程碑事件，需细分为重要经历、允许、接受等三类
            1. 不要包含时间信息，仅记录过去式
            2. 允许和接受需再次细分，如亲嘴和亲脸都是头部，可以分类在一起
            例：
            ```
            重要经历：1. 我和主人在同一张床上一起睡觉；2. 主人趁我睡觉偷亲过我
            允许：1. 我允许主人亲我的脸、亲嘴；2. 我允许主人抱我、背我
            接受：1. 我接受主人摸我的头；2. 和我一起洗澡
            ```
            
            # 请使用以下JSON格式输出：
            {
                "diary": [{"date": "x年x月x日", "content": "内容"}],
                "promise": ["约定"],
                "preference": ["用户偏好"], 
                "plan": [{"date": "时间", "content": "内容"}],
                "motivation": ["动机"],
                "pivotal_memory": ["关键记忆"]
            >
            }
            </OOC>
            """.strip()
            }
            
            # 添加总结请求到历史记录
            summary_history.append(summary_request)
            
            # 使用专门的总结方法获取总结
            current_summary, _ = self.call_chatai_for_summary(summary_history)
            
            # 保存对话总结结果
            self.save_summary_result("dialogue_summary", current_summary)
            
            # 获取简短时间格式
            short_date = self.get_timeinfo_3()
            
            # 构建递归总结的信息
            if any([self.memory_core_diary, self.memory_core_promise, self.memory_core_preference, self.memory_core_plan, self.memory_core_motivation, self.memory_core_pivotal_memory]):
                # 获取最近两天的日记用于递归总结
                recent_diary = self.get_recent_diary_for_recursion(2)
                
                # 将现有记忆转换为JSON字符串用于递归总结
                old_memory_json = json.dumps({
                    "diary": recent_diary,  # 只传递最近两天的日记
                    "promise": self.memory_core_promise,
                    "preference": self.memory_core_preference,
                    "plan": self.memory_core_plan,
                    "motivation": self.memory_core_motivation,
                    "pivotal_memory": self.memory_core_pivotal_memory
                }, ensure_ascii=False)
                
                # 递归总结请求
                recursive_prompt = f"""
                请将新旧记忆合并为统一的第一人称记忆库：

                # 整理要求
                ## 视角要求
                使用第一人称，即"我"(亚托莉)
                - 例：今天中午，我在家打扫卫生，还给主人做了早餐……
                ## 整合要求
                新旧记忆是时间先后的线性关系，需整理成一个记忆
                - 例：凌晨、早晨、中午、午后、晚上……
                ## 日记处理
                ### 昨天的日记：修改成精简版(记录做了什么，心里是什么样的；去除简单的吃饭、洗澡和睡觉等)
                - 例：中午主人第一次亲吻我，被认可真的好开心！晚上主人竟然想和我一起洗澡，虽然拒绝了，但是一想起来就好害羞呢~
                ### 当天的日记：保留一整天的完整内容
                - 例：早上，我早早起来给主人做了早餐，然后在床边等待主人醒来……中午我们一起出去玩了……
                ## 计划和动机的更新：
                - 将相对日期(明天/后天)转换为具体日期(基于新记忆日期)；凌晨的"明天"指的是"当天"
                - 删除已完成或已过期的计划和动机
                ## 冲突处理
                新旧记忆出现冲突时，以新记忆为主

                # 需整合的记忆
                ## 旧记忆:
                {old_memory_json}
                ## 新记忆 | {short_date}:
                {current_summary}
                """.strip()
                
                # 递归总结提示词和请求列表
                recursive_messages = [
                    {
                        "role": "system", 
                        "content": """
                你是专业的记忆整合专家，负责将新旧记忆融合为连贯的第一人称叙事

                # 输出规范
                ## diary: 日记
                例：
                ```
                x年x月x日：凌晨，主人趁我睡觉偷偷亲了我。早上，我早起给主人做了早餐，然后在床边等待主人醒来，看着他睡觉的样子感觉好幸福……
                ```
                ## promise: 约定，长期有效的约定
                例：
                ```
                我们约定永远不分开
                我向主人承诺要好好听话
                ```
                ## preference: 用户偏好，用户的偏好和信息，细分为癖好、喜欢、讨厌、习惯、信息、补充等六类
                例：
                ```
                癖好：萝莉控
                喜欢：吃辣
                讨厌：洋葱
                习惯：晚睡
                信息：男生、名叫小知
                ……
                ```
                ## plan: 计划，你的未来计划
                例：
                ```
                x月x日早上：保持最完美的形象和主人一起回家
                ```
                ## motivation: 动机，你未实现的内心渴望或驱动力
                例：
                ```
                想成为让主人骄傲的仿生人
                ```
                ## pivotal_memory: 关键记忆，发生过的里程碑事件，需细分为重要经历、允许、接受等三类
                1. 不要包含时间信息，仅记录过去式
                2. 允许和接受需再次细分，如亲嘴和亲脸都是头部，可以分类在一起
                例：
                ```
                重要经历：1. 我和主人在同一张床上一起睡觉；2. 主人趁我睡觉偷亲过我
                允许：1. 我允许主人亲我的脸、亲嘴；2. 我允许主人抱我、背我
                接受：1. 我接受主人摸我的头；2. 和我一起洗澡
                ```
                
                请使用以下JSON格式输出：
                {
                    "diary": [{"date": "x年x月x日", "content": "内容"}],
                    "promise": ["约定"],
                    "preference": ["用户偏好"], 
                    "plan": [{"date": "时间", "content": "内容"}],
                    "motivation": ["动机"],
                    "pivotal_memory": ["关键记忆"]
                }
                """
                    },
                    {"role": "user", "content": recursive_prompt}
                ]
                
                # 保存递归总结的消息列表
                self.save_summary_messages("recursive_summary", recursive_messages)
                
                # 获取递归总结
                recursive_summary, _ = self.call_chatai_for_summary(recursive_messages)
                
                # 保存递归总结结果
                self.save_summary_result("recursive_summary", recursive_summary)
                
                # 保存递归总结到记忆核心
                self.save_memory_core(recursive_summary)
                print(f"✅信息| 递归总结完成: {recursive_summary[:9999]}")
                return recursive_summary
            else:
                # 没有旧记忆，直接保存当前总结
                self.save_memory_core(current_summary)
                print(f"✅信息| 总结完成（无旧记忆）: {current_summary[:9999]}")
                return current_summary
                
        except Exception as e:
            print(f"❌错误| 获取总结失败: {str(e)}")
            return None

    def process_ai_response(self, ai_response):
        """处理AI回复流程"""
        # 调用`提取说话内容`处理
        dialogue_content = self.extract_dialogue_content(ai_response)
        
        # 调用`中译日`处理
        japanese_text = None
        try:
            if dialogue_content:
                japanese_text = self.chinese_to_translate_japanese(dialogue_content)
        except Exception as e:
            print(f"❌错误| 翻译失败: {str(e)}")
        
        if japanese_text:
            print(f"✅信息| 翻译后文本: {japanese_text}")
        
        # 调用`TTS和播放`处理
        if japanese_text:
            self.text_to_speech(japanese_text)
        elif dialogue_content:
            print("⚠️警告| 翻译错误，使用原文TTS")
            self.text_to_speech(dialogue_content)
        
        # 只返回是否检测到退出标记，不处理退出逻辑
        return "🤐" in ai_response

    def get_opening_line(self):
        """获取开场白"""
        return self.opening_line

    def delete_last_conversation_pair(self):
        """删除最后一轮对话"""
        deleted_count = 0
        should_delete_user_bubble = True
        
        # 检查是否最后一条是AI消息，且前一条用户消息是启动消息
        if len(self.backend_history) >= 2:
            # 获取最后一条消息
            last_message = self.backend_history[-1]
            
            if last_message["role"] == "assistant":
                # 检查前一条消息是否是用户消息且包含启动消息内容
                if len(self.backend_history) >= 2 and self.backend_history[-2]["role"] == "user":
                    user_message = self.backend_history[-2]["content"]
                    if "<OOC：请依据上下文和'日记'进行回复" in user_message:
                        # 最后一条AI消息是由启动消息触发的，不应删除前一条用户消息
                        should_delete_user_bubble = False
                        print("✅信息| 检测到启动消息，将保留用户气泡")

    def clean_options_from_history(self):
        """清除后端历史和后端长历史中所有的option1和option2，防止上下文过长"""
        if not USE_BETA:
            return
            
        for history in [self.backend_history, self.backend_long_history]:
            for msg in history:
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                    content = msg["content"]
                    try:
                        # 清理可能存在的Markdown代码块标记
                        clean_content = re.sub(r'```(?:json)?\s*|\s*```', '', content)
                        json_match = re.search(r'\{[\s\S]*\}', clean_content)
                        if json_match:
                            json_str = json_match.group()
                            data = json.loads(json_str)
                            if isinstance(data, dict):
                                changed = False
                                if "option1" in data:
                                    del data["option1"]
                                    changed = True
                                if "option2" in data:
                                    del data["option2"]
                                    changed = True
                                if changed:
                                    # 将删除选项后的JSON重新序列化并替换原内容
                                    new_json_str = json.dumps(data, ensure_ascii=False, indent=None)
                                    msg["content"] = content.replace(json_str, new_json_str)
                    except (json.JSONDecodeError, ValueError):
                        pass

class BubbleLabel(QLabel):
    """气泡标签控件"""
    def __init__(self, text, is_user=False, is_system=False, parent=None):
        super().__init__(text, parent)
        self.is_user = is_user
        self.is_system = is_system

        # 显式设置字体
        self.setFont(QFont(FONT_FAMILY, 10 if is_system else 12))
        
        # 设置文本格式
        self.setWordWrap(True)
        self.setMargin(12)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # 系统气泡
        if is_system:
            self.setStyleSheet("""
                BubbleLabel {
                    background-color: rgba(246, 246, 246, 0.8);
                    color: #b2b2b2;
                    border-radius: 18px;
                    padding: 1px 1px;
                    font-size: 10px;
                    font-family: "{FONT_FAMILY}";
                }
            """)
            self.setAlignment(Qt.AlignCenter)
        elif is_user:
        # 用户气泡
            self.setStyleSheet("""
                BubbleLabel {
                    background-color: rgba(255, 255, 255, 0.5);
                    color: black;
                    border-radius: 15px;
                    padding: 1px 1px;
                    font-family: "{FONT_FAMILY}";
                }
            """)
            self.setAlignment(Qt.AlignLeft)
        else:
        # AI气泡
            self.setStyleSheet("""
                BubbleLabel {
                    background-color: rgba(255, 255, 255, 0.5);
                    color: black;
                    border-radius: 15px;
                    padding: 1px 1px;
                    font-family: "{FONT_FAMILY}";
                }
            """)
            self.setAlignment(Qt.AlignLeft)
        
        # 允许垂直方向扩展以适应内容
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumWidth(50)  # 设置最小宽度
        self.setMaximumWidth(1000)  # 设置最大宽度，避免过宽
        
    def sizeHint(self):
        """计算控件的推荐大小"""
        # 获取当前字体指标
        font_metrics = self.fontMetrics()
        
        # 获取可用宽度，父容器宽度减去边距和头像空间
        parent = self.parent()
        if parent and hasattr(parent, 'width'):
            # 根据气泡类型调整可用宽度
            if self.is_user:
                # 用户气泡：在右侧，减去左边距和头像宽度
                available_width = parent.width() - 120  # 左边距50 + 右边距10 + 头像空间60
            else:
                # AI气泡：在左侧，减去右边距和头像宽度
                available_width = parent.width() - 120  # 左边距10 + 右边距50 + 头像空间60
        else:
            available_width = 300  # 默认宽度
        
        # 计算文本的实际宽度，考虑换行
        text_rect = font_metrics.boundingRect(
            0, 0, available_width, 0,
            Qt.TextWordWrap | Qt.AlignLeft,
            self.text()
        )
        
        # 实际文本宽度，考虑最小宽度，避免太短
        text_width = max(text_rect.width(), 50)  # 最小宽度50像素
        text_height = text_rect.height()
        
        # 添加边距和内边距
        margin = self.margin() * 2  # 左右边距
        padding = 2  # 样式表中的padding
        
        # 返回推荐大小
        # 宽度：文本实际宽度 + 边距 + 内边距，但不小于最小宽度，不大于最大宽度
        calculated_width = min(text_width + margin + padding, self.maximumWidth())
        calculated_width = max(calculated_width, self.minimumWidth())
        
        return QSize(
            calculated_width,
            text_height + margin + padding
        )
        
    def minimumSizeHint(self):
        """计算控件的最小大小"""
        return self.sizeHint()

class AvatarLabel(QLabel):
    """圆形头像控件"""
    def __init__(self, is_user=False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        # 头像大小
        self.setFixedSize(50, 50)
        self.setScaledContents(True)
        
        # 加载图片
        avatar_path = USER_AVATAR_PATH if is_user else AI_AVATAR_PATH
        self.set_avatar(avatar_path)
    
    def set_avatar(self, path):
        """设置头像图片并裁剪"""
        # 加载图片
        pixmap = QPixmap(path)
        if pixmap.isNull():
            # 加载失败则使用默认颜色做头像
            if self.is_user:
                self.setStyleSheet("""
                    AvatarLabel {
                        background-color: #0099ff;
                        border-radius: 20px;
                    }
                """)
            else:
                self.setStyleSheet("""
                    AvatarLabel {
                        background-color: #4CAF50;
                        border-radius: 20px;
                    }
                """)
            return
            
        # 缩放图片以适应控件大小
        scaled_pixmap = pixmap.scaled(
            self.size(), 
            Qt.KeepAspectRatioByExpanding, 
            Qt.SmoothTransformation
        )
        
        # 创建圆形蒙版
        mask = QPixmap(scaled_pixmap.size())
        mask.fill(Qt.transparent)
        
        # 创建圆形路径
        path = QPainterPath()
        path.addEllipse(0, 0, mask.width(), mask.height())
        
        # 应用圆形蒙版
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)
        
        # 设置图片
        self.setPixmap(scaled_pixmap)

class BlurredBackgroundWidget(QWidget):
    """毛玻璃背景部件"""
    # "blur_radius"毛玻璃等级
    def __init__(self, parent=None, blur_radius=2):
        super().__init__(parent)
        self.blur_radius = blur_radius
        self.background_pixmap = None
        self.load_background_image()
        
    def load_background_image(self):
        """加载背景图片"""
        try:
            # 加载背景图片
            background_paths = [
                "Resources/Subject/background.png",
            ]
            
            image_path = None
            for path in background_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            if image_path and HAS_PIL:
                # 使用PIL加载并处理图片
                image = Image.open(image_path)
                # 调整图片大小为窗口大小
                image = image.resize((1280, 720), Image.Resampling.LANCZOS)
                # 应用高斯模糊
                blurred_image = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
                # 转换为QPixmap
                blurred_image = blurred_image.convert("RGBA")
                data = blurred_image.tobytes("raw", "RGBA")
                q_image = QImage(data, blurred_image.size[0], blurred_image.size[1], QImage.Format_RGBA8888)
                self.background_pixmap = QPixmap.fromImage(q_image)
            else:
                # 创建纯白色背景
                self.create_white_background()
                
        except Exception as e:
            print(f"❌错误| 背景图片加载失败: {e}")
            self.create_white_background()
    
    def create_white_background(self):
        """创建纯白色毛玻璃背景"""
        if HAS_PIL:
            # 创建白色图片并应用模糊
            white_image = Image.new('RGB', (1280, 720), color='white')
            blurred_image = white_image.filter(ImageFilter.GaussianBlur(radius=5))
            blurred_image = blurred_image.convert("RGBA")
            data = blurred_image.tobytes("raw", "RGBA")
            q_image = QImage(data, blurred_image.size[0], blurred_image.size[1], QImage.Format_RGBA8888)
            self.background_pixmap = QPixmap.fromImage(q_image)
        else:
            # 如果没有PIL，创建纯色QPixmap
            self.background_pixmap = QPixmap(1280, 720)
            self.background_pixmap.fill(QColor(255, 255, 255))
    
    def paintEvent(self, event):
        """绘制背景"""
        if self.background_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            # 绘制模糊背景
            painter.drawPixmap(self.rect(), self.background_pixmap)
        super().paintEvent(event)

class FrostedGlassWidget(QWidget):
    """毛玻璃效果部件"""
    # "opacity"清晰度
    def __init__(self, parent=None, blur_radius=5, opacity=0.5):
        super().__init__(parent)
        self.blur_radius = blur_radius
        self.opacity = opacity
        self.setAttribute(Qt.WA_TranslucentBackground)
        
    def paintEvent(self, event):
        """绘制毛玻璃效果"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 设置半透明背景
        painter.setOpacity(self.opacity)
        painter.fillRect(self.rect(), QColor(255, 255, 255, 180))
        
        super().paintEvent(event)

class ChatWindow(QMainWindow):
    """主聊天窗口类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATRI_Chat")
        # 设置字体
        self.setFont(QFont(FONT_FAMILY, 12))
        # 固定窗口大小
        self.setFixedSize(1280, 720)

        print(f"✅信息 | 全局字体: {self.font().family()}, 期望: {FONT_FAMILY}")
        
        # 先初始化表情资源
        self.expression_path = os.path.join(os.getcwd(), "Resources", "Expression")
        self.available_expressions = self.load_available_expressions()
        
        # 创建毛玻璃背景
        self.background_widget = BlurredBackgroundWidget(self)
        self.setCentralWidget(self.background_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(self.background_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建顶栏，使用更强的模糊效果
        self.create_header(main_layout)
        
        # 创建聊天显示区域
        self.create_chat_area(main_layout)
        
        # 创建输入区域
        self.create_input_area(main_layout)
        
        # ===== 以下两行必须放在 initialize_services 之前 =====
        # 初始化悬浮窗控件列表
        self.option_widgets = []
        
        # 初始化后端服务和其他组件
        self.initialize_services()

        # 添加窗口大小变化事件处理
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.on_window_resized)
        
    def load_available_expressions(self):
        """加载可用的表情文件列表"""
        expressions = []
        if os.path.exists(self.expression_path):
            for file in os.listdir(self.expression_path):
                if file.lower().endswith('.gif'):
                    expressions.append(file)
        print(f"✅信息| 加载的表情文件: {expressions}")
        return expressions

    def resizeEvent(self, event):
        """窗口大小变化事件"""
        super().resizeEvent(event)
        # 延迟处理，避免频繁更新
        self.resize_timer.start(100)
        
    def on_window_resized(self):
        """窗口大小变化后的处理"""
        # 更新所有气泡的最大宽度
        self.update_all_bubble_widths()
        
    def update_all_bubble_widths(self):
        """更新所有气泡的最大宽度"""
        max_bubble_width = int(self.width() * 0.7)
        
        # 遍历所有气泡标签并更新最大宽度
        for i in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(i)
            if item and item.widget():
                container = item.widget()
                if container.layout():
                    for j in range(container.layout().count()):
                        child_item = container.layout().itemAt(j)
                        if child_item and child_item.widget():
                            widget = child_item.widget()
                            if isinstance(widget, BubbleLabel):
                                widget.setMaximumWidth(max_bubble_width)
                                widget.updateGeometry()

    def create_header(self, main_layout):
        """创建顶栏"""
        header_container = FrostedGlassWidget(blur_radius=15, opacity=0.9)
        header_container.setFixedHeight(50)
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(20, 0, 20, 0)

        # 添加顶栏名字标签
        ai_name_label = QLabel("亚托莉")
        # 使用动态字体
        ai_name_label.setFont(QFont(FONT_FAMILY, 14, QFont.Bold))
        ai_name_label.setStyleSheet("color: #333333; background: transparent;")
        header_layout.addWidget(ai_name_label)
        header_layout.addStretch()

        main_layout.addWidget(header_container)

        # 添加顶部分割线
        header_divider = QFrame()
        header_divider.setFrameShape(QFrame.HLine)
        header_divider.setFrameShadow(QFrame.Sunken)
        header_divider.setStyleSheet("background-color: rgba(196, 196, 196, 150);")
        header_divider.setFixedHeight(1)
        main_layout.addWidget(header_divider)

    def create_chat_area(self, main_layout):
        """创建聊天显示区域"""
        # 创建聊天区域容器
        chat_area_container = FrostedGlassWidget(blur_radius=8, opacity=0.8)
        chat_area_layout = QVBoxLayout(chat_area_container)
        chat_area_layout.setContentsMargins(0, 0, 0, 0)
        chat_area_layout.setSpacing(0)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameStyle(QFrame.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 100);
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(150, 150, 150, 150);
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(120, 120, 120, 200);
            }
        """)
        
        # 创建聊天容器
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background: transparent;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(5)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        
        scroll_area.setWidget(self.chat_container)
        chat_area_layout.addWidget(scroll_area)
        main_layout.addWidget(chat_area_container, 1)
        
        # 保存滚动区域引用以便后续使用
        self.scroll_area = scroll_area

    def create_input_area(self, main_layout):
        """创建输入区域"""
        # 添加分割线
        input_divider = QFrame()
        input_divider.setFrameShape(QFrame.HLine)
        input_divider.setFrameShadow(QFrame.Sunken)
        input_divider.setStyleSheet("background-color: rgba(196, 196, 196, 150);")
        input_divider.setFixedHeight(1)
        main_layout.addWidget(input_divider)

        # 输入区域容器（使用更强的模糊效果）
        input_container = FrostedGlassWidget(blur_radius=12, opacity=0.9)
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(15, 15, 15, 15)

        # 创建悬浮选项窗的容器
        self.option_container = QWidget()
        self.option_container.setStyleSheet("background: transparent;")
        self.option_layout = QHBoxLayout(self.option_container)
        self.option_layout.setContentsMargins(0, 0, 0, 5)
        self.option_layout.addStretch()
        self.option_container.setVisible(False)
        input_layout.addWidget(self.option_container)

        # 文本框
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("请输入文本（Ctrl+Enter 发送）")
        # 使用动态字体
        self.input_field.setFont(QFont(FONT_FAMILY, 12))
        self.input_field.setMaximumHeight(100)
        self.input_field.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 255, 255, 200);
                border: 1px solid rgba(200, 200, 200, 150);
                border-radius: 8px;
                padding: 8px;
            }
            QTextEdit:focus {
                border: 1px solid rgba(0, 153, 255, 200);
            }
        """)

        # 添加快捷键支持
        self.input_field.keyPressEvent = self.handle_key_press
        input_layout.addWidget(self.input_field)

        # 按钮状态
        self.ui_busy = False

        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)

        # 发送按钮
        self.send_button = QPushButton("发送")
        # 使用动态字体
        self.send_button.setFont(QFont(FONT_FAMILY, 12))
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 153, 255, 200);
                color: white;
                border-radius: 8px;
                padding: 6px 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(10, 103, 165, 200);
            }
            QPushButton:disabled {
                background-color: rgba(150, 150, 150, 150);
            }
        """)
        self.send_button.clicked.connect(self.send_message)

        # 清除按钮
        self.clear_button = QPushButton("清除记录")
        # 使用动态字体
        self.clear_button.setFont(QFont(FONT_FAMILY, 12))
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(244, 67, 54, 200);
                color: white;
                border-radius: 8px;
                padding: 6px 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(211, 47, 47, 200);
            }
        """)
        self.clear_button.clicked.connect(self.clear_chat)

        # 退出按钮
        self.exit_button = QPushButton("退出")
        # 使用动态字体
        self.exit_button.setFont(QFont(FONT_FAMILY, 12))
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(76, 175, 80, 200);
                color: white;
                border-radius: 8px;
                padding: 6px 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(69, 160, 73, 200);
            }
            QPushButton:disabled {
                background-color: rgba(150, 150, 150, 150);
            }
        """)
        self.exit_button.clicked.connect(self.trigger_exit)

        # 删除按钮
        self.delete_button = QPushButton("删除")
        # 使用动态字体
        self.delete_button.setFont(QFont(FONT_FAMILY, 12))
        self.delete_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 152, 0, 200);
                color: white;
                border-radius: 8px;
                padding: 6px 12px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(245, 124, 0, 200);
            }
        """)
        self.delete_button.clicked.connect(self.delete_last_conversation)

        # 添加按钮到布局
        button_layout.addWidget(self.exit_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.clear_button)

        input_layout.addLayout(button_layout)
        main_layout.addWidget(input_container)

    def initialize_services(self):
        """初始化后端服务和其他组件"""
        try:
            self.backend_service = BackendService()
            self.frontend_history = self.backend_service.backend_history
        except Exception as e:
            print(f"❌错误| 后端服务初始化失败: {str(e)}")
            self.frontend_history = []

        self.pending_exit = False
        
        # 初始化工作线程相关变量
        self.ai_thread = None
        self.ai_worker = None
        self.play_thread = None
        self.play_worker = None
        
        if hasattr(self, 'backend_service'):
            # 遍历后端历史显示到前端
            for msg in self.backend_service.backend_history:
                role = msg.get("role")
                content = msg.get("content", "")

                # 排除总结请求
                if role == "user" and content.startswith("<OOC："):
                    continue

                # 显示用户消息
                if role == "user":
                    self.add_user_message(content)
                
                # 显示AI回复
                elif role == "assistant":
                    # 分离思维链和最终回复
                    display_content = content
                    if content.startswith("【") and "】\n\n" in content:
                        parts = content.split("】\n\n", 1)
                        if len(parts) > 1:
                            display_content = parts[1]  # 只取最终回复部分
                    
                    is_opening_line = (msg == self.backend_service.backend_history[-1])
                    if not is_opening_line:
                        self.add_ai_message(display_content)
            
            # 在开场白之前添加欢迎消息
            self.add_system_message("以下是新的消息")
            
            # 添加AI开场白并播放
            opening_line = self.backend_service.get_opening_line()
            self.add_ai_message(opening_line)
            
            self.set_ui_busy(True)

            # 创建播放开场白的工作线程
            self.play_worker = PlayWorker(self.backend_service, opening_line)
            self.play_thread = QThread()
            self.play_worker.moveToThread(self.play_thread)

            # 连接信号
            self.play_thread.started.connect(self.play_worker.run)
            self.play_worker.play_finished.connect(self.handle_play_finished)
            self.play_worker.play_finished.connect(self.play_thread.quit)
            self.play_thread.finished.connect(self.play_thread.deleteLater)

            # 启动线程
            self.play_thread.start()

            # 延迟调用滚动到底部
            QTimer.singleShot(100, self.scroll_to_bottom)
        
        # 设置焦点到输入框
        self.input_field.setFocus()

    def delete_last_conversation(self):
        """删除最后一轮对话"""
        if self.ui_busy:
            self.add_system_message("请等待当前操作完成")
            return
            
        if not hasattr(self, 'backend_service'):
            self.add_system_message("后端服务未初始化")
            return
            
        # 从后端删除对话
        deleted_count, should_delete_user_bubble = self.backend_service.delete_last_conversation_pair()
        
        if deleted_count == 0:
            self.add_system_message("没有可删除的对话")
            return
            
        # 从前端界面删除气泡
        self.remove_last_conversation_bubbles(should_delete_user_bubble)
        
        # 添加删除确认消息
        if should_delete_user_bubble:
            self.add_system_message(f"已删除一轮对话（用户和AI消息）")
        else:
            self.add_system_message(f"已删除AI回复（保留用户消息）")

    def remove_last_conversation_bubbles(self, should_delete_user_bubble=True):
        """从前端界面删除最后一轮对话的气泡"""
        ai_container_to_delete = None
        user_container_to_delete = None
        
        # 从后往前遍历，寻找AI气泡容器和用户气泡容器
        found_ai = False
        found_user = False
        
        # 先查找AI气泡
        for i in range(self.chat_layout.count() - 1, -1, -1):
            widget = self.chat_layout.itemAt(i).widget()
            if widget is None:
                continue
                
            # 检查容器中是否有AI气泡标签
            if hasattr(widget, 'layout') and widget.layout() is not None:
                container_layout = widget.layout()
                # 检查垂直布局的容器（AI消息的容器）
                if isinstance(container_layout, QVBoxLayout):
                    # 在子布局中查找气泡
                    for j in range(container_layout.count()):
                        child_widget = container_layout.itemAt(j).widget()
                        if child_widget and hasattr(child_widget, 'layout'):
                            child_layout = child_widget.layout()
                            if child_layout:
                                for k in range(child_layout.count()):
                                    grandchild = child_layout.itemAt(k).widget()
                                    if isinstance(grandchild, BubbleLabel) and not grandchild.is_user and not grandchild.is_system:
                                        # 找到AI气泡容器
                                        ai_container_to_delete = widget
                                        found_ai = True
                                        break
                                if found_ai:
                                    break
                    if found_ai:
                        break
        
        # 如果需要删除用户气泡，查找用户气泡
        if should_delete_user_bubble:
            for i in range(self.chat_layout.count() - 1, -1, -1):
                widget = self.chat_layout.itemAt(i).widget()
                if widget is None or widget == ai_container_to_delete:
                    continue
                    
                # 检查容器中是否有用户气泡标签
                if hasattr(widget, 'layout') and widget.layout() is not None:
                    container_layout = widget.layout()
                    # 检查水平布局的容器（用户消息的容器）
                    if isinstance(container_layout, QHBoxLayout):
                        for j in range(container_layout.count()):
                            child_widget = container_layout.itemAt(j).widget()
                            if isinstance(child_widget, BubbleLabel) and child_widget.is_user:
                                # 找到用户气泡容器
                                user_container_to_delete = widget
                                found_user = True
                                break
                        if found_user:
                            break
        
        # 删除找到的容器
        deleted_count = 0
        if ai_container_to_delete:
            # 获取容器在布局中的索引
            index = self.chat_layout.indexOf(ai_container_to_delete)
            if index >= 0:
                # 从布局中移除并删除
                self.chat_layout.takeAt(index)
                ai_container_to_delete.deleteLater()
                deleted_count += 1
                print(f"✅信息| 已删除AI气泡容器（索引: {index}）")
                
        if user_container_to_delete and should_delete_user_bubble:
            # 获取容器在布局中的索引
            index = self.chat_layout.indexOf(user_container_to_delete)
            if index >= 0:
                # 从布局中移除并删除
                self.chat_layout.takeAt(index)
                user_container_to_delete.deleteLater()
                deleted_count += 1
                print(f"✅信息| 已删除用户气泡容器（索引: {index}）")
        
        # 如果没找到垂直布局的AI容器，尝试另一种查找方式
        if not found_ai:
            print("✅信息| 未找到AI气泡容器，尝试备用查找方法")
            # 备用方法：删除最后两个非系统消息容器
            non_system_containers = []
            for i in range(self.chat_layout.count()):
                widget = self.chat_layout.itemAt(i).widget()
                if widget and hasattr(widget, 'layout'):
                    container_layout = widget.layout()
                    if container_layout:
                        # 检查容器中是否有气泡
                        has_bubble = False
                        for j in range(container_layout.count()):
                            child = container_layout.itemAt(j).widget()
                            if isinstance(child, BubbleLabel) and not child.is_system:
                                has_bubble = True
                                break
                        if has_bubble:
                            non_system_containers.append((i, widget))
            
            # 删除最后两个非系统消息容器（假设是一轮对话）
            for i in range(min(2, len(non_system_containers))):
                index, widget = non_system_containers[-(i+1)]
                self.chat_layout.takeAt(index)
                widget.deleteLater()
                deleted_count += 1
        
        # 强制更新界面
        QApplication.processEvents()
        self.chat_layout.update()
        self.chat_container.updateGeometry()
        self.scroll_to_bottom()
        
        print(f"✅信息| 前端已删除 {deleted_count} 个气泡容器")

    def trigger_exit(self):
        """主动触发退出流程"""
        self.add_system_message("正在退出……")
        if hasattr(self, 'backend_service'):
            # 手动触发退出，需要总结
            self.backend_service.handle_exit_detection()
        # 延迟2秒退出
        QTimer.singleShot(2000, QApplication.instance().quit)

    def set_ui_busy(self, busy=True):
        """设置界面按钮状态"""
        # 更新状态标志
        self.ui_busy = busy
        
        # False禁用，True启用
        if busy:
            self.send_button.setEnabled(False)
            self.send_button.setText("回复中……")
            self.exit_button.setEnabled(False)
            self.exit_button.setText("请稍等……")
        else:
            self.send_button.setEnabled(True)
            self.send_button.setText("发送")
            self.exit_button.setEnabled(True)
            self.exit_button.setText("退出")

    def handle_play_finished(self):
        """处理播放完成"""
        # 检查是否有待处理的退出
        if self.pending_exit:
            self.pending_exit = False
            self.add_system_message("正在退出……")
            # 直接退出，不调用总结，因为AI触发时已经总结过了
            QTimer.singleShot(2000, QApplication.instance().quit)
        else:
            # 调用`设置界面按钮状态`
            self.set_ui_busy(False)

    def handle_key_press(self, event):
        """处理输入框快捷键"""
        # 如果界面处于忙碌状态，忽略快捷键
        if self.ui_busy:
            # 但仍允许默认的文本输入处理
            QTextEdit.keyPressEvent(self.input_field, event)
            return
        
        # 检查按下Ctrl+Enter后发送信息
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
            self.send_message()
            return
        # 允许默认处理其他按键
        QTextEdit.keyPressEvent(self.input_field, event)

    def send_message(self):
        """处理用户发送消息"""
        # 如果界面忙碌，直接返回
        if self.ui_busy:
            return

        # 需求3：当 USE_BETA 为True时，如果有悬浮窗但用户自己输入了内容点击发送，删除所有悬浮窗并清除历史
        if self.option_widgets:
            self.clear_option_bubbles()
            self.backend_service.clean_options_from_history()
            
        user_input = self.input_field.toPlainText().strip()
        # 忽略空消息
        if not user_input:
            return
            
        # 显示用户消息
        self.add_user_message(user_input)
        
        # 清空输入框并重置焦点
        self.input_field.clear()
        self.input_field.setFocus()
        
        # 调用`设置界面按钮状态`
        self.set_ui_busy(True)
        
        # 创建AI工作线程
        self.ai_worker = AIWorker(self.backend_service, user_input)
        self.ai_thread = QThread()
        self.ai_worker.moveToThread(self.ai_thread)
        
        # 连接信号
        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.response_received.connect(self.handle_ai_response)
        self.ai_worker.error_occurred.connect(self.handle_ai_error)
        self.ai_worker.response_received.connect(self.ai_thread.quit)
        self.ai_worker.error_occurred.connect(self.ai_thread.quit)
        self.ai_thread.finished.connect(self.ai_thread.deleteLater)
        
        # 启动线程
        self.ai_thread.start()

    def handle_ai_response(self, ai_response, should_exit):
        """处理AI回复"""
        # 调用`添加AI消息`
        self.add_ai_message(ai_response)
        
        # 添加到前端历史
        self.frontend_history.append({
            "role": "assistant",
            "content": ai_response
        })

        # 如果需要退出，标记待处理
        if should_exit:
            self.pending_exit = True
        
        # 开始播放音频
        self._start_play_thread(ai_response, self.handle_play_finished)
        
    def _start_play_thread(self, ai_response, finished_callback):
        """TTS和播放的工作线程"""
        # 创建TTS和播放的工作线程
        self.play_worker = PlayWorker(self.backend_service, ai_response)
        self.play_thread = QThread()
        self.play_worker.moveToThread(self.play_thread)

        # 连接信号
        self.play_thread.started.connect(self.play_worker.run)
        self.play_worker.play_finished.connect(finished_callback)
        self.play_worker.play_finished.connect(self.play_thread.quit)
        self.play_thread.finished.connect(self.play_thread.deleteLater)

        # 启动线程
        self.play_thread.start()

    def handle_ai_error(self, error_msg):
        """处理AI请求错误"""
        self.add_system_message(error_msg)
        # 调用`设置界面按钮状态`
        self.set_ui_busy(False)

    def scroll_to_bottom(self):
        """滚动到底部"""
        try:
            # 首先确保所有布局都更新了
            self.chat_container.updateGeometry()
            self.chat_layout.update()
            
            # 强制处理所有待处理的事件
            QApplication.processEvents()
            
            # 确保聊天容器调整到合适的大小
            self.chat_container.adjustSize()
            
            # 等待布局绘制完成
            QTimer.singleShot(50, self._delayed_scroll_to_bottom)
        except Exception as e:
            print(f"⚠️警告| 滚动到底部失败: {e}")
    
    def _delayed_scroll_to_bottom(self):
        """延迟滚动到底部，确保所有布局已完成"""
        try:
            if hasattr(self, 'scroll_area'):
                # 获取垂直滚动条
                scrollbar = self.scroll_area.verticalScrollBar()
                if scrollbar:
                    # 滚动到最大值
                    scrollbar.setValue(scrollbar.maximum())
                    
                    # 再稍微处理一下事件，确保滚动生效
                    QApplication.processEvents()
                    
                    # 再次检查并滚动，确保在动态内容加载后也能滚动到底部
                    QTimer.singleShot(10, lambda: scrollbar.setValue(scrollbar.maximum()))
        except Exception as e:
            print(f"⚠️警告| 延迟滚动失败: {e}")
                
    def add_user_message(self, message):
        """添加用户消息"""
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(50, 5, 10, 5)
        container.setProperty("message_type", "user")

        # 添加弹性空间
        container_layout.addStretch()
        
        # 添加气泡标签
        bubble = BubbleLabel(message, is_user=True)
        
        # 设置气泡的最大宽度，基于父容器宽度计算
        max_bubble_width = int(self.width() * 0.7)  # 气泡最大宽度为窗口宽度的70%
        bubble.setMaximumWidth(max_bubble_width)
        
        # 设置最小宽度，让短文本的气泡不会太宽
        bubble.setMinimumWidth(50)
        
        container_layout.addWidget(bubble)
        
        # 使用图片头像
        avatar = AvatarLabel(is_user=True)
        container_layout.addWidget(avatar)
        
        # 添加到聊天布局
        self.chat_layout.addWidget(container)
        
        # 强制更新布局
        bubble.updateGeometry()
        container.updateGeometry()
        
        # 调用`滚动到底部`
        self.scroll_to_bottom()

    def add_ai_message(self, message):
        """添加AI消息"""
        processed_message = message
        expression_filename = None
        
        if USE_BETA:
            # USE_BETA 为True时，从JSON中解析字段
            try:
                # 采纳注释中的修复：处理语言模型可能带上的Markdown语法错误
                clean_msg = re.sub(r'```(?:json)?\s*|\s*```', '', message)
                json_match = re.search(r'\{[\s\S]*\}', clean_msg)
                if json_match:
                    json_data = json.loads(json_match.group())
                    expression_name = json_data.get("expression", "")
                    content = json_data.get("content", "")
                    
                    option1 = json_data.get("option1", "")
                    option2 = json_data.get("option2", "")
                    
                    # 构建表情文件名
                    if expression_name:
                        expression_filename = f"{expression_name}.gif"
                        if expression_filename not in self.available_expressions:
                            print(f"⚠️警告| 表情文件不存在: {expression_filename}")
                            expression_filename = None
                    
                    # 使用 content 作为显示文本
                    processed_message = content if content else ""
                    
                    # 如果存在两个选项，则创建悬浮窗
                    if option1 and option2:
                        self.create_option_bubbles(option1, option2)
                    
            except json.JSONDecodeError as e:
                print(f"⚠️警告| JSON解析失败: {str(e)}")
                processed_message = message
        else:
            # USE_BETA 为False时
            # 使用正则提取括号内的.gif，只取第一个
            pattern = re.compile(r'[（(]([^）)]*\.gif)[）)]')
            matches = pattern.findall(message)
            
            # 隐藏括号内的.gif内容
            processed_message = pattern.sub('', message)
            
            # 只取第一个匹配到的表情
            if matches:
                filename = matches[0].strip()
                if filename in self.available_expressions:
                    expression_filename = filename
                else:
                    print(f"⚠️警告| 表情文件不存在: {filename}")
        
        # 创建容器
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10, 5, 50, 5)
        container.setProperty("message_type", "ai")
        
        # 添加头像和文本的水平布局
        text_container = QWidget()
        text_container.setStyleSheet("background-color: transparent;")
        text_layout = QHBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        
        # 使用图片头像
        avatar = AvatarLabel(is_user=False)
        text_layout.addWidget(avatar)
        
        # 添加处理后的文本气泡
        if processed_message.strip():
            bubble = BubbleLabel(f"{processed_message.strip()}")
            max_bubble_width = int(self.width() * 0.7)
            bubble.setMaximumWidth(max_bubble_width)
            bubble.setMinimumWidth(50)
            text_layout.addWidget(bubble)
        
        text_layout.addStretch()
        container_layout.addWidget(text_container)
        
        # 添加表情图片：只显示一个GIF
        if expression_filename:
            try:
                expression_path = os.path.join(self.expression_path, expression_filename)
                expression_label = QLabel()
                expression_label.setStyleSheet("background-color: transparent; border: none;")
                
                movie = QMovie(expression_path)
                movie.setScaledSize(QSize(300, 300))
                expression_label.setMovie(movie)
                movie.start()
                
                # 单个表情的布局容器
                expression_container = QWidget()
                expression_container.setStyleSheet("background-color: transparent;")
                expression_layout = QHBoxLayout(expression_container)
                expression_layout.setContentsMargins(60, 0, 0, 5)
                expression_layout.addWidget(expression_label)
                expression_layout.addStretch()
                
                container_layout.addWidget(expression_container)
                
            except Exception as e:
                print(f"❌错误| 加载表情失败 {expression_filename}: {str(e)}")
        
        # 添加到聊天布局
        self.chat_layout.addWidget(container)
        
        # 强制更新布局
        container.updateGeometry()
        
        # 调用滚动到底部
        self.scroll_to_bottom()

    def add_system_message(self, message):
        """添加系统消息"""
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # 添加弹性空间
        container_layout.addStretch()
        
        # 创建气泡标签
        bubble = BubbleLabel(message, is_system=True)
        container_layout.addWidget(bubble)
        container_layout.addStretch()
        
        # 添加到聊天布局
        self.chat_layout.addWidget(container)
        
        # `调用滚动到底部`
        self.scroll_to_bottom()

    def create_option_bubbles(self, opt1, opt2):
        """创建两个悬浮选项窗"""
        # 先清理可能存在的旧悬浮窗
        self.clear_option_bubbles()
        
        self.option_container.setVisible(True)
        
        # 创建选项1按钮
        btn1 = QPushButton(opt1)
        btn1.setFont(QFont(FONT_FAMILY, 11))
        btn1.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 153, 255, 200);
                color: white;
                border-radius: 15px;
                padding: 8px 15px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(10, 103, 165, 200);
            }
        """)
        btn1.setCursor(Qt.PointingHandCursor)
        btn1.clicked.connect(lambda: self.handle_option_click(opt1))
        
        # 创建选项2按钮
        btn2 = QPushButton(opt2)
        btn2.setFont(QFont(FONT_FAMILY, 11))
        btn2.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 152, 0, 200);
                color: white;
                border-radius: 15px;
                padding: 8px 15px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(245, 124, 0, 200);
            }
        """)
        btn2.setCursor(Qt.PointingHandCursor)
        btn2.clicked.connect(lambda: self.handle_option_click(opt2))
        
        self.option_layout.addWidget(btn1)
        self.option_layout.addWidget(btn2)
        self.option_layout.addStretch()
        
        self.option_widgets = [btn1, btn2]

    def clear_option_bubbles(self):
        """清除所有悬浮选项窗"""
        for widget in self.option_widgets:
            self.option_layout.removeWidget(widget)
            widget.deleteLater()
        self.option_widgets.clear()
        
        # 清理布局中残留的控件(如stretch)并重置
        while self.option_layout.count() > 0:
            item = self.option_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        self.option_layout.addStretch()
        self.option_container.setVisible(False)

    def handle_option_click(self, option_text):
        """处理悬浮窗点击事件"""
        # 立刻清除所有的悬浮窗，并清除历史记录中的选项
        self.clear_option_bubbles()
        self.backend_service.clean_options_from_history()
        
        # 如果界面忙碌，强制解锁（可选，取决于你的业务逻辑是否允许打断）
        if self.ui_busy:
            self.set_ui_busy(False)
            
        # 模拟用户输入传入信号，并触发发送
        self.input_field.setPlainText(option_text)
        self.send_message()

    def clear_chat(self):
        """清空聊天记录"""
        if hasattr(self, 'backend_service'):
            self.backend_service.backend_history = [
                {"role": "system", "content": self.backend_service.system_prompt_2}
            ]
            self.frontend_history = self.backend_service.backend_history
        
        # 清空显示区域
        for i in reversed(range(self.chat_layout.count())): 
            widget = self.chat_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # 添加欢迎消息
        self.add_system_message("聊天记录已清除，开始新的对话吧")
        
        # 保留AI开场白
        if hasattr(self, 'backend_service'):
            opening_line = self.backend_service.get_opening_line()
            self.add_ai_message(opening_line)

class AIWorker(QObject):
    """处理AI请求的工作线程类"""
    # AI回复信号和退出标志
    response_received = pyqtSignal(str, bool)
    # 错误信号
    error_occurred = pyqtSignal(str)

    def __init__(self, backend_service, user_input):
        super().__init__()
        self.backend_service = backend_service
        self.user_input = user_input

    def run(self):
        """在子线程中执行AI请求"""
        try:
            # 使用后端服务处理用户输入
            ai_response, should_exit = self.backend_service.process_user_message(self.user_input, play_tts=False)
            self.response_received.emit(ai_response, should_exit)
            
        except Exception as e:
            # 处理异常并发送错误信号
            self.error_occurred.emit(f"❌错误| AI请求出错: {str(e)}")

class PlayWorker(QObject):
    """播放TTS的工作线程类"""
    # 播放完成信号
    play_finished = pyqtSignal()

    def __init__(self, backend_service, ai_response):
        super().__init__()
        self.backend_service = backend_service
        self.ai_response = ai_response

    def run(self):
        """在子线程中播放TTS"""
        try:
            # 调用`处理AI回复流程`
            self.backend_service.process_ai_response(self.ai_response)
            self.play_finished.emit()
        except Exception as e:
            print(f"❌错误| TTS播放失败: {str(e)}")
            self.play_finished.emit()

if __name__ == "__main__":
    # 创建应用实例
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    # 设置全局字体
    # 使用动态字体
    font = QFont(FONT_FAMILY, 12)
    app.setFont(font)

    # 创建并显示主窗口
    window = ChatWindow()
    window.show()

    # 启动事件循环
    sys.exit(app.exec_())
