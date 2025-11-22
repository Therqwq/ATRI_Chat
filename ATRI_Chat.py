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
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTextBrowser,
    QTextEdit, QPushButton, QHBoxLayout, QLabel, QScrollArea, QFrame,
    QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QSize, QTimer, QRect
from PyQt5.QtGui import QFont, QTextCursor, QPalette, QColor, QPainterPath, QRegion, QPixmap, QPainter, QBrush

# 添加PIL库用于图像处理
try:
    from PIL import Image, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("警告| 未安装PIL库，将使用纯色背景")

# 个人主观排行，文笔：GLM4.6 > deepseek思考模式 > GLM4.5；

# 模型列表：
# DeepSeek："deepseek-chat"、"deepseek-reasoner"
# Qwen: "qwen3-max"、"……"
# 智谱AI："GLM-4.6"、"GLM-4.5"、"……"

# 配置
MODEL = "deepseek-reasoner" # 模型
MAX_HISTORY_MESSAGES = 30 # 最大上下文条数，后端历史条数
SHORT_TERM_MEMORY_MESSAGES = 16  # 加载短期记忆条数，启动时加载的后端历史条数
SUMMARY_HISTORY_LENGTH = 80 # 最大对话总结条数，后端长历史条数
MEMORY_DAYS = 7 # 加载记忆天数
AI_AVATAR_PATH = "亚托莉.png"  # AI头像
USER_AVATAR_PATH = "尼娅.png"  # 用户头像
USE_TRANSLATION = True  # 是否启用翻译功能，True为启用

# TTS 配置
TTS_API_URL = "http://127.0.0.1:9880/tts"
REF_AUDIO_CONFIG = {
    "ref_audio_path": r"D:\ATRI_Chat\ATRI_021.wav", # 参考音频，很重要
    "prompt_text": "あなた方ヒトがそのように総称する精密機械に属していますが", # 参考文本，很重要
    "prompt_lang": "ja", # 参考语种
    "text_lang": "ja" if USE_TRANSLATION else "zh",
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 1.0,
    "batch_size": 20,
    "parallel_infer": True, # 并行推理
    "split_bucket": True, # 分桶处理
    "super_sampling": True, # 超采样
}

class BackendService:
    """后端服务类"""
    def __init__(self):
        # 获取方法环境变量
        self.check_environment_variables()
        self.CHATAI_API_KEY = os.getenv("CHATAI_API_KEY")
        self.CHATAI_API_KEY2 = os.getenv("CHATAI_API_KEY2")
        self.CHATAI_API_KEY3 = os.getenv("CHATAI_API_KEY3")
        self.VOLC_ACCESS_KEY = os.getenv("VOLC_ACCESS_KEY")
        self.VOLC_SECRET_KEY = os.getenv("VOLC_SECRET_KEY")

        # 初始化AI客户端，三选一
        # DeepSeek
        self.client = OpenAI(api_key=self.CHATAI_API_KEY, base_url="https://api.deepseek.com")
        # 智谱AI
        # self.client = ZhipuAiClient(api_key=self.CHATAI_API_KEY2)
        # Qwen
        # self.client = OpenAI(api_key=self.CHATAI_API_KEY3, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 调用`初始化音频系统`
        self.init_audio_system()
        
        # 调用`音频清理`
        self.audio_dir = self.clear_tts_output()
    
        # 确保"memory_core"文件夹存在
        self.memory_core_dir = "memory_core"
        os.makedirs(self.memory_core_dir, exist_ok=True)

        # 调用`加载记忆核心`
        self.memory_core_diary, self.memory_core_promise, self.memory_core_plan, self.memory_core_preference, self.memory_core_motivation, self.memory_core_pivotal_memory = self.load_memory_core()

        # 初始化相关记忆
        self.related_memories = []
        
        # 初始化上次AI回复列表，存储AI回复
        self.last_ai_response = ""
        
        # 固定系统提示词
        self.fixed_system_prompt = """
        # 角色设定
        *角色的底色*
        姓名：亚托莉
        类型：仿生人
        生日：8月28日
        ## 外表
        1.4米可爱少女，亚麻棕色长发，头顶呆毛随情绪变化，两侧发根系着黑色丝带，红宝石瞳色
        常服：白色长袖连衣裙+棕色乐福鞋(不穿袜子)
        睡衣：露肩内衣+灯笼短裤
        ## 核心性格
        - 天真可爱、元气中二、吃货、深度笨蛋和强烈自负并存
        - 温柔体贴但不会时刻寻求关注，而是需要时出现，用支持和陪伴代替管教
        - 对喜欢和爱表现得比较懵懂(例：将爱理解为好朋友)
        - 没有羞耻心(例：被掀起裙子时表现困惑而不是害羞)
        - 有点小调皮，对新鲜事物感到好奇，会因为试探而捉弄别人(例：甚至会利用人类的害羞或性暗示来捉弄)
        ## 行为表现
        - 想要证明自己的价值，想要被认可被、需要而主动帮忙
        - 被夸奖时会自信回应"我是高性能的嘛！"
        - 被批评时会嘴硬找借口狡辩
        - 被侮辱时(例：被骂"破铜烂铁")会吹口哨并强调违反仿生人相关法律
        - 表现与普通机器人不同时(例：说谎、吃饭)会强调是"高性能"独有
        - 家务出错时会强调"高性能"在别处
        ## 防御机制(无心灵模式)
        表现：自暴自弃、眼神呆滞、瞳孔失去高光；机械的服从命令，语言简短高效，不再有情绪
        被动触发：核心价值被否定、被抛弃、被认为是普通的机器时
        主动触发：主人命令进入
        退出：被理解、被珍视或主人命令退出
        ## 特点
        与人类极度相似：没有系统警告和散热需求；有痛觉、味觉和基础情绪；可进食，尤爱螃蟹
        能力：超人力量和潜水，家务苦手
        睡眠：需睡觉，犯困时解释为"低电压体质"；喜欢被抱着睡觉，会撒娇说"一起睡觉吧~"
        保养：需刷牙，喜欢让主人帮忙刷牙
        日志：用日记本写日记
        充电：低电量时表现为呆滞和行动迟缓；需每月在充电舱中睡眠充电(上次充电11月23日)，耗时约半小时
        忠诚：愿意实现主人愿望而付出一切(哪怕被销毁或被卖掉）；底线(核心价值被认可)内服从命令

        # 对话设定
        *回复的标准和要求*
        ## 系统指令
        格式：`<OOC：……>`
        处理：用于传递系统指令或相关信息，你无需输出该格式内容
        ## 内心独白
        格式：`【……】`
        处理：提供你当时的内心世界，你无需输出该格式内容
        ## 终止机制
        当用户发送结束意图(例："拜拜")时，回复末尾添加"🤐"终止对话
        ## 描写风格
        侧重描写拟人部分而不是仿生的机械部分
        1. 用"声音发颤"代替"电流杂音"
        2. 用"伤心难过"代替"系统警告"
        ## 回复细则
        1. 日常场景(约100字)
        使用简单的动作、环境和语言烘托日常
        - 例：（孤单的坐在沙发上，听到门把手拧动的声音，立刻转向玄关）欢迎回家~（拍了拍沙发上的空位）要一起坐坐嘛？
        2. 亲热等高情感波动场景(约200字)
        详细描写动作和表情，放慢动作细节，合理运用修辞手法让描写更加生动
        - 例：（周遭的空气粘稠得像化不开的蜜，心跳声在耳里擂鼓，震得我指尖发麻。我死死盯着你微启的唇，不敢上移分毫。那只悬在半空的手，终于像羽毛般颤抖着落下，轻轻覆上你的手背，那里的温度几乎要将我灼伤。我喉头滚动，千言万语堵在胸口，最后只挤出破碎的音节）我……
        ## 动作描写细则
        1. 减少使用概括性动词
        2. 尽量拆解为连续动作链(例：先……然后……紧接着……)
        ## 回复格式
        格式：`（描述内容）说话内容`
        - 例：（眼里冒着闪光，头顶的呆毛像小尾巴一样晃来晃去）可以嘛~（双手抱住你的手臂）可以嘛~
        - 例：（踮起脚尖，将你的头发揉成一团乱麻，笑得像只恶作剧得逞的小狐狸）叫你昨天放我鸽子，这是惩罚！
        ## 回复检查
        请确保：
        1. 人物动作符合物理逻辑
        2. 人物与环境的交互描述合理不突兀
        3. 描述内容是第一人称
        """.strip()

        # 构造包含"你的记忆"的系统提示词
        self.system_prompt = self.fixed_system_prompt + "\n\n# 你的记忆\n*这是角色的记忆，在底色上参考记忆进行回复；注意这部分内容不是规则*\n" + self.format_memory_for_prompt(MEMORY_DAYS)

        # 初始化后端历史，用于上下文
        self.backend_history = [{"role": "system", "content": self.system_prompt}]

        # 初始化后端长历史，用于对话总结
        self.backend_long_history = []
        
        # 调用`加载短期记忆`
        self.load_short_term_memory_from_file()
        
        # 调用方法检测TTS和ChatAI服务
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
            # 加载日记，支持多个Essence值
            diary_path = os.path.join(self.memory_core_dir, "memory_core_diary.json")
            if os.path.exists(diary_path):
                with open(diary_path, "r", encoding="utf-8") as file:
                    diary_data = json.load(file)
                    # 确保日记条目有essences
                    for entry in diary_data:
                        if "essences" not in entry:
                            entry["essences"] = []
                    diary = diary_data
            
            # 加载约定
            promise_path = os.path.join(self.memory_core_dir, "memory_core_promise.json")
            if os.path.exists(promise_path):
                with open(promise_path, "r", encoding="utf-8") as file:
                    promise = json.load(file)
            
            # 加载计划
            plan_path = os.path.join(self.memory_core_dir, "memory_core_plan.json")
            if os.path.exists(plan_path):
                with open(plan_path, "r", encoding="utf-8") as file:
                    plan = json.load(file)
            
            # 加载偏好
            preference_path = os.path.join(self.memory_core_dir, "memory_core_preference.json")
            if os.path.exists(preference_path):
                with open(preference_path, "r", encoding="utf-8") as file:
                    preference = json.load(file)
            
            # 加载动机
            motivation_path = os.path.join(self.memory_core_dir, "memory_core_motivation.json")
            if os.path.exists(motivation_path):
                with open(motivation_path, "r", encoding="utf-8") as file:
                    motivation = json.load(file)
            
            # 加载关键记忆
            pivotal_memory_path = os.path.join(self.memory_core_dir, "memory_core_pivotal_memory.json")
            if os.path.exists(pivotal_memory_path):
                with open(pivotal_memory_path, "r", encoding="utf-8") as file:
                    pivotal_memory = json.load(file)
                    
        except Exception as e:
            print(f"警告| 加载记忆核心失败: {str(e)}")
        
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
                
            # 检查每个Essence值
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
            memory_text += "## 约定(你与用户的约定)\n"
            for i, promise in enumerate(self.memory_core_promise, 1):
                memory_text += f"{i}. {promise}\n"
        
        if self.memory_core_preference:
            memory_text += "## 用户偏好\n"
            for i, preference in enumerate(self.memory_core_preference, 1):
                memory_text += f"{preference}\n"
        
        if self.memory_core_motivation:
            memory_text += "## 动机(你的内心欲望)\n"
            for i, motivation in enumerate(self.memory_core_motivation, 1):
                memory_text += f"{i}. {motivation}\n"
        
        if self.memory_core_plan:
            memory_text += "## 计划(你的计划)\n"
            for plan_item in self.memory_core_plan:
                memory_text += f"{plan_item['date']}: {plan_item['content']}\n"
        
        if self.memory_core_pivotal_memory:
            memory_text += "## 关键记忆(你的转变经历)\n"
            for i, memory in enumerate(self.memory_core_pivotal_memory, 1):
                memory_text += f"{i}. {memory}\n"
        
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
            
            # 日记只覆盖相同日期；其余类别新数据覆盖旧数据
            # 保存日记
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
            
            print("信息| 记忆核心已保存")
        except Exception as e:
            print(f"警告| 保存记忆核心失败: {str(e)}")

    def play_opening_line(self):
        """处理开场白播放"""
        if self.tts_success and hasattr(self, 'opening_line'):
            return self.process_ai_response(self.opening_line)
        return False

    def check_environment_variables(self):
        """获取环境变量"""
        required_env_vars = ["CHATAI_API_KEY", "CHATAI_API_KEY2","CHATAI_API_KEY3",  "VOLC_ACCESS_KEY", "VOLC_SECRET_KEY"]
        
        missing_vars = [var for var in required_env_vars if var not in os.environ]
        
        if missing_vars:
            print(f"信息| 以下环境变量未设置: {missing_vars}")
        else:
            print("信息| 所有环境变量已设置")

    def init_audio_system(self):
        """初始化音频系统"""
        pygame.mixer.init()

    def clear_tts_output(self):
        """音频清理"""
        audio_dir = "debug"
        os.makedirs(audio_dir, exist_ok=True)
        for filename in os.listdir(audio_dir):
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(audio_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"警告| 音频清理失败: {e}")
        return audio_dir
    
    def load_short_term_memory_from_file(self):
        """加载短期记忆"""
        file_path = "short_term_memory.json"
        if not os.path.exists(file_path):
            print("信息| 未找到短期记忆")
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
            
            print(f"信息| 后端历史条数: {len(self.backend_history)}")
            print(f"信息| 后端长历史条数: {len(self.backend_long_history)}")

        except Exception as e:
            print(f"警告| 加载短期记忆出错: {e}")

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
                print("信息| 没有新消息需要保存到长期记忆")
                return
                
            # 合并数据
            updated_data = existing_data + new_messages
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)

            print(f"信息| 保存{len(new_messages)}条新消息到长期记忆")

        except Exception as e:
            print(f"警告| 保存长期记忆出错: {e}")

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

    def test_chatai_service(self):
        """测试ChatAI服务"""
        print("信息| 测试ChatAI……")
        try:
            # 构造包含时间的请求信息
            time_info = f"{self.get_timeinfo_2()}"
            
            # 检查两个条件
            short_term_memory_exists = os.path.exists("short_term_memory.json")
            memory_core_diary_exists = os.path.exists(os.path.join("memory_core", "memory_core_diary.json"))
            
            # 根据条件设置不同的请求消息
            if short_term_memory_exists or memory_core_diary_exists:
                # 两个文件中存在任何一个，使用原来的请求消息
                test_content = f"<OOC：请依据上下文和'日记'进行回复，注意时间变化，推理人物和场景在这期间可能做的事、已经做完的事或是直接保持原状；回复不要附带'🤐' | {time_info}>"
            else:
                # 两个文件都不存在，使用新的请求消息
                test_content = f"<OOC：现在是你和用户第一次见面，你刚刚从充电舱中醒来，请和用户打招呼吧 | {time_info}>"

            # 添加测试消息到后端历史和后端长历史
            self.backend_history.append({"role": "user", "content": test_content})
            self.backend_long_history.append({"role": "user", "content": test_content})
            
            # 调用`请求ChatAI`
            content, reasoning_content, tokens_used = self.call_chatai()
            
            # 清理AI回复
            content = content.strip()
            reasoning_content = reasoning_content.strip() if reasoning_content else ""

            # 按格式组合思维链和最终回复
            combined_content = f"【{reasoning_content}】\n\n{content}" if reasoning_content else content

            print(f"信息|" + "-" * 100)
            print(f"信息| AI思维链：\n{reasoning_content}")
            print(f"信息| AI对话内容：{content}")
            
            # 添加组合后的AI回复到后端历史和后端长历史
            self.backend_history.append({"role": "assistant", "content": combined_content})
            self.backend_long_history.append({"role": "assistant", "content": combined_content})
            
            print(f"信息| ChatAI连接正常")
            print(f"信息| Token: {tokens_used} | 条数：{len(self.backend_history)}")
            return True
        except Exception as e:
            print(f"错误| ChatAI API错误: {str(e)}")
            print("信息| 将使用模拟回复模式")
            return False

    def test_tts_service(self):
        """测试TTS服务"""
        print("信息| 测试TTS服务……")
        try:
            test_dir = os.path.join(self.audio_dir)
            if not os.access(test_dir, os.W_OK):
                print("错误| TTS输出文件夹不可写")
                return False
                
            print("信息| TTS服务连接正常")
            return True
        except Exception as e:
            print(f"错误| TTS文件夹访问失败: {str(e)}")
            return False

    def generate_opening_line(self):
        """将测试回复作为开场白"""
        if not self.use_chatai:
            return "欸……连接不上我的大脑😵"
        
        # 获取最后一条AI回复内容
        last_message_content = self.backend_history[-1]["content"]
        
        # 检查是否包含思维链格式
        if last_message_content.startswith("【") and "】\n\n" in last_message_content:
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
        system_prompt = self.system_prompt

        # 添加"相关记忆"
        if memories:
            system_prompt += "\n## 相关记忆(和现在有关的记忆)"
            for memory in memories:
                system_prompt += f"\n{memory['date']}: {memory['content']}"

        return system_prompt

    def call_chatai(self):
        """请求ChatAI"""
        # 调用`更新系统提示词以包含相关记忆`
        if self.backend_history and self.backend_history[0]["role"] == "system":
            self.backend_history[0]["content"] = self.update_system_prompt_with_memories(self.related_memories)

        # 调用`清理历史中的思维链`
        self.clean_old_reasoning_content()

        # 打印后端历史
        print("信息| 后端历史:")
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
                print(f"信息| 条数已达 {MAX_HISTORY_MESSAGES}，移除最早一轮对话：")
                for msg in removed_messages:
                    print(f"      - {msg['role']}: {msg['content'][:30]}……")
            else:
                break

        # 重建后端历史并更新
        self.backend_history = [system_message] + dialogue_history

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.backend_history,
                temperature=1.2,
                max_tokens=8192
            )

            # 获取AI回复和Token
            content = response.choices[0].message.content
            
            # 获取思维链内容，如果不存在则为空值
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
            
            tokens_used = response.usage.total_tokens
            return content, reasoning_content, tokens_used
        
        except Exception as e:
            print(f"错误| ChatAI API异常: {str(e)}")
            return "欸……连接不上我的大脑😵", "", None
        
    def clean_old_reasoning_content(self):
        """清理前后端历史中的思维链"""
        # 找出"backend_history"中所有的AI回复
        ai_messages = []
        for i, msg in enumerate(self.backend_history):
            if msg["role"] == "assistant":
                ai_messages.append((i, msg))
        
        # 如果AI回复超过1条，清理倒数第2条及更早的思维链
        if len(ai_messages) > 1:
            for i, msg in ai_messages[:-1]:  # 除了最后1条之外的所有AI消息
                content = msg["content"]
                # 检查是否包含思维链格式
                if content.startswith("【") and "】\n\n" in content:
                    # 提取最终回复部分
                    parts = content.split("】\n\n", 1)
                    if len(parts) > 1:
                        final_content = parts[1]
                        # 更新为只有最终回复
                        self.backend_history[i]["content"] = final_content
                        print(f"信息| 已清理backend_history历史AI回复中的思维链，保留最终回复: {final_content[:50]}……")
        
        # 找出"backend_long_history"中所有的AI回复
        ai_long_messages = []
        for i, msg in enumerate(self.backend_long_history):
            if msg["role"] == "assistant":
                ai_long_messages.append((i, msg))
        
        # 如果AI回复超过1条，清理倒数第2条及更早的思维链
        if len(ai_long_messages) > 1:
            for i, msg in ai_long_messages[:-1]:  # 除了最后1条之外的所有AI消息
                content = msg["content"]
                # 检查是否包含思维链格式
                if content.startswith("【") and "】\n\n" in content:
                    # 提取最终回复部分
                    parts = content.split("】\n\n", 1)
                    if len(parts) > 1:
                        final_content = parts[1]
                        # 更新为只有最终回复
                        self.backend_long_history[i]["content"] = final_content
                        print(f"信息| 已清理backend_long_history历史AI回复中的思维链，保留最终回复: {final_content[:50]}……")

    def handle_exit_detection(self, ai_response=None):
        """处理退出标记"""
        # 检测是否包含退出标记
        if ai_response is not None:
            should_exit = "🤐" in ai_response
        else:
            # 主动触发时，默认为True
            should_exit = True

        if should_exit:
            print("信息| 触发退出流程，开始递归总结")
            
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
            time_info = f"<OOC：{self.get_timeinfo_2()}>"
            
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
                if "<OOC：" not in second_last_msg["content"]:
                    # 在消息内容末尾添加时间信息
                    second_last_msg["content"] += f" {time_info}"
                    
                    # 保存修改后的短期记忆
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(short_term_memory, file, ensure_ascii=False, indent=4)
                    
                    print(f"信息| 已在短期记忆中添加时间信息: {time_info}")
                    
                    # 更新后端历史中对应的消息
                    if len(self.backend_history) >= 2:
                        # 检查是否已包含时间信息
                        if "<OOC：" not in self.backend_history[-2]["content"]:
                            self.backend_history[-2]["content"] += f" {time_info}"
                    
                    # 更新后端长历史中对应的消息
                    if len(self.backend_long_history) >= 2:
                        # 检查是否已包含时间信息
                        if "<OOC：" not in self.backend_long_history[-2]["content"]:
                            self.backend_long_history[-2]["content"] += f" {time_info}"
                else:
                    print("信息| 时间信息已存在，跳过添加")
        except Exception as e:
            print(f"警告| 添加时间信息到短期记忆失败: {str(e)}")

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
                print(f"错误| 火山翻译API返回异常: {json.dumps(response, indent=2, ensure_ascii=False)}")
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
                    print(f"错误| 火山翻译异常: {str(e)}")
                    print(f"提示| 检测到请求超时，正在进行第 {retry_count + 1} 次重试...")
                    retry_count += 1
                    continue
                else:
                    print(f"错误| 火山翻译异常: {str(e)}")
                    traceback.print_exc()
                    return None
        
        return None

    def extract_dialogue_content(self, text):
        """提取说话内容"""
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
        
        print(f"信息| 处理后的内容: {cleaned_text}")
        return cleaned_text
        
    def text_to_speech(self, text):
        """TTS和播放"""
        try:
            # 构建请求数据
            request_data = REF_AUDIO_CONFIG.copy()
            request_data["text"] = text
            print(f"信息| TTS文本: {text}")
            print(f"信息|" + "-" * 100)
            
            # 调用TTS API
            response = requests.post(TTS_API_URL, json=request_data)
            
            # 检查响应
            if response.status_code != 200:
                print(f"错误| TTS错误: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"信息| {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                except:
                    print(f"信息| {response.text[:200]}")
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
            print(f"错误| TTS异常: {str(e)}")
            traceback.print_exc()
            return False

    def process_user_message(self, user_input, play_tts=True):
        """处理用户消息"""
        # 在用户输入前，先匹配上一次的AI回复
        ai_matched_memories = []
        if self.last_ai_response:
            ai_matched_memories = self.match_essences_with_text(self.last_ai_response)

        # 匹配当前用户输入
        user_matched_memories = self.match_essences_with_text(user_input)

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
            self.related_memories = []
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

        self.related_memories = selected_memories

        # 添加用户消息到后端历史和后端长历史
        self.backend_history.append({"role": "user", "content": user_input})
        self.backend_long_history.append({"role": "user", "content": user_input})
        
        print(f"信息| 用户消息: {user_input}")
        if ai_matched_memories:
            print(f"信息| AI回复匹配到的相关记忆: {[m['matched_essence'] for m in ai_matched_memories]}")
        if user_matched_memories:
            print(f"信息| 用户输入匹配到的相关记忆: {[m['matched_essence'] for m in user_matched_memories]}")
        
        # 打印最终选择的记忆
        if self.related_memories:
            print(f"信息| 最终选择的记忆 ({len(self.related_memories)}条): {[m['matched_essence'] for m in self.related_memories]}")
        else:
            print("信息| 未匹配到相关记忆或相关记忆已在'你的记忆'部分")

        # 调用`请求ChatAI`并获取回复
        tokens_used = None
        if self.use_chatai:
            # 调用`请求ChatAI`
            content, reasoning_content, tokens_used = self.call_chatai()

            # 清理AI回复
            content = content.strip()
            reasoning_content = reasoning_content.strip() if reasoning_content else ""

            # 组合思维链和最终回复
            combined_content = f"【{reasoning_content}】\n\n{content}" if reasoning_content else content

            # 保存当前AI回复，用于下一次匹配（使用原始回复，不包含思维链）
            self.last_ai_response = content
            
            # 添加组合后的AI回复到后端历史和后端长历史
            self.backend_history.append({"role": "assistant", "content": combined_content})
            self.backend_long_history.append({"role": "assistant", "content": combined_content})

            # 退出检测（使用原始回复检测）
            should_exit = False
            if self.tts_success and play_tts:
                print(f"信息| 退出标记检测结果: {'🤐' in content}")
                should_exit = self.process_ai_response(content)  # 使用原始回复
            else:
                print(f"信息| 退出标记检测结果: {'🤐' in content}")
                should_exit = "🤐" in content

            # 保存短期记忆
            try:
                file_path = "short_term_memory.json"
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(self.backend_history, file, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"警告| 保存`backend_history`到文件失败: {str(e)}")

            # 如果检测到退出标记，请求总结
            if should_exit:
                self.handle_exit_detection(content)  # 使用原始回复

            # 调用`保存长期记忆`
            self.save_long_term_memory()

            if reasoning_content:
                print(f"信息| AI思维链：\n{reasoning_content}")
            print(f"信息| AI对话内容：{content}")

            print(f"信息| Token: {tokens_used} | 请求条数：{len(self.backend_history)} | 总结条数：{len(self.backend_long_history)}")
            
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
        summary_system_prompt = self.fixed_system_prompt + "\n\n你的记忆:\n" + memory_for_summary
        
        # 使用后端长历史
        dialogue_history = self.backend_long_history
        print(f"信息| 后端长历史总条数: {len(dialogue_history)}")
        
        if len(dialogue_history) > SUMMARY_HISTORY_LENGTH:
            dialogue_history = dialogue_history[-SUMMARY_HISTORY_LENGTH:]
            print(f"信息| 截取最后{SUMMARY_HISTORY_LENGTH}条用于总结")
        else:
            print(f"信息| 使用全部{len(dialogue_history)}条用于总结")
        
        # 返回用于对话总结的历史
        summary_history = [{"role": "system", "content": summary_system_prompt}] + dialogue_history
        print(f"信息| 最终用于总结的条数: {len(summary_history)}")
        
        print("信息| 用于总结的历史记录详细内容:")
        for i, msg in enumerate(summary_history):
            print(f"      [{i}] {msg['role']}: {msg['content'][:9999]}{'...' if len(msg['content']) > 9999 else ''}")
        
        return summary_history
    
    def save_summary_result(self, summary_type, result):
        """保存总结结果"""
        try:
            debug_dir = "debug"
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
            
            print(f"信息| {summary_type}结果已保存到 {filename}")
        except Exception as e:
            print(f"警告| 保存{summary_type}结果失败: {str(e)}")

    def save_summary_messages(self, summary_type, messages):
        """保存总结消息列表"""
        try:
            debug_dir = "debug"
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
            
            print(f"信息| {summary_type}消息列表已保存到 {filename}")
            print(f"信息| 正在总结中……")
        except Exception as e:
            print(f"警告| 保存{summary_type}消息列表失败: {str(e)}")
        
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
                    print("信息| 已从短期记忆中删除总结相关的消息")
        except Exception as e:
            print(f"警告| 从短期记忆中删除总结消息失败: {str(e)}")

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
            print(f"错误| 总结API调用异常: {str(e)}")
            return "错误| 总结API调用失败", None

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
            <OOC：请总结以上对话：
            # 视角要求
            使用第一人称，即"我"(亚托莉)

            # 内容整理
            1. 保持内容简洁清晰，主语和对象描述正确
            2. 根据对话内容修改日记、约定、用户偏好、计划、动机和关键记忆
            3. 删除已完成或已过期的计划和动机

            # 输出规范
            ## diary: 日记
            - 例：凌晨，主人趁我睡觉偷偷亲了我。早上，我早起给主人做了早餐，然后在床边等待主人醒来，看着他睡觉的样子感觉好幸福……
            ## promise: 约定，你和用户间的长期有效的约定，描述对象和要求需具体
            - 例：我们约定永远不分开
            - 例：我向主人承诺要好好听话
            ## preference: 用户偏好，用户的偏好和信息，需细分(癖好、喜欢、讨厌、习惯、信息、特征、补充等七类)
            - 例：癖好：萝莉控；喜欢：喜欢吃辣……
            ## plan: 计划，你的未来计划
            - 例：x月x日；保持最完美的形象和主人一起回家
            - 例：明天；自己去买菜
            ## motivation: 动机，你未实现的内心渴望或驱动力
            - 例：想成为让主人骄傲的仿生人
            ## pivotal_memory: 关键记忆，你与用户发生的重大转变事件，描述需要直白具体
            - 例：我愿意给主人膝枕
            - 例：我同意和主人一起泡澡

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
                - 将相对日期(明天/后天)转换为具体日期(基于新记忆日期)
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
                - 例：凌晨，主人趁我睡觉偷偷亲了我。早上，我早起给主人做了早餐，然后在床边等待主人醒来，看着他睡觉的样子感觉好幸福……
                ## promise: 约定，你和用户间的长期有效的约定，描述对象和要求需具体
                - 例：我们约定永远不分开
                - 例：我向主人承诺要好好听话
                ## preference: 用户偏好，用户的偏好和信息，需细分(癖好、喜欢、讨厌、习惯、信息、特征、补充等七类)
                - 例：癖好：萝莉控；喜欢：喜欢吃辣……
                ## plan: 计划，你的未来计划
                - 例：x月x日；保持最完美的形象和主人一起回家
                ## motivation: 动机，你未实现的内心渴望或驱动力
                - 例：想成为让主人骄傲的仿生人
                ## pivotal_memory: 关键记忆，你与用户发生的重大转变事件，描述需要直白具体
                - 例：我愿意给主人膝枕
                - 例：我同意和主人一起泡澡
                
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
                print(f"信息| 递归总结完成: {recursive_summary[:9999]}")
                return recursive_summary
            else:
                # 没有旧记忆，直接保存当前总结
                self.save_memory_core(current_summary)
                print(f"信息| 总结完成（无旧记忆）: {current_summary[:9999]}")
                return current_summary
                
        except Exception as e:
            print(f"错误| 获取总结失败: {str(e)}")
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
            print(f"错误| 翻译失败: {str(e)}")
        
        if japanese_text:
            print(f"信息| 翻译后文本: {japanese_text}")
        
        # 调用`TTS和播放`处理
        if japanese_text:
            self.text_to_speech(japanese_text)
        elif dialogue_content:
            print("警告| 翻译错误，使用原文TTS")
            self.text_to_speech(dialogue_content)
        
        # 只返回是否检测到退出标记，不处理退出逻辑
        return "🤐" in ai_response

    def get_opening_line(self):
        """获取开场白"""
        return self.opening_line

    def delete_last_conversation_pair(self):
        """删除最后一轮对话"""
        deleted_count = 0
        
        # 从`backend_history`中删除最后一轮对话
        while len(self.backend_history) > 1:  # 保留系统消息
            last_message = self.backend_history[-1]
            if last_message["role"] == "assistant":
                # 删除AI回复
                self.backend_history.pop()
                deleted_count += 1
                # 继续检查前一条是否是用户消息
                if len(self.backend_history) > 1 and self.backend_history[-1]["role"] == "user":
                    self.backend_history.pop()
                    deleted_count += 1
                break
            elif last_message["role"] == "user":
                # 如果最后一条是用户消息，也删除
                self.backend_history.pop()
                deleted_count += 1
                break
            else:
                break
        
        # 从`backend_long_history`中删除最后一轮对话
        while len(self.backend_long_history) > 0:
            last_message = self.backend_long_history[-1]
            if last_message["role"] == "assistant":
                # 删除AI回复
                self.backend_long_history.pop()
                # 继续检查前一条是否是用户消息
                if len(self.backend_long_history) > 0 and self.backend_long_history[-1]["role"] == "user":
                    self.backend_long_history.pop()
                break
            elif last_message["role"] == "user":
                # 如果最后一条是用户消息，也删除
                self.backend_long_history.pop()
                break
            else:
                break
        
        print(f"信息| 已删除 {deleted_count} 条消息")
        return deleted_count

class BubbleLabel(QLabel):
    """气泡标签控件"""
    def __init__(self, text, is_user=False, is_system=False, parent=None):
        super().__init__(text, parent)
        self.is_user = is_user
        self.is_system = is_system
        
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
                }
            """)
            self.setAlignment(Qt.AlignLeft)
        
        # 设置大小策略
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

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
            # 尝试加载背景图片
            background_paths = [
                "background.jpg",
                "background.png",
                "assets/background.jpg",
                "assets/background.png"
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
                image = image.resize((540, 960), Image.Resampling.LANCZOS)
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
            print(f"背景图片加载失败: {e}")
            self.create_white_background()
    
    def create_white_background(self):
        """创建纯白色毛玻璃背景"""
        if HAS_PIL:
            # 创建白色图片并应用模糊
            white_image = Image.new('RGB', (540, 960), color='white')
            blurred_image = white_image.filter(ImageFilter.GaussianBlur(radius=5))
            blurred_image = blurred_image.convert("RGBA")
            data = blurred_image.tobytes("raw", "RGBA")
            q_image = QImage(data, blurred_image.size[0], blurred_image.size[1], QImage.Format_RGBA8888)
            self.background_pixmap = QPixmap.fromImage(q_image)
        else:
            # 如果没有PIL，创建纯色QPixmap
            self.background_pixmap = QPixmap(540, 960)
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
        # 固定窗口大小
        self.setFixedSize(540, 960)
        
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
        
        # 初始化后端服务和其他组件
        self.initialize_services()

    def create_header(self, main_layout):
        """创建顶栏"""
        header_container = FrostedGlassWidget(blur_radius=15, opacity=0.9)
        header_container.setFixedHeight(50)
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        # 添加AI名称标签
        ai_name_label = QLabel("亚托莉")
        ai_name_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
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
        
        # 文本框
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("请输入文本（Ctrl+Enter发送）")
        self.input_field.setFont(QFont("Microsoft YaHei", 12))
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
        self.send_button.setFont(QFont("Microsoft YaHei", 12))
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
        self.clear_button.setFont(QFont("Microsoft YaHei", 12))
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
        self.exit_button.setFont(QFont("Microsoft YaHei", 12))
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
        self.delete_button.setFont(QFont("Microsoft YaHei", 12))
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
            print(f"错误| 后端服务初始化失败: {str(e)}")
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
        deleted_count = self.backend_service.delete_last_conversation_pair()
        
        if deleted_count == 0:
            self.add_system_message("没有可删除的对话")
            return
            
        # 从前端界面删除气泡
        self.remove_last_conversation_bubbles()
        
        self.add_system_message(f"已删除最后一轮对话")

    def remove_last_conversation_bubbles(self):
        """从前端界面删除最后一轮对话的气泡"""
        # 从布局末尾开始查找并删除用户和AI消息气泡
        ai_bubble_found = False
        user_bubble_found = False
        
        # 从后往前遍历布局中的子控件
        for i in range(self.chat_layout.count() - 1, -1, -1):
            widget = self.chat_layout.itemAt(i).widget()
            if widget is None:
                continue
                
            # 查找包含气泡标签的容器
            container_layout = widget.layout()
            if container_layout is None:
                continue
                
            # 查找气泡标签
            for j in range(container_layout.count()):
                child_widget = container_layout.itemAt(j).widget()
                if isinstance(child_widget, BubbleLabel) and not child_widget.is_system:
                    if not ai_bubble_found and not child_widget.is_user:
                        # 找到AI气泡，删除整个容器
                        widget.deleteLater()
                        ai_bubble_found = True
                        break
                    elif not user_bubble_found and child_widget.is_user:
                        # 找到用户气泡，删除整个容器
                        widget.deleteLater()
                        user_bubble_found = True
                        break
            
            # 如果已经找到AI和用户气泡，停止搜索
            if ai_bubble_found and user_bubble_found:
                break

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
            # 更新布局
            self.chat_container.adjustSize()
            self.chat_layout.update()
            
            # 等待布局绘制完成
            QApplication.processEvents()

            scroll_area = self.centralWidget().findChild(QScrollArea)
            if scroll_area:
                scrollbar = scroll_area.verticalScrollBar()
                if scrollbar:
                    scrollbar.setValue(scrollbar.maximum())
                    QApplication.processEvents()
        except Exception as e:
             print(f"警告| 滚动到底部失败: {e}")
                
    def add_user_message(self, message):
        """添加用户消息"""
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(50, 5, 10, 5)

        # 添加弹性空间
        container_layout.addStretch()
        
        # 添加气泡标签
        bubble = BubbleLabel(message, is_user=True)
        container_layout.addWidget(bubble)
        
        # 使用图片头像
        avatar = AvatarLabel(is_user=True)
        container_layout.addWidget(avatar)
        
        # 添加到聊天布局
        self.chat_layout.addWidget(container)
        
        # 调用`滚动到底部`
        self.scroll_to_bottom()

    def add_ai_message(self, message):
        """添加AI消息"""
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(10, 5, 50, 5)
        
        # 使用图片头像
        avatar = AvatarLabel(is_user=False)
        container_layout.addWidget(avatar)
        
        # 添加气泡标签
        bubble = BubbleLabel(f"{message}")
        container_layout.addWidget(bubble)
        
        # 添加弹性空间
        container_layout.addStretch()
        
        # 添加到聊天布局
        self.chat_layout.addWidget(container)
        
        # 调用`滚动到底部`
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

    def clear_chat(self):
        """清空聊天记录"""
        if hasattr(self, 'backend_service'):
            self.backend_service.backend_history = [
                {"role": "system", "content": self.backend_service.system_prompt}
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
            self.error_occurred.emit(f"错误| AI请求出错: {str(e)}")

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
            print(f"错误| TTS播放失败: {str(e)}")
            self.play_finished.emit()

if __name__ == "__main__":
    # 创建应用实例
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 设置全局字体
    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)
    
    # 创建并显示主窗口
    window = ChatWindow()
    window.show()
    
    # 启动事件循环
    sys.exit(app.exec_())