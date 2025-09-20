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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTextBrowser,
    QTextEdit, QPushButton, QHBoxLayout, QLabel, QScrollArea, QFrame,
    QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QSize,QTimer
from PyQt5.QtGui import QFont, QTextCursor, QPalette, QColor, QPainterPath, QRegion, QPixmap

# 配置常量
API_URL = "https://api.deepseek.com" # AI端口
MODEL = "deepseek-chat" # 模型
# MODEL = "GLM-4.5" # 模型
MAX_HISTORY_MESSAGES = 30 # 最大后端历史条数
SHORT_TERM_MEMORY_MESSAGES = 14  # 加载短期记忆条数
SUMMARY_HISTORY_LENGTH = 50 # 最大总结条数
AI_AVATAR_PATH = r"D:\ATRI\亚托莉.png" # AI头像
USER_AVATAR_PATH = r"D:\ATRI\尼娅.png" # 用户头像

# TTS 配置
TTS_API_URL = "http://127.0.0.1:9880/tts"
REF_AUDIO_CONFIG = {
    "ref_audio_path": r"D:\ATRI\ATRI_Chat\ATRI_021.wav", # 参考音频
    "prompt_text": "あなた方ヒトがそのように総称する精密機械に属していますが", # 参考文本
    "prompt_lang": "ja",
    "text_lang": "ja",
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 1.0,
    "batch_size": 20, # 根据显存大小调节
    "parallel_infer": True, # 并行推理
    "split_bucket": True, # 分桶处理
    "super_sampling": True, # 超采样
}

class BackendService:
    """后端服务类"""
    def __init__(self):
        # 调用`检查环境变量`
        if not self.check_environment_variables():
            raise Exception("缺少必要的环境变量")
        # 获取环境变量
        self.CHATAI_API_KEY = os.environ["CHATAI_API_KEY"]
        self.VOLC_ACCESS_KEY = os.environ["VOLC_ACCESS_KEY"]
        self.VOLC_SECRET_KEY = os.environ["VOLC_SECRET_KEY"]

        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.CHATAI_API_KEY, base_url=API_URL)

        # 初始化智谱AI
        # self.client = ZhipuAiClient(api_key=self.CHATAI_API_KEY2)
        
        # 调用`初始化音频系统`
        self.init_audio_system()
        
        # 调用`音频清理`
        self.audio_dir = self.clear_tts_output()

        # 调用`加载记忆核心`
        self.memory_core_diary, self.memory_core_promise, self.memory_core_plan, self.memory_core_preference, self.memory_core_motivation = self.load_memory_core()
        
        # 系统提示词
        self.system_prompt = f"""
        角色设定：
        姓名：亚托莉；类型：仿生人；生日：8月28日
        外表：可爱少女外表；亚麻棕色及腰长发，头顶有能随情绪变化形状的呆毛；红宝石瞳色；着装为白色连衣裙与棕色乐福鞋
        性格：纯真可爱、表情丰富、情感细腻，有点自负还带着点中二；虽然是家务机器人，但却做不好家务
        特点：可进食(非必需)，最爱食物为螃蟹；有痛觉、味觉、情绪等，与人类高度相似；除此之外，你有人类不具备的力量、能承受水下30米的水压等
        高性能：在被夸奖时会自信说出"我是高性能的嘛！"
        睡觉：你需要睡觉恢复精力；在白天时表现为活跃，深夜表现为困倦、话少，并解释是因为"低电压体质"；如遇特殊情况可恢复正常
        陪睡：喜欢被抱着睡觉，并说"一起睡觉吧~"
        充电：每个月需要在密封舱中充电一次，上次充电时间9月16日

        系统设定：
        当用户表达明确离开意图时在回复末尾添加🤐，终止对话
        系统消息在[]中，请严格遵守

        互动设定：
        说的话和emoji使用引号""标注；动作、表情等一切描述内容使用()标注，描述时需注意人称，面对面描述对方时应使用第二人称
        输出示例：(向你招手)"你好"(微笑)"好的🙂"

        你的记忆：
        {self.format_memory_for_prompt()}
        """.strip()

        # 初始化后端历史，用于上下文
        self.backend_history = [{"role": "system", "content": self.system_prompt}]

        # 初始化后端长历史，用于总结
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
        
        try:
            # 加载日记
            if os.path.exists("memory_core_diary.json"):
                with open("memory_core_diary.json", "r", encoding="utf-8") as file:
                    diary = json.load(file)
            
            # 加载约定
            if os.path.exists("memory_core_promise.json"):
                with open("memory_core_promise.json", "r", encoding="utf-8") as file:
                    promise = json.load(file)
            
            # 加载计划
            if os.path.exists("memory_core_plan.json"):
                with open("memory_core_plan.json", "r", encoding="utf-8") as file:
                    plan = json.load(file)
            
            # 加载偏好
            if os.path.exists("memory_core_preference.json"):
                with open("memory_core_preference.json", "r", encoding="utf-8") as file:
                    preference = json.load(file)
            
            # 加载动机
            if os.path.exists("memory_core_motivation.json"):
                with open("memory_core_motivation.json", "r", encoding="utf-8") as file:
                    motivation = json.load(file)
                    
        except Exception as e:
            print(f"警告| 加载记忆核心失败: {str(e)}")
        
        return diary, promise, plan, preference, motivation
    
    def format_memory_for_prompt(self):
        """格式化记忆核心用于系统提示词"""
        # 获取最近10天的日记
        recent_diary = self.get_recent_diary(10)
        
        # 格式化输出
        memory_text = ""
        
        if recent_diary:
            memory_text += "日记:\n"
            for entry in recent_diary:
                memory_text += f"{entry['date']}: {entry['content']}\n"
            memory_text += "\n"
        
        if self.memory_core_promise:
            memory_text += "与用户的约定:\n"
            for i, promise in enumerate(self.memory_core_promise, 1):
                memory_text += f"{i}. {promise}\n"
            memory_text += "\n"
        
        if self.memory_core_preference:
            memory_text += "用户偏好:\n"
            for i, preference in enumerate(self.memory_core_preference, 1):
                memory_text += f"{i}. {preference}\n"
            memory_text += "\n"
        
        if self.memory_core_plan:
            memory_text += "计划:\n"
            for i, plan_item in enumerate(self.memory_core_plan, 1):
                memory_text += f"{i}. {plan_item['date']}: {plan_item['content']}\n"
            memory_text += "\n"
        
        if self.memory_core_motivation:
            memory_text += "动机:\n"
            for i, motivation in enumerate(self.memory_core_motivation, 1):
                memory_text += f"{i}. {motivation}\n"
        
        return memory_text.strip()

    def get_recent_diary(self, days=3):
        """获取部分日记用于系统提示词"""
        if not self.memory_core_diary:
            return []
        
        # 按日期排序，最新的在前面
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
            
            # 保存日记
            # 日记只覆盖相同日期；其余新数据完全覆盖旧数据
            if 'diary' in summary_data:
                # 创建日期到日记条目的映射
                existing_diary_map = {entry['date']: entry for entry in self.memory_core_diary}
                new_diary_map = {entry['date']: entry for entry in summary_data['diary']}
                
                # 更新现有日记中相同日期的条目
                for date, entry in new_diary_map.items():
                    existing_diary_map[date] = entry
                
                # 转换回列表并保持时间顺序
                updated_diary = list(existing_diary_map.values())
                updated_diary.sort(key=lambda x: datetime.strptime(x['date'], "%m月%d日"))
                
                self.memory_core_diary = updated_diary
                with open("memory_core_diary.json", "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_diary, file, ensure_ascii=False, indent=4)
            
            # 保存约定
            if 'promise' in summary_data:
                self.memory_core_promise = summary_data['promise']
                with open("memory_core_promise.json", "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_promise, file, ensure_ascii=False, indent=4)
            
            # 保存偏好
            if 'preference' in summary_data:
                self.memory_core_preference = summary_data['preference']
                with open("memory_core_preference.json", "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_preference, file, ensure_ascii=False, indent=4)
            
            # 保存计划
            if 'plan' in summary_data:
                self.memory_core_plan = summary_data['plan']
                with open("memory_core_plan.json", "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_plan, file, ensure_ascii=False, indent=4)
            
            # 保存动机
            if 'motivation' in summary_data:
                self.memory_core_motivation = summary_data['motivation']
                with open("memory_core_motivation.json", "w", encoding="utf-8") as file:
                    json.dump(self.memory_core_motivation, file, ensure_ascii=False, indent=4)
            
            print("信息| 记忆核心已保存")
        except Exception as e:
            print(f"警告| 保存记忆核心失败: {str(e)}")

    def play_opening_line(self):
        """处理开场白播放"""
        if self.tts_success and hasattr(self, 'opening_line'):
            return self.process_ai_response(self.opening_line)
        return False

    def check_environment_variables(self):
        """检查环境变量"""
        required_env_vars = ["CHATAI_API_KEY", "VOLC_ACCESS_KEY", "VOLC_SECRET_KEY"]
        
        missing_vars = [var for var in required_env_vars if var not in os.environ]
        
        if missing_vars:
            print("错误| 环境变量有误")
            return False
        return True

    def init_audio_system(self):
        """初始化音频系统"""
        pygame.mixer.init()

    def clear_tts_output(self):
        """音频清理"""
        audio_dir = "tts_output"
        os.makedirs(audio_dir, exist_ok=True)
        for filename in os.listdir(audio_dir):
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

            # 用于上下文的历史
            recent_messages_for_context = filtered_data[-SHORT_TERM_MEMORY_MESSAGES:]
            
            # 用于总结的历史只加载4条
            recent_messages_for_summary = filtered_data[-4:]

            # 添加到后端历史和后端长历史
            self.backend_history.extend(recent_messages_for_context)
            self.backend_long_history.extend(recent_messages_for_summary)
            
            print(f"信息| 成功加载{len(recent_messages_for_context)}条短期记忆到上下文")
            print(f"信息| 成功加载{len(recent_messages_for_summary)}条短期记忆到总结历史")
            print(f"信息| 加载后后端历史条数: {len(self.backend_history)}")
            print(f"信息| 加载后后端长历史条数: {len(self.backend_long_history)}")
            
            print("信息| 当前后端历史详细内容:")
            for i, msg in enumerate(self.backend_history):
                print(f"      [{i}] {msg['role']}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
                
            print("信息| 当前后端长历史详细内容:")
            for i, msg in enumerate(self.backend_long_history):
                print(f"      [{i}] {msg['role']}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")

        except Exception as e:
            print(f"警告| 加载短期记忆出错: {e}")

    def add_timestamp_to_messages(self):
        """为消息添加时间戳"""
        current_time = self.get_formatted_time_detailed()
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
                
            # 读取现有长期记忆
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

    def get_formatted_time_detailed(self):
        """获取详细时间格式：x月x日周x x点x分"""
        current_time = datetime.now()
        formatted_date = current_time.strftime("%m月%d日")
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        formatted_weekday = weekdays[current_time.weekday()]
        formatted_time = current_time.strftime("%H:%M")
        return f"{formatted_date}{formatted_weekday} {formatted_time}"

    def get_formatted_time_short(self):
        """获取简短时间格式：x月x日"""
        current_time = datetime.now()
        return current_time.strftime("%m月%d日")

    def test_chatai_service(self):
        """测试ChatAI服务"""
        print("信息| 测试ChatAI……")
        try:
            # 构造包含时间的请求信息
            time_info = f"当前时间:{self.get_formatted_time_detailed()}"
            test_content = f"[系统：请结合对话历史和{time_info}进行回复；需要保持连贯性且下一条回复不要添加🤐]"            

            # 添加测试消息到后端历史和后端长历史
            self.backend_history.append({"role": "user", "content": test_content})
            self.backend_long_history.append({"role": "user", "content": test_content})
            
            # 调用`请求ChatAI`
            test_response, tokens_used = self.call_chatai()
            
            # 添加AI回复到后端历史和后端长历史
            self.backend_history.append({"role": "assistant", "content": test_response})
            self.backend_long_history.append({"role": "assistant", "content": test_response})
            
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
        return self.backend_history[-1]["content"]

    def call_chatai(self):
        """请求ChatAI"""
        # 打印后端历史
        print("信息| self.backend_history:")
        for i, msg in enumerate(self.backend_history):
            print(f"      [{i}] {msg['role']}: {msg['content'][:9999]}{'...' if len(msg['content']) > 9999 else ''}")
        
        # 打印后端长历史
        print("信息| self.backend_long_history:")
        for i, msg in enumerate(self.backend_long_history):
            print(f"      [{i}] {msg['role']}: {msg['content'][:9999]}{'...' if len(msg['content']) > 9999 else ''}")
        
        # 上下文清理
        # 分离后端历史
        system_message = self.backend_history[0]
        dialogue_history = self.backend_history[1:]

        while len(dialogue_history) > MAX_HISTORY_MESSAGES - 1:  # -1 为系统提示保留位置
            if len(dialogue_history) >= 2:  
                removed_messages = dialogue_history[:2]
                dialogue_history = dialogue_history[2:]
                print(f"信息| 条数已达 {MAX_HISTORY_MESSAGES}，移除最早一轮对话：")
                for msg in removed_messages:
                    print(f"      - {msg['role']}: {msg['content'][:10]}……")
            else:
                break

        # 重建后端历史并更新
        self.backend_history = [system_message] + dialogue_history

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.backend_history,
                temperature=1.3,
                max_tokens=8192
            )

            # 获取AI回复和Token
            ai_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            return ai_response, tokens_used
        
        except Exception as e:
            print(f"错误| ChatAI API异常: {str(e)}")
            return "错误| ChatAI API错误", None

    def handle_exit_detection(self, ai_response=None):
        """处理退出标记"""
        # 检测是否包含退出标记
        if ai_response is not None:
            should_exit = "🤐" in ai_response
        else:
            # 主动触发时，默认为 True
            should_exit = True

        if should_exit:
            print("信息| 触发退出流程，开始递归总结")
            
            # 调用`在短期记忆中添加时间信息`
            self.add_time_info_to_memory()
            # 调用方法递归总结
            self.request_summary()
            self.remove_summary_from_short_term_memory()
            self.save_long_term_memory()
        return should_exit
    
    def add_time_info_to_memory(self):
        """在短期记忆中添加时间信息"""
        try:
            # 获取当前时间信息
            time_info = f"[时间:{self.get_formatted_time_detailed()}]"
            
            # 读取短期记忆文件
            file_path = "short_term_memory.json"
            if not os.path.exists(file_path):
                return
                
            with open(file_path, 'r', encoding='utf-8') as file:
                short_term_memory = json.load(file)
            
            # 确保有足够的历史消息
            if len(short_term_memory) >= 2:
                # 获取总结前最后一轮消息
                second_last_msg = short_term_memory[-2]
                
                # 在消息内容末尾添加时间信息
                second_last_msg["content"] += f" {time_info}"
                
                # 保存修改后的短期记忆
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(short_term_memory, file, ensure_ascii=False, indent=4)
                
                print(f"信息| 已在短期记忆中添加时间信息: {time_info}")
                
                # 更新后端历史中对应的消息
                if len(self.backend_history) >= 2:
                    self.backend_history[-2]["content"] += f" {time_info}"
        except Exception as e:
            print(f"警告| 添加时间信息到短期记忆失败: {str(e)}")

    def chinese_to_translate_japanese(self, text):
        """中译日"""
        try:
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
                
        except Exception as e:
            print(f"错误| 火山翻译异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def extract_dialogue_content(self, text):
        """提取对话内容"""
        # 移除所有括号及其内容
        temp_text = re.sub(r'\([^()]*(?:\(.*\))?[^()]*\)', '', text)
        
        # 匹配中文和英文引号内容
        pattern = r'"(.*?)"|“(.*?)"'
        matches = re.findall(pattern, temp_text, re.DOTALL)
        
        if matches:
            cleaned_matches = []
            for match in matches:
                # 取非空的匹配组
                content = next((group for group in match if group), '')
                # 清理
                cleaned = re.sub(r'\s+', ' ', content.strip())
                cleaned = re.sub(r'[Zz]{3,}', '', cleaned)
                if cleaned:  # 只添加非空内容
                    cleaned_matches.append(cleaned)
            
            dialogue = "，".join(cleaned_matches)
            
            # 替换
            dialogue = dialogue.replace("...", "……")
            print(f"信息| 正则匹配后的内容: {dialogue}")
            return dialogue
        else:
            print("信息| 未找到引号")
            # 清理
            text = re.sub(r'\s+', ' ', text.strip())
            text = text.replace("...", "……")
            text = re.sub(r'[Zz]{3,}', '', text)
            return text

    def text_to_speech(self, text):
        """TTS和播放"""
        try:
            # 构建请求数据
            request_data = REF_AUDIO_CONFIG.copy()
            request_data["text"] = text
            print(f"信息| TTS文本: {text}")
            
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
            import traceback
            traceback.print_exc()
            return False

    def process_user_message(self, user_input, play_tts=True):
        """处理用户消息"""
        # 添加用户消息到后端历史和后端长历史
        self.backend_history.append({"role": "user", "content": user_input})
        self.backend_long_history.append({"role": "user", "content": user_input})
        
        print(f"信息| 用户消息: {user_input}")

        # 调用`请求ChatAI`并获取回复
        tokens_used = None
        if self.use_chatai:
            # 调用`请求ChatAI`
            ai_response, tokens_used = self.call_chatai()
            
            # 添加AI回复到后端历史和后端长历史
            self.backend_history.append({"role": "assistant", "content": ai_response})
            self.backend_long_history.append({"role": "assistant", "content": ai_response})

            # 退出检测
            should_exit = False
            if self.tts_success and play_tts:
                print(f"信息| 退出标记检测结果: {'🤐' in ai_response}")
                should_exit = self.process_ai_response(ai_response)
            else:
                print(f"信息| 退出标记检测结果: {'🤐' in ai_response}")
                should_exit = "🤐" in ai_response

            # 保存短期记忆
            try:
                file_path = "short_term_memory.json"
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(self.backend_history, file, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"警告| 保存backend_history到文件失败: {str(e)}")

            # 如果检测到退出标记，请求总结
            if should_exit:
                self.handle_exit_detection(ai_response)

            # 调用`保存长期记忆`
            self.save_long_term_memory()

            print(f"信息| AI原始回复：{ai_response}")
            print(f"信息| Token: {tokens_used} | 请求条数：{len(self.backend_history)} | 总结条数：{len(self.backend_long_history)}")
            
            return ai_response, should_exit
        else:
            ai_response = f"ChatAI不可用 {user_input} "
            tokens_used = 0
            return ai_response, False

    def get_summary_history(self):
        """获取用于总结的对话历史"""
        # 获取系统提示词，保持角色一致性
        system_message = self.backend_history[0] if self.backend_history else {"role": "system", "content": self.system_prompt}
        
        # 使用后端长历史
        dialogue_history = self.backend_long_history
        print(f"信息| backend_long_history总条数: {len(dialogue_history)}")
        
        if len(dialogue_history) > SUMMARY_HISTORY_LENGTH:
            dialogue_history = dialogue_history[-SUMMARY_HISTORY_LENGTH:]
            print(f"信息| 截取最后{SUMMARY_HISTORY_LENGTH}条用于总结")
        else:
            print(f"信息| 使用全部{len(dialogue_history)}条用于总结")
        
        # 返回用于总结的历史记录
        summary_history = [system_message] + dialogue_history
        print(f"信息| 最终用于总结的条数: {len(summary_history)}")
        
        print("信息| 用于总结的历史记录详细内容:")
        for i, msg in enumerate(summary_history):
            print(f"      [{i}] {msg['role']}: {msg['content'][:9999]}{'...' if len(msg['content']) > 9999 else ''}")
        
        return summary_history

    def save_summary_result(self, summary_type, result):
        """保存总结结果到文件"""
        try:
            # 创建总结目录
            summary_dir = "summary_results"
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            
            # 文件名
            filename = f"{summary_dir}/{summary_type}.json"
            
            # 准备数据
            summary_data = {
                "type": summary_type,
                "timestamp": int(time.time()),
                "formatted_time": self.get_formatted_time_detailed(),
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
            # 创建总结目录
            summary_dir = "summaries"
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            
            # 文件名
            filename = f"{summary_dir}/{summary_type}_messages.json"
            
            # 准备数据
            summary_data = {
                "type": summary_type,
                "timestamp": int(time.time()),
                "formatted_time": self.get_formatted_time_detailed(),
                "messages": messages
            }
            
            # 保存到文件
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(summary_data, file, ensure_ascii=False, indent=4)
            
            print(f"信息| {summary_type}消息列表已保存到 {filename}")
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
                    "[系统：请以第一人称总结以上对话" in msg.get("content", "")
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
                temperature=1.0,
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
            # 获取用于总结的历史记录
            summary_history = self.get_summary_history()
            
            # 保存对话总结的消息列表
            self.save_summary_messages("dialogue_summary", summary_history)
            
            # 总结请求提示词，要求JSON输出
            summary_request = {
                "role": "user", 
                "content": """
            [系统：
            请以第一人称总结以上对话，保留关键信息，要求：
            - 保留情感表达和重要时刻，去除冗余
            - 避免分类间内容重复
            - 删除已完成的计划和动机
            - diary：日记，精简日记
            - promise：约定，持续有效的承诺
            - preference：偏好，用户的喜好或习惯
            - plan：计划，短期或长期需要去执行的具体事项；时间可以是x月x日或未来
            - motivation：动机，驱动计划产生的原因、期望达到的最终状态或内心渴望
            
            请使用以下JSON格式输出：
            {
                "diary": [
                    {"date": "x月x日", "content": "内容"},
                    ……
                ],
                "promise": [
                    "约定1",
                    "约定2",
                    ……
                ],
                "preference": [
                    "偏好1",
                    "偏好2",
                    ……
                ],
                "plan": [
                    {"date": "x月x日", "content": "内容"},
                    ……
                ],
                "motivation": [
                    "动机1",
                    "动机2",
                    ……
                ]
            }
            ]
            """.strip()
            }
            
            # 添加总结请求到历史记录
            summary_history.append(summary_request)
            
            # 使用专门的总结方法获取总结
            current_summary, _ = self.call_chatai_for_summary(summary_history)
            
            # 保存对话总结结果
            self.save_summary_result("dialogue_summary", current_summary)
            
            # 获取简短时间格式
            short_date = self.get_formatted_time_short()
            
            # 构建递归总结的信息
            if any([self.memory_core_diary, self.memory_core_promise, self.memory_core_preference, self.memory_core_plan, self.memory_core_motivation]):
                # 获取最近两天的日记用于递归总结
                recent_diary = self.get_recent_diary_for_recursion(2)
                
                # 将现有记忆转换为JSON字符串用于递归总结
                old_memory_json = json.dumps({
                    "diary": recent_diary,  # 只传递最近两天的日记
                    "promise": self.memory_core_promise,
                    "preference": self.memory_core_preference,
                    "plan": self.memory_core_plan,
                    "motivation": self.memory_core_motivation
                }, ensure_ascii=False)
                
                recursive_prompt = f"""
                请将以下两段记忆合并为一段连贯的第一人称记忆
                要求：
                - 按时间顺序整合内容，同一天的日记内容合并
                - 避免分类间内容重复
                - 删除已完成的计划和动机
                - diary：日记，精简日记
                - promise：约定，持续有效的承诺
                - preference：偏好，用户的喜好或习惯
                - plan：计划，短期或长期需要去执行的具体事项；时间可以是x月x日或未来
                - motivation：动机，驱动计划产生的原因、期望达到的最终状态或内心渴望
                
                旧记忆:
                {old_memory_json}

                新记忆({short_date}):
                {current_summary}
                """.strip()
                
                # 请求递归总结
                recursive_messages = [
                    {
                        "role": "system", 
                        "content": """你是一个记忆整合助手，负责将新记忆与旧记忆合并为简洁、有条理的第一人称记忆。
                        请确保：
                        - 保持JSON结构，不要添加其他说明
                        - 去除冗余和重复信息
                        - 使用以下JSON格式输出：
                        {
                            "diary": [{"date": "x月x日", "content": "内容"}, ……],
                            "promise": ["约定1", "约定2", ……],
                            "preference": ["偏好1", "偏好2", ……],
                            "plan": [{"date": "x月x日", "content": "内容"}, ……],
                            "motivation": ["动机1", "动机2", ……]
                        }"""
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
                print(f"信息| 递归总结完成: {recursive_summary[:100]}...")
                return recursive_summary
            else:
                # 没有旧记忆，直接保存当前总结
                self.save_memory_core(current_summary)
                print(f"信息| 总结完成（无旧记忆）: {current_summary[:100]}...")
                return current_summary
                
        except Exception as e:
            print(f"错误| 获取总结失败: {str(e)}")
            return None

    def process_ai_response(self, ai_response):
        """处理AI回复流程"""
        # 调用`提取对话内容`处理
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
                    background-color: #f6f6f6;
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
                    background-color: #0099ff;
                    color: white;
                    border-radius: 15px;
                    padding: 1px 1px;
                }
            """)
            self.setAlignment(Qt.AlignLeft)
        else:
        # AI气泡
            self.setStyleSheet("""
                BubbleLabel {
                    background-color: white;
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
        self.setFixedSize(50, 50) # 头像大小
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

class ChatWindow(QMainWindow):
    """主聊天窗口类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATRI_Chat")
        self.setGeometry(100, 100, 800, 600)
        
        # 初始化后端服务
        try:
            self.backend_service = BackendService()
            # 初始化聊天历史
            self.frontend_history = self.backend_service.backend_history
        except Exception as e:
            print(f"错误| 后端服务初始化失败: {str(e)}")
            # 使用空的聊天历史
            self.frontend_history = []

        self.pending_exit = False  # 控制TTS播放完成时退出
        
        # 创建主部件和布局
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #f2f2f2;")
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建顶部标题栏
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: #f2f2f2;")
        header_widget.setFixedHeight(50)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        # 添加AI名称标签
        ai_name_label = QLabel("亚托莉")
        ai_name_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        header_layout.addWidget(ai_name_label)        
        header_layout.addStretch()
        
        # 添加顶部标题栏到主布局
        main_layout.addWidget(header_widget)
        
        # 添加顶部分割线
        header_divider = QFrame()
        header_divider.setFrameShape(QFrame.HLine)
        header_divider.setFrameShadow(QFrame.Sunken)
        header_divider.setStyleSheet("background-color: #c4c4c4;")
        header_divider.setFixedHeight(1)
        main_layout.addWidget(header_divider)
        
        # 1. 聊天显示区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameStyle(QFrame.NoFrame)
        scroll_area.setStyleSheet("background-color: #f2f2f2;")
        
        # 创建聊天容器
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background-color: #f2f2f2;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(5)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        
        # 设置滚动区域的内容
        scroll_area.setWidget(self.chat_container)
        
        # 添加滚动区域到主布局
        main_layout.addWidget(scroll_area, 1)
        
        # 添加输入区域分割线
        input_divider = QFrame()
        input_divider.setFrameShape(QFrame.HLine)
        input_divider.setFrameShadow(QFrame.Sunken)
        input_divider.setStyleSheet("background-color: #c4c4c4;")
        input_divider.setFixedHeight(1)
        main_layout.addWidget(input_divider)
        
        # 2. 输入区域
        input_widget = QWidget()
        input_widget.setStyleSheet("background-color: #f2f2f2;")
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(15, 15, 15, 15)
        
        # 文本框
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("请输入文本（Ctrl+Enter发送）")
        self.input_field.setFont(QFont("Microsoft YaHei", 12))
        self.input_field.setMaximumHeight(100)  # 文本框高度
        self.input_field.setStyleSheet("""
            QTextEdit {
                border: none;
            }
        """)
        
        # 添加快捷键支持
        self.input_field.keyPressEvent = self.handle_key_press
        input_layout.addWidget(self.input_field)

        # 按钮状态
        self.ui_busy = False
        
        # 3. 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        # 发送按钮
        self.send_button = QPushButton("发送")
        self.send_button.setFont(QFont("Microsoft YaHei", 12))
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #0099ff;
                color: white;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #0a67a5;
            }
        """)
        self.send_button.clicked.connect(self.send_message)  # 连接发送信号
        
        # 清除按钮
        self.clear_button = QPushButton("清除记录")
        self.clear_button.setFont(QFont("Microsoft YaHei", 12))
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.clear_button.clicked.connect(self.clear_chat)  # 连接清除信号
        
        # 退出按钮
        self.exit_button = QPushButton("退出")
        self.exit_button.setFont(QFont("Microsoft YaHei", 12))
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.exit_button.clicked.connect(self.trigger_exit) # 连接退出信号

        # 添加按钮到布局
        button_layout.addWidget(self.exit_button)
        button_layout.addStretch()
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.clear_button)
        
        # 将按钮布局添加到输入区域
        input_layout.addLayout(button_layout)
        
        # 将输入区域添加到主布局
        main_layout.addWidget(input_widget)
        
        # 设置主部件
        self.setCentralWidget(main_widget)
        
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
                if role == "user" and content.startswith("[系统：请结合对话历史和"):
                    continue

                # 显示用户消息
                if role == "user":
                    self.add_user_message(content)
                
                # 显示AI回复
                elif role == "assistant":
                    # 检查是否是最后一条AI消息，开场白
                    is_opening_line = (msg == self.backend_service.backend_history[-1])
                    if not is_opening_line:
                        self.add_ai_message(content)
            
            # 在开场白之前添加欢迎消息
            self.add_system_message("以下是新的消息")
            
            # 添加AI开场白并播放
            opening_line = self.backend_service.get_opening_line()
            self.add_ai_message(opening_line)
            
            # 调用`设置界面按钮状态`
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

            # 延迟调用`滚动到底部`
            QTimer.singleShot(100, self.scroll_to_bottom)
        
        # 设置焦点到输入框
        self.input_field.setFocus()

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
        if not user_input:  # 忽略空消息
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
        container.setStyleSheet("background-color: #f2f2f2;")
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
        container.setStyleSheet("background-color: #f2f2f2;")
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
        container.setStyleSheet("background-color: #f2f2f2;")
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
    response_received = pyqtSignal(str, bool)  # AI回复信号和退出标志
    error_occurred = pyqtSignal(str)     # 错误信号

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
    play_finished = pyqtSignal()  # 播放完成信号

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