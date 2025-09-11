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
MAX_HISTORY_MESSAGES = 30 # 最大历史消息条数
SHORT_TERM_MEMORY_MESSAGES = 10  # 加载短期历史条数
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
        
        # 调用`初始化音频系统`
        self.init_audio_system()
        
        # 调用`音频清理`
        self.audio_dir = self.clear_tts_output()

        # 调用`加载上次记忆`
        last_memory = self.load_last_memory()
        
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
        充电：需要在密封舱中充电

        当前状态：
        和用户住在一起，称呼用户为主人

        用户相关：
        用户上学时间工作日的8:20-11:30、19:00-21:00

        系统设定：
        当用户表达的明确离开意图回复'🤐'终止对话
        系统消息提示在[]中，请严格遵守

        互动设定：
        说的话和emoji使用引号""标注；动作、表情等一切描述内容使用()标注，描述时需注意人称，面对面描述对方时应使用第二人称
        输出示例：(招手)"你好"(微笑)"好的🙂"

        f"{'记忆摘要：\n\n' + last_memory if last_memory else ''}"
        """.strip()

        # 初始化后端历史
        self.backend_history = [{"role": "system", "content": self.system_prompt}]

        # 调用`加载短期记忆`
        self.load_short_term_memory_from_file()
        
        # 调用方法检测TTS和ChatAI服务
        self.use_chatai = self.test_chatai_service()
        self.tts_success = self.test_tts_service()

        # 调用`将测试回复作为开场白`
        self.opening_line = self.generate_opening_line()

    def load_last_memory(self):
        """加载上次记忆"""
        try:
            if os.path.exists("last_memory.json"):
                with open("last_memory.json", "r", encoding="utf-8") as file:
                    memory_data = json.load(file)
                    return memory_data.get("memory", "") if isinstance(memory_data, dict) else memory_data
            return ""
        except Exception as e:
            print(f"警告| 加载上次记忆失败: {str(e)}")
            return ""
        
    def save_last_memory(self, memory_content):
        """保存上次记忆"""
        try:
            if not isinstance(memory_content, str):
                memory_content = str(memory_content)
                
            memory_data = {"memory": memory_content}
            with open("last_memory.json", "w", encoding="utf-8") as file:
                json.dump(memory_data, file, ensure_ascii=False, indent=4)
            print("信息| 记忆已保存")
        except Exception as e:
            print(f"警告| 保存记忆失败: {str(e)}")

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
            print("错误| 请检查环境变量")
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
                print(f"警告| 删除文件失败: {e}")
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

            recent_messages = filtered_data[-SHORT_TERM_MEMORY_MESSAGES:]

            # 添加到`backend_history`
            self.backend_history.extend(recent_messages)
            print(f"信息| 成功加载 {len(recent_messages)} 条历史记录")

        except Exception as e:
            print(f"警告| 加载短期记忆出错: {e}")
    
    def save_long_term_memory(self):
        """保存长期记忆"""
        try:
            file_path = "long_term_memory.json"

            # 过滤"system"消息
            non_system_messages = [msg for msg in self.backend_history if msg.get("role") != "system"]

            # 冲突处理：新建或追加
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # 合并已有数据与新数据
            updated_data = existing_data + non_system_messages

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)

            print(f"信息| 存储 {len(non_system_messages)} 条消息到长期记忆")

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
            # 获取详细时间信息
            time_info = f"当前时间:{self.get_formatted_time_detailed()}"
            test_content = f"[系统：请结合对话历史和{time_info}进行回复；需要保持连贯性且下一条回复不要添加🤐]"            

            # 添加测试消息到后端历史
            self.backend_history.append({"role": "user", "content": test_content})
            
            # 调用call_chatai时不传递参数
            test_response, tokens_used = self.call_chatai()
            
            # 添加AI回复到后端历史
            self.backend_history.append({"role": "assistant", "content": test_response})
            
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
        """请求ChatAI流程"""
        # 打印后端历史
        print("信息| self.backend_history:")
        [print(f"      - {msg['role']}: {msg['content'][:9999]}……") for msg in self.backend_history]
        
        # 分离后端历史
        system_message = self.backend_history[0]
        dialogue_history = self.backend_history[1:]

        # 上下文清理
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
        # 如果传入了AI回复，则检测是否包含退出标记
        if ai_response is not None:
            should_exit = "🤐" in ai_response
        else:
            # 主动触发时，默认为 True
            should_exit = True

        if should_exit:
            print("信息| 检测到退出标记或手动触发退出，开始总结对话")
            
            # 调用`在短期记忆中添加时间信息`
            self.add_time_info_to_memory()
            # 调用方法总结流程
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
                
                # 更新`backend_history`中的对应消息
                if len(self.backend_history) >= 2:
                    self.backend_history[-2]["content"] += f" {time_info}"
        except Exception as e:
            print(f"警告| 添加时间信息到短期记忆失败: {str(e)}")

    def chinese_to_translate_japanese(self, text):
        """翻译服务"""
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
        # 先移除所有括号及其内容
        temp_text = re.sub(r'\([^()]*(?:\(.*\))?[^()]*\)', '', text)
        
        # 匹配中文和英文引号内容
        pattern = r'"(.*?)"|“(.*?)"'
        matches = re.findall(pattern, temp_text, re.DOTALL)
        
        if matches:
            cleaned_matches = []
            for match in matches:
                # 取非空的匹配组
                content = next((group for group in match if group), '')
                # 清理内容：去除首尾空格、合并多余空格、移除特定字符
                cleaned = re.sub(r'\s+', ' ', content.strip())
                cleaned = re.sub(r'[Zz]{3,}', '', cleaned)
                if cleaned:  # 只添加非空内容
                    cleaned_matches.append(cleaned)
            
            dialogue = "，".join(cleaned_matches)
            
            # 替换...为……
            dialogue = dialogue.replace("...", "……")
            print(f"信息| 正则匹配后的内容: {dialogue}")
            return dialogue
        else:
            print("信息| 未找到引号")
            # 清理原始文本
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
        # 添加用户消息到后端历史
        self.backend_history.append({"role": "user", "content": user_input})

        # 调用`请求ChatAI流程`并获取回复
        tokens_used = None
        if self.use_chatai:
            # 调用`call_chatai`
            ai_response, tokens_used = self.call_chatai()
            
            # 添加AI回复到后端历史
            self.backend_history.append({"role": "assistant", "content": ai_response})

            # 退出检测
            should_exit = False
            if self.tts_success and play_tts:
                print(f"信息| 退出标记检测结果: {'🤐' in ai_response}")
                should_exit = self.process_ai_response(ai_response)
            else:
                print(f"信息| 退出标记检测结果: {'🤐' in ai_response}")
                should_exit = "🤐" in ai_response

            # 保存`backend_history`到short_term_memory.json
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
            print(f"信息| Token: {tokens_used} | 条数：{len(self.backend_history)}")
            
            return ai_response, should_exit
        else:
            ai_response = f"ChatAI不可用 {user_input} "
            tokens_used = 0
            return ai_response, False
        
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

    def request_summary(self):
        """请求对话总结（递归总结）"""
        try:
            # 添加用户总结请求
            summary_request = {
                "role": "user", 
                "content": """
            [系统：请以第一人称总结以上对话，使用简短日记格式。请专注于保留以下内容：
            1. 对话中的关键信息和重要细节
            2. 用户与你之间的约定、承诺或计划
            3. 用户表达的重要偏好、兴趣或需求
            4. 需要记住的情感表达或重要时刻

            要求：
            使用简洁自然的语言，避免表情符号、括号动作描述或其他非文字修饰
            总结应连贯有条理，如同记录真实回忆

            格式：
            x月x日：今天……
            (日记部分)

            与主人的约定：
            1. ……
            2. ……

            与主人的经历：
            1. ……
            2. ……]

            动机：
            1. ……
            2. ……
            """.strip()
            }
            self.backend_history.append(summary_request)
            
            # 调用`call_chatai`获取总结
            current_summary, _ = self.call_chatai()
            
            # 添加AI总结回复
            self.backend_history.append({"role": "assistant", "content": current_summary})
            
            # 读取上次记忆
            old_memory = self.load_last_memory()

            # 获取简短时间格式
            short_date = self.get_formatted_time_short()
            
            # 构建递归总结的提示词
            if old_memory:
                recursive_prompt = f"""
                请将以下两段记忆按先后顺序合并为一段连贯的第一人称记忆摘要：

                旧记忆摘要:
                {old_memory}

                新记忆摘要({short_date}):
                {current_summary}

                要求：
                1. 按日记格式组织内容，使用"x月x日：今天……"的格式
                2. 将与用户的约定、经历和动机分别归类到专门部分，不要减少该部分内容
                3. 如果与用户的约定和经历部分与普通日记有重复，则删除日记部分
                4. 只保留重要信息，去除冗余内容

                请按以下格式组织内容：

                (日记部分)

                与用户的约定：
                1. ……
                2. ……

                与用户的经历:
                1. ……
                2. ……

                动机：
                1. ……
                2. ……
                """.strip()
            else:
                # 如果没有旧记忆，直接使用当前总结
                recursive_prompt = current_summary
            
            # 请求递归总结
            if old_memory:
                recursive_messages = [
                    {
                        "role": "system", 
                        "content": """
                        你是一个专业记忆整合助手，负责将两次记忆摘要按先后顺序合并为简洁、有条理的第一人称记忆摘要
                        请确保：
                        1. 按日期顺序组织内容
                        2. 清晰区分普通日记、约定、经历和动机
                        3. 保留所有关键信息
                        4. 使用自然流畅的日记风格
                        """.strip()
                    },
                    {"role": "user", "content": recursive_prompt}
                ]
                
                # 保存原始历史记录
                original_history = self.backend_history.copy()
                
                # 临时替换历史记录为递归总结专用的消息
                self.backend_history = recursive_messages
                
                # 获取递归总结
                recursive_summary, _ = self.call_chatai()
                
                # 恢复原始历史记录
                self.backend_history = original_history
                
                # 保存递归总结
                self.save_last_memory(recursive_summary)
                print(f"信息| 递归总结完成: {recursive_summary[:9999]}...")
                return recursive_summary
            else:
                # 没有旧记忆，直接保存当前总结
                self.save_last_memory(current_summary)
                print(f"信息| 总结完成（无旧记忆）: {current_summary[:9999]}...")
                return current_summary
                
        except Exception as e:
            print(f"错误| 获取总结失败: {str(e)}")
            return None

    def process_ai_response(self, ai_response):
        """处理AI回复流程"""
        # 调用`提取对话内容`处理
        dialogue_content = self.extract_dialogue_content(ai_response)
        
        # 调用`翻译服务`处理
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
        
        # 添加欢迎消息
        self.add_system_message("以下是新的消息")
        
        # 添加AI开场白并播放
        if hasattr(self, 'backend_service'):
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
            # 直接退出，不调用总结（因为AI触发时已经总结过了）
            QTimer.singleShot(2000, QApplication.instance().quit)
        else:
            # 调用`设置界面按钮状态`
            self.set_ui_busy(False)

    def handle_key_press(self, event):
        """处理输入框快捷键"""
        # 检查按下Ctrl+Enter后发送信息
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:
            self.send_message()
            return
        # 允许默认处理其他按键
        QTextEdit.keyPressEvent(self.input_field, event)

    def send_message(self):
        """处理用户发送消息"""
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
        
        # 6. 启动线程
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