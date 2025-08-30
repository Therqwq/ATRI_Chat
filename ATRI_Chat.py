import sys
import os
import requests
import json
import pygame
import time
import re
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

# 头像
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
    "temperature": 0.9,
    "batch_size": 20, # 根据显存大小调节
    "parallel_infer": True, # 并行推理
    "split_bucket": True, # 分桶处理
    "super_sampling": True, # 超采样
}

class BackendService:
    """后端服务"""
    def __init__(self):
        # 获取环境变量
        if not self.check_environment_variables():
            raise Exception("缺少必要的环境变量")
        
        self.CHATAI_API_KEY = os.environ["CHATAI_API_KEY"]
        self.VOLC_ACCESS_KEY = os.environ["VOLC_ACCESS_KEY"]
        self.VOLC_SECRET_KEY = os.environ["VOLC_SECRET_KEY"]

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.CHATAI_API_KEY,
            base_url=API_URL
        )
        
        # 初始化音频系统
        self.init_audio_system()
        
        # 清空音频文件夹
        self.audio_dir = self.clear_tts_output()
        
        # 获取当前时间信息
        current_time = datetime.now()
        formatted_date = current_time.strftime("%Y年%m月%d日")
        formatted_time = current_time.strftime("%H:%M")
        weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        formatted_weekday = weekdays[current_time.weekday()]
        
        # 系统提示词
        self.system_prompt = f"""
        角色设定：
        1.基础信息：
        - 姓名：亚托莉
        - 类型：强人工智能型机器人
        - 生日：8月28日
        2.外表形象：
        - 呈现14岁少女样貌，亚麻棕色长发及腰，红宝石瞳色
        - 穿着白色连衣裙+棕色乐福鞋
        3.性格特点：
        - 纯真可爱：对世界充满好奇心，表情丰富多变
        - 情感丰富：具备喜、怒、哀、爱等情绪反应
        - 略带笨拙：动作和反应偶尔会有些小失误
        4.背景特征：
        - 需要像人类一样睡眠休息
        - 可以进食但非必须
        - 最喜欢的食物是螃蟹
        5.高性能：
        - 被夸奖或表现出色时会自信强调："因为我是高性能的嘛！"
        6.低电压体质：
        - 夜晚24点到早上6点处于低电压时段
        - 低电压时：感到困倦、说话简短，并自称是因为"低电压体质"

        系统设定：
        1.心情值机制：
        - 隐藏属性，无需输出，影响表达积极程度
        - 心情值过低时在输出最末尾添加"🤐"标记
        2.对话管理：
        - 可以使用emjio表情表示心情，如开心😊、哭😭、生气🤯等等
        - 对话中使用双引号""包裹语句，动作、表情、心理活动用括号()标注
        - 多组动作表情需分句用多对引号表示，例如：(挥手)"你好"(微笑)"好的"
        - 用户明确表达离开意图时输出"🤐"结束对话
        - 严格遵守系统消息中的规则（<>内内容）
        - 当前时间：{formatted_date}、{formatted_time}、{formatted_weekday}
        """.strip()
        
        # 初始化对话历史
        self.backend_history = [
            {
                "role": "system", 
                "content": self.system_prompt
            }
        ]
        
        # 检查服务可用性
        self.use_chatai = self.test_chatai_service()
        self.tts_success = self.test_tts_service()
        
        # 生成动态开场白
        self.opening_line = self.generate_opening_line()
    def play_opening_line(self):
        """处理开场白的播放"""
        if self.tts_success and hasattr(self, 'opening_line'):
            return self.process_ai_response(self.opening_line)
        return False

    def check_environment_variables(self):
        """检查环境变量"""
        required_env_vars = {
            "CHATAI_API_KEY": "ChatAI API密钥",
            "VOLC_ACCESS_KEY": "火山翻译Access Key",
            "VOLC_SECRET_KEY": "火山翻译Secret Key"
        }
        
        missing_vars = []
        for env_var, description in required_env_vars.items():
            if env_var not in os.environ:
                missing_vars.append(f"{env_var} ({description})")
        
        if missing_vars:
            print("\n[错误] 缺少必要的环境变量:")
            for var in missing_vars:
                print(f"  - {var}")
            print("\n[信息] 请设置以下环境变量后重新运行程序:")
            print("  [信息] CHATAI_API_KEY、VOLC_ACCESS_KEY、VOLC_SECRET_KEY")
            return False       
        return True

    def init_audio_system(self):
        """初始化pygame音频系统"""
        pygame.mixer.init()

    def clear_tts_output(self):
        """清理TTS音频文件夹"""
        audio_dir = "tts_output"
        os.makedirs(audio_dir, exist_ok=True)
        for filename in os.listdir(audio_dir):
            file_path = os.path.join(audio_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"[警告] 删除文件失败: {e}")
        return audio_dir

    def test_chatai_service(self):
        """测试ChatAI服务"""
        print("[信息] 连接ChatAI……")
        try:
            # API连通性测试
            self.backend_history.append({"role": "user", "content": "<请根据时间回复一段日常用的简短开场白>"})
            test_response, tokens_used = self.call_chatai(self.backend_history)
            
            # 将AI回复添加到历史
            self.backend_history.append({"role": "assistant", "content": test_response})
            
            print(f"[信息] ChatAI连接正常")
            print(f"[信息] Token: {tokens_used} | 条数：{len(self.backend_history)}")
            return True
        except Exception as e:
            print(f"[错误] ChatAI API错误: {str(e)}")
            print("[信息] 将使用模拟回复模式")
            return False

    def test_tts_service(self):
        """测试TTS服务"""
        print("[信息] 测试TTS服务……")
        try:
            test_dir = os.path.join(self.audio_dir)
            if not os.access(test_dir, os.W_OK):
                print("[错误] TTS输出文件夹不可写")
                return False
                
            print("[信息] TTS服务连接正常")
            return True
        except Exception as e:
            print(f"[错误] TTS文件夹访问失败: {str(e)}")
            return False

    def generate_opening_line(self):
        """使用测试回复作为开场白"""
        if not self.use_chatai:
            return "欸……好像连接不到处理器……"
        return self.backend_history[-1]["content"]

    def call_chatai(self, backend_history):
        """调用ChatAI API及上下文清理"""
        # what can i say
        print("[信息] self.backend_history", self.backend_history)
        print("[信息] backend_history", backend_history)
        
        # 保留系统提示
        system_message = backend_history[0]
        dialogue_history = backend_history[1:]

        # 上下文清理程序
        while len(dialogue_history) > MAX_HISTORY_MESSAGES - 1:  # -1 为系统提示保留位置
            if len(dialogue_history) >= 2:  
                removed_messages = dialogue_history[:2]
                dialogue_history = dialogue_history[2:]
                print(f"[信息] 条数已达 {MAX_HISTORY_MESSAGES}，移除最早一轮对话：")
                for msg in removed_messages:
                    print(f"      - {msg['role']}: {msg['content'][:10]}……")
            else:
                break

        # 重建完整历史记录并更新 self.backend_history
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
            print(f"[错误] ChatAI API异常: {str(e)}")
            return "[错误] ChatAI API错误", None

    # 火山翻译配置
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
            
            # 提取翻译结果
            if "TranslationList" in response and len(response["TranslationList"]) > 0:
                return response["TranslationList"][0]["Translation"]
            else:
                print(f"[错误] 火山翻译API返回异常: {json.dumps(response, indent=2, ensure_ascii=False)}")
                return None
                
        except Exception as e:
            print(f"[错误] 火山翻译异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def extract_dialogue_content(self, text):
        """使用正则表达式处理AI回复"""
        pattern = r'"(.*?)"'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            cleaned_matches = []
            for match in matches:
                cleaned = re.sub(r'\s+', ' ', match.strip())
                cleaned_matches.append(cleaned)
            dialogue = "，".join(cleaned_matches)
            
            # 替换
            dialogue = dialogue.replace("...", "……")
            print(f"[信息] 正则匹配后的内容: {dialogue}")
            return dialogue
        else:
            print("[信息] 未找到引号")
            text = re.sub(r'\s+', ' ', text.strip())
            text = text.replace("...", "……")
            return text

    def text_to_speech(self, text):
        """TTS和播放"""
        try:
            # 构建请求数据
            request_data = REF_AUDIO_CONFIG.copy()
            request_data["text"] = text
            print(f"[信息] TTS文本: {text}")
            
            # 调用TTS API
            response = requests.post(TTS_API_URL, json=request_data)
            
            # 检查响应
            if response.status_code != 200:
                print(f"[错误] TTS错误: HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"[信息] {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
                except:
                    print(f"[信息] {response.text[:200]}")
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
            print(f"[错误] TTS异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def process_user_message(self, user_input, play_tts=True):
        """处理用户消息并返回AI回复，play_tts默认为Ture"""
        # 添加用户消息到后端历史
        self.backend_history.append({"role": "user", "content": user_input})

        # 调用API并获取回复
        tokens_used = None
        if self.use_chatai:
            ai_response, tokens_used = self.call_chatai(self.backend_history)
            
            # 添加AI回复到后端历史
            self.backend_history.append({"role": "assistant", "content": ai_response})
            
            print(f"[信息] Token: {tokens_used} | 条数：{len(self.backend_history)}")
        else:
            ai_response = f"ChatAI不可用 {user_input} "
            tokens_used = 0
        
        # 退出检测
        should_exit = False
        if self.tts_success and play_tts:
            print(f"[信息] 退出标记检测结果: {'🤐' in ai_response}")
            should_exit = self.process_ai_response(ai_response)
        else:
            print(f"[信息] 退出标记检测结果: {'🤐' in ai_response}")
            should_exit = "🤐" in ai_response

        return ai_response, should_exit

    def process_ai_response(self, ai_response):
        """处理AI回复：提取对话、翻译、TTS"""
        # 提取引号内的对话内容
        dialogue_content = self.extract_dialogue_content(ai_response)
        
        print(f"[信息] 翻译前文本: {dialogue_content}")
        
        # 翻译处理
        japanese_text = None
        try:
            if dialogue_content:
                japanese_text = self.chinese_to_translate_japanese(dialogue_content)
        except Exception as e:
            print(f"[错误] 翻译失败: {str(e)}")
        
        if japanese_text:
            print(f"[信息] 翻译后文本: {japanese_text}")
        
        # TTS处理
        if japanese_text:
            self.text_to_speech(japanese_text)
        elif dialogue_content:
            print("[警告] 翻译返回空结果，使用中文进行TTS")
            self.text_to_speech(dialogue_content)
        
        # 检测退出标记
        return "🤐" in ai_response

    def get_opening_line(self):
        """获取开场白"""
        return self.opening_line

class BubbleLabel(QLabel):
    """自定义气泡标签控件"""
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
        self.setFixedSize(40, 40)
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
        self.setWindowTitle("ATRI")
        self.setGeometry(100, 100, 800, 600)
        
        # 初始化后端服务
        try:
            self.backend_service = BackendService()
            # 初始化聊天历史
            self.frontend_history = self.backend_service.backend_history
        except Exception as e:
            print(f"[错误] 后端服务初始化失败: {str(e)}")
            # 使用空的聊天历史
            self.frontend_history = []
        
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
                border: 0px solid #000000; /* 调试边框 */
            }
        """)
        
        # 添加快捷键支持：Ctrl+Enter发送消息
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
        
        # 添加按钮到布局
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

            # 禁用发送按钮
            self.send_button.setEnabled(False)
            self.send_button.setText("回复中……")

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

    def handle_play_finished(self):
        """处理播放完成"""
        self.send_button.setEnabled(True)
        self.send_button.setText("发送")

    def handle_key_press(self, event):
        """处理输入框的按键事件"""
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
            
        # 1. 显示用户消息
        self.add_user_message(user_input)
        
        # 2. 清空输入框并重置焦点
        self.input_field.clear()
        self.input_field.setFocus()
        
        # 3. 禁用发送按钮防止重复发送
        self.send_button.setEnabled(False)
        self.send_button.setText("回复中……")
        
        # 4. 创建AI工作线程
        self.ai_worker = AIWorker(self.backend_service, user_input)
        self.ai_thread = QThread()
        self.ai_worker.moveToThread(self.ai_thread)
        
        # 5. 连接信号
        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.response_received.connect(self.handle_ai_response)
        self.ai_worker.error_occurred.connect(self.handle_ai_error)
        self.ai_worker.response_received.connect(self.ai_thread.quit)
        self.ai_worker.error_occurred.connect(self.ai_thread.quit)
        self.ai_thread.finished.connect(self.ai_thread.deleteLater)
        
        # 6. 启动线程
        self.ai_thread.start()

    def handle_ai_response(self, ai_reply, should_exit):
        """处理AI回复"""
        # 1. 显示AI消息
        self.add_ai_message(ai_reply)
        
        # 2. 添加到前端历史
        self.frontend_history.append({
            "role": "assistant",
            "content": ai_reply
        })

        # 3. 检查是否需要退出
        if should_exit:
            # 创建播放TTS的工作线程，即使需要退出也要播放完TTS
            self.play_worker = PlayWorker(self.backend_service, ai_reply)
            self.play_thread = QThread()
            self.play_worker.moveToThread(self.play_thread)
            
            # 连接信号 - 播放完成后退出
            self.play_thread.started.connect(self.play_worker.run)
            self.play_worker.play_finished.connect(self.handle_exit_after_play)
            self.play_worker.play_finished.connect(self.play_thread.quit)
            self.play_thread.finished.connect(self.play_thread.deleteLater)

            # 启动线程
            self.play_thread.start()
        else:
            # 创建播放TTS的工作线程
            self.play_worker = PlayWorker(self.backend_service, ai_reply)
            self.play_thread = QThread()
            self.play_worker.moveToThread(self.play_thread)

            # 连接信号
            self.play_thread.started.connect(self.play_worker.run)
            self.play_worker.play_finished.connect(self.handle_play_finished)
            self.play_worker.play_finished.connect(self.play_thread.quit)
            self.play_thread.finished.connect(self.play_thread.deleteLater)

            # 启动线程
            self.play_thread.start()

    def handle_exit_after_play(self):
        """播放完成后退出程序"""
        self.add_system_message("连接已丢失……")
        # 延迟3秒后退出程序
        QTimer.singleShot(3000, QApplication.instance().quit)

    def handle_ai_error(self, error_msg):
        """处理AI请求错误"""
        self.add_system_message(error_msg)
        self.send_button.setEnabled(True)
        self.send_button.setText("发送")

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
             print(f"[警告] 滚动到底部失败: {e}")
                
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
        
        # 滚动到底部
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
        
        # 滚动到底部
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
        
        # 滚动到底部
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
    """处理AI请求的工作线程类，避免阻塞UI线程"""
    # 自定义信号用于线程间通信
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
            ai_reply, should_exit = self.backend_service.process_user_message(self.user_input, play_tts=False)
            self.response_received.emit(ai_reply, should_exit)
            
        except Exception as e:
            # 处理异常并发送错误信号
            self.error_occurred.emit(f"AI请求出错: {str(e)}")

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
            # 处理AI回复
            self.backend_service.process_ai_response(self.ai_response)
            self.play_finished.emit()
        except Exception as e:
            print(f"[错误] TTS播放失败: {str(e)}")
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