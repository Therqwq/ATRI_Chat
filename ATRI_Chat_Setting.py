# ATRI_Chat_Setting
# 版本：v0.2.4
# 兼容平台：Linux、Windows

import sys
import os
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QFormLayout, QLineEdit, QCheckBox, QPushButton,
                             QMessageBox, QLabel, QScrollArea,
                             QComboBox, QHBoxLayout, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

# 配置文件路径，与脚本同目录
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def get_platform_font():
    """获取平台和字体"""
    platform_name = sys.platform
    if platform_name.startswith('win'):
        return "Microsoft YaHei"
    elif platform_name.startswith('linux'):
        return "Noto Sans"
    else:
        return "DejaVu Sans"

# 配置项的说明映射
CONFIG_DESCRIPTIONS = {
    "PROVIDER": "模型服务提供商，选择后会自动切换模型列表和 API 密钥",
    "MODEL": "模型名称，使用的语言模型",
    "CHAT_API_KEY": "深度求索 API Key",
    "CHAT_API_KEY2": "智谱 API Key",
    "TRANSLATION_ENGINE": "翻译服务引擎，目前仅支持火山引擎",
    "VOLC_ACCESS_KEY": "Access Key ID",
    "VOLC_SECRET_KEY": "Secret Access Key",
    "MAX_HISTORY_MESSAGES": "最大上下文条数，后端历史条数，填整数",
    "SHORT_TERM_MEMORY_MESSAGES": "加载短期记忆条数，启动时加载的到后端历史的条数，填整数",
    "SUMMARY_HISTORY_LENGTH": "最大对话总结条数，后端长历史条数，填整数",
    "MEMORY_DAYS": "加载记忆天数，填整数",
    "USE_TRANSLATION": "是否启用翻译功能",
    "USE_COT": "是否使用思维链",
    "USE_BETA": "使用 JSON 格式化语言模型输出，并使用更精细化的描述和场景推理"
}

PROVIDER_MODELS = {
    "深度求索": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "智谱": ["GLM-4.6", "GLM-4.7"]
}

TRANSLATION_ENGINES = ["火山引擎"]

class ConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATRI_Chat_Setting")
        self.resize(600, 750)
        self.config_data = {}
        self.widgets = {}

        self.combo_provider = None
        self.combo_model = None
        self.input_api_key = None
        self.input_api_key2 = None
        self.label_api_key = None
        self.label_api_key2 = None

        self.combo_trans_engine = None
        self.input_volc_ak = None
        self.input_volc_sk = None

        self.init_ui()
        self.load_config()

    def init_ui(self):
        main_container = QWidget()
        self.setCentralWidget(main_container)
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部标题栏
        header_widget = QWidget()
        header_widget.setFixedHeight(70)
        header_widget.setStyleSheet("""
            QWidget {
                background-color: #F5F7FA;
                border-bottom: 1px solid #E4E7ED;
            }
        """)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(25, 0, 25, 0)

        title = QLabel("<h2>ATRI_Chat 配置中心</h2>")
        # PyQt6: 使用 Qt.AlignmentFlag
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title.setStyleSheet("color: #303133; background: transparent; border: none;")
        header_layout.addWidget(title)

        version_label = QLabel("v0.2.4") # 更新版本号以匹配顶部注释
        version_label.setStyleSheet("color: #909399; background: transparent; border: none;")
        header_layout.addStretch()
        header_layout.addWidget(version_label)

        main_layout.addWidget(header_widget)

        # 滚动区域 
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # PyQt6: 使用 Qt.ScrollBarPolicy
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("background-color: #F5F7FA; border: none;")
        main_layout.addWidget(scroll)

        container = QWidget()
        container.setStyleSheet("background-color: #F5F7FA;")
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(20)
        container_layout.setContentsMargins(25, 25, 25, 25)

        # 创建卡片容器
        def create_card(title_text):
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border-radius: 8px;
                    border: 1px solid #EBEEF5;
                }
                QLabel { background-color: transparent; }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(20, 15, 20, 20)
            card_layout.setSpacing(10)

            card_title = QLabel(title_text)
            card_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #303133; border: none; border-bottom: 1px solid #EBEEF5; padding-bottom: 8px;")
            card_layout.addWidget(card_title)
            return card, card_layout

        # 语言模型配置卡片
        llm_card, llm_card_layout = create_card("语言模型配置")

        llm_form_widget = QWidget()
        llm_form_widget.setStyleSheet("background: transparent;")
        llm_layout = QFormLayout(llm_form_widget)
        llm_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        llm_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        llm_layout.setSpacing(15)
        llm_layout.setContentsMargins(0, 10, 0, 0)

        self.combo_provider = QComboBox()
        self.combo_provider.addItems(list(PROVIDER_MODELS.keys()))
        self.combo_provider.setToolTip(CONFIG_DESCRIPTIONS["PROVIDER"])
        self.combo_provider.currentIndexChanged.connect(self.on_provider_changed)
        llm_layout.addRow(QLabel("模型提供商"), self.combo_provider)

        self.combo_model = QComboBox()
        self.combo_model.setToolTip(CONFIG_DESCRIPTIONS["MODEL"])
        llm_layout.addRow(QLabel("模型名称"), self.combo_model)

        self.input_api_key = QLineEdit()
        # PyQt6: 使用 QLineEdit.EchoMode.Password
        self.input_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_api_key.setPlaceholderText("请输入密钥")
        self.label_api_key = QLabel("DeepSeek Key")
        self.label_api_key.setToolTip(CONFIG_DESCRIPTIONS["CHAT_API_KEY"])
        llm_layout.addRow(self.label_api_key, self.input_api_key)

        self.input_api_key2 = QLineEdit()
        self.input_api_key2.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_api_key2.setPlaceholderText("请输入密钥")
        self.label_api_key2 = QLabel("Zhipu Key")
        self.label_api_key2.setToolTip(CONFIG_DESCRIPTIONS["CHAT_API_KEY2"])
        llm_layout.addRow(self.label_api_key2, self.input_api_key2)

        llm_card_layout.addWidget(llm_form_widget)
        container_layout.addWidget(llm_card)

        # 翻译服务配置卡片
        trans_card, trans_card_layout = create_card("翻译服务配置")

        trans_form_widget = QWidget()
        trans_form_widget.setStyleSheet("background: transparent;")
        trans_layout = QFormLayout(trans_form_widget)
        trans_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        trans_layout.setSpacing(15)
        trans_layout.setContentsMargins(0, 10, 0, 0)

        self.combo_trans_engine = QComboBox()
        self.combo_trans_engine.addItems(TRANSLATION_ENGINES)
        self.combo_trans_engine.setToolTip(CONFIG_DESCRIPTIONS["TRANSLATION_ENGINE"])
        trans_layout.addRow(QLabel("翻译引擎"), self.combo_trans_engine)

        self.input_volc_ak = QLineEdit()
        self.input_volc_ak.setPlaceholderText("请输入火山引擎 Access Key ID")
        self.input_volc_ak.setToolTip(CONFIG_DESCRIPTIONS["VOLC_ACCESS_KEY"])
        trans_layout.addRow(QLabel("Access Key ID"), self.input_volc_ak)

        self.input_volc_sk = QLineEdit()
        self.input_volc_sk.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_volc_sk.setPlaceholderText("请输入火山引擎 Secret Access Key")
        self.input_volc_sk.setToolTip(CONFIG_DESCRIPTIONS["VOLC_SECRET_KEY"])
        trans_layout.addRow(QLabel("Secret Key"), self.input_volc_sk)

        trans_card_layout.addWidget(trans_form_widget)
        container_layout.addWidget(trans_card)

        # 主服务配置卡片
        main_card, main_card_layout = create_card("主服务配置")

        main_form_widget = QWidget()
        main_form_widget.setStyleSheet("background: transparent;")
        form_layout = QFormLayout(main_form_widget)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_layout.setSpacing(15)
        form_layout.setContentsMargins(0, 10, 0, 0)

        exclude_keys = ["PROVIDER", "MODEL", "CHAT_API_KEY", "CHAT_API_KEY2",
                        "TRANSLATION_ENGINE", "VOLC_ACCESS_KEY", "VOLC_SECRET_KEY"]
        order = [
            "MAX_HISTORY_MESSAGES",
            "SHORT_TERM_MEMORY_MESSAGES",
            "SUMMARY_HISTORY_LENGTH",
            "MEMORY_DAYS",
            "USE_TRANSLATION",
            "USE_COT",
            "USE_BETA"
        ]

        for key in order:
            if key in exclude_keys:
                continue

            desc = CONFIG_DESCRIPTIONS.get(key, "无描述")
            label = QLabel(key)
            label.setToolTip(desc)
            label.setStyleSheet("color: #606266; font-weight: bold; background: transparent;")

            if key.startswith("USE_"):
                widget = QCheckBox()
                widget.setToolTip(desc + " (勾选为 True)")
                widget.setStyleSheet("background: transparent;")
            else:
                widget = QLineEdit()
                widget.setToolTip(desc)
                widget.setPlaceholderText(desc)

            self.widgets[key] = widget
            form_layout.addRow(label, widget)

        main_card_layout.addWidget(main_form_widget)
        container_layout.addWidget(main_card)

        # 底部提示与按钮
        hint_label = QLabel("提示：将鼠标悬停在参数名称上可查看详细说明。")
        hint_label.setStyleSheet("color: #909399; font-size: 12px; margin-top: 5px;")
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_label.setWordWrap(True)
        container_layout.addWidget(hint_label)

        btn_widget = QWidget()
        btn_widget.setStyleSheet("background: transparent;")
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setSpacing(15)
        btn_layout.setContentsMargins(0, 10, 0, 0)

        save_btn = QPushButton("保存配置")
        save_btn.clicked.connect(self.save_config)
        # PyQt6: 使用 Qt.CursorShape
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.setFixedHeight(40)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #409EFF;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #66B1FF;
            }
            QPushButton:pressed {
                background-color: #3A8EE6;
            }
        """)

        close_btn = QPushButton("退出")
        close_btn.clicked.connect(self.close)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setFixedHeight(40)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: #606266;
                border-radius: 4px;
                border: 1px solid #DCDFE6;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #409EFF;
                border-color: #C6E2FF;
                background-color: #ECF5FF;
            }
        """)

        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        container_layout.addWidget(btn_widget)

        container_layout.addStretch()
        scroll.setWidget(container)

    def on_provider_changed(self, index):
        provider = self.combo_provider.currentText()
        self.combo_model.clear()
        models = PROVIDER_MODELS.get(provider, [])
        self.combo_model.addItems(models)

        if provider == "深度求索":
            self.label_api_key.setVisible(True)
            self.input_api_key.setVisible(True)
            self.label_api_key2.setVisible(False)
            self.input_api_key2.setVisible(False)
        elif provider == "智谱":
            self.label_api_key.setVisible(False)
            self.input_api_key.setVisible(False)
            self.label_api_key2.setVisible(True)
            self.input_api_key2.setVisible(True)
        else:
            self.label_api_key.setVisible(False)
            self.input_api_key.setVisible(False)
            self.label_api_key2.setVisible(False)
            self.input_api_key2.setVisible(False)

    def load_config(self):
        default_config = {
            "PROVIDER": "深度求索",
            "MODEL": "deepseek-chat",
            "MAX_HISTORY_MESSAGES": 30,
            "SHORT_TERM_MEMORY_MESSAGES": 16,
            "SUMMARY_HISTORY_LENGTH": 80,
            "MEMORY_DAYS": 7,
            "USE_TRANSLATION": True,
            "USE_COT": True,
            "USE_BETA": False,
            "CHAT_API_KEY": "",
            "CHAT_API_KEY2": "",
            "TRANSLATION_ENGINE": "火山引擎",
            "VOLC_ACCESS_KEY": "",
            "VOLC_SECRET_KEY": ""
        }

        if not os.path.exists(CONFIG_FILE):
            self.config_data = default_config
        else:
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    self.config_data = {**default_config, **loaded_data}
            except json.JSONDecodeError as e:
                QMessageBox.critical(self, "JSON 错误", f"配置文件格式错误：{str(e)}")
                return
            except Exception as e:
                QMessageBox.critical(self, "错误", f"读取配置失败：{str(e)}")
                return

        saved_provider = self.config_data.get("PROVIDER", "深度求索")
        provider_index = self.combo_provider.findText(saved_provider)
        if provider_index >= 0:
            self.combo_provider.setCurrentIndex(provider_index)
        else:
            self.combo_provider.setCurrentIndex(0)

        self.on_provider_changed(self.combo_provider.currentIndex())

        saved_model = self.config_data.get("MODEL", "")
        model_index = self.combo_model.findText(saved_model)
        if model_index >= 0:
            self.combo_model.setCurrentIndex(model_index)
        elif self.combo_model.count() > 0:
            self.combo_model.setCurrentIndex(0)

        self.input_api_key.setText(self.config_data.get("CHAT_API_KEY", ""))
        self.input_api_key2.setText(self.config_data.get("CHAT_API_KEY2", ""))

        self.input_volc_ak.setText(self.config_data.get("VOLC_ACCESS_KEY", ""))
        self.input_volc_sk.setText(self.config_data.get("VOLC_SECRET_KEY", ""))
        saved_trans_engine = self.config_data.get("TRANSLATION_ENGINE", "火山引擎")
        trans_index = self.combo_trans_engine.findText(saved_trans_engine)
        if trans_index >= 0:
            self.combo_trans_engine.setCurrentIndex(trans_index)

        for key, widget in self.widgets.items():
            value = self.config_data.get(key)
            if value is not None:
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                else:
                    widget.setText(str(value))

    def save_config(self):
        new_data = {}
        new_data["PROVIDER"] = self.combo_provider.currentText()
        new_data["MODEL"] = self.combo_model.currentText()
        new_data["CHAT_API_KEY"] = self.input_api_key.text().strip()
        new_data["CHAT_API_KEY2"] = self.input_api_key2.text().strip()

        new_data["TRANSLATION_ENGINE"] = self.combo_trans_engine.currentText()
        new_data["VOLC_ACCESS_KEY"] = self.input_volc_ak.text().strip()
        new_data["VOLC_SECRET_KEY"] = self.input_volc_sk.text().strip()

        for key, widget in self.widgets.items():
            if isinstance(widget, QCheckBox):
                new_data[key] = widget.isChecked()
            else:
                text = widget.text().strip()
                if key not in ["MODEL"]:
                    try:
                        new_data[key] = int(text)
                    except ValueError:
                        if text:
                            QMessageBox.warning(self, "格式错误", f"参数 {key} 需要是整数，当前值为：{text}")
                            return
                        else:
                            new_data[key] = 0
                else:
                    new_data[key] = text

        self.config_data.update(new_data)

        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "成功", "配置已保存到 config.json")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法写入文件：{str(e)}")

if __name__ == "__main__":
    # PyQt6 默认启用了高 DPI 缩放，不再需要设置 AA_EnableHighDpiScaling 和 AA_UseHighDpiPixmaps
    # 这些属性在 PyQt6 中已被移除，直接删除即可。

    app = QApplication(sys.argv)
    font_family = get_platform_font()

    # 全局样式优化
    app.setStyleSheet(f"""
        QWidget {{
            font-family: '{font_family}';
            font-size: 14px;
        }}
        QLabel {{
            color: #303133;
        }}

        /* 1. 输入框与下拉框：去除边框，改为底部分割线 */
        QLineEdit {{
            border: none;
            border-bottom: 1px solid #EBEEF5;
            padding: 8px 2px;
            background-color: transparent;
            selection-background-color: #409EFF;
        }}
        QLineEdit:focus {{
            border-bottom: 2px solid #409EFF; /* 聚焦时加粗变蓝 */
            background-color: transparent;
        }}

        QComboBox {{
            border: none;
            border-bottom: 1px solid #EBEEF5;
            padding: 8px 2px;
            background-color: transparent;
        }}
        QComboBox:focus {{
            border-bottom: 2px solid #409EFF;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox::down-arrow {{
            image: none; /* 隐藏默认箭头图标，或使用自定义图标 */
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #909399; /* CSS绘制下拉箭头 */
            margin-right: 5px;
        }}

        /* 2. 下拉列表悬浮样式修复：解决白字白底问题 */
        QComboBox QAbstractItemView {{
            border: 1px solid #E4E7ED;
            background-color: white;
            selection-background-color: #F5F7FA; /* 悬浮背景改为浅灰 */
            selection-color: #303133; /* 悬浮文字改为深色 */
            outline: none;
        }}

        /* 3. 复选框样式优化：添加灰色边框 */
        QCheckBox {{
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid #DCDFE6; /* 灰色边框 */
            border-radius: 2px;
            background-color: white;
        }}
        QCheckBox::indicator:hover {{
            border-color: #409EFF; /* 悬浮时边框变蓝 */
        }}
        QCheckBox::indicator:checked {{
            background-color: #409EFF;
            border-color: #409EFF;
            /* 使用背景色模拟勾选效果，或加载图片 */
        }}

        /* 滚动条样式 */
        QScrollBar:vertical {{
            border: none;
            background: #F5F7FA;
            width: 8px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: #C0C4CC;
            min-height: 20px;
            border-radius: 4px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
    """)

    window = ConfigWindow()
    window.show()
    # PyQt6 推荐使用 exec() 而不是 exec_()
    sys.exit(app.exec())