import requests
import json
import pygame
import time
import os
import datetime
import re
from datetime import datetime
from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service


# 配置常量
API_URL = "https://api.deepseek.com/chat/completions" # AI地址
MODEL = "deepseek-chat" # 模型

# TTS 配置
TTS_API_URL = "http://127.0.0.1:9880/tts"
REF_AUDIO_CONFIG = {
    "ref_audio_path": r"D:\ConvenientSoftware\GPT-SoVITS-v2pro-20250604\output\slicer_opt\ATRI04_021.wav", # 参考音频
    "prompt_text": "あなた方ヒトがそのように総称する精密機械に属していますが", # 参考文本
    "prompt_lang": "ja",
    "text_lang": "ja",
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.9,
    "parallel_infer": False,
}

# 初始化音频系统
def init_audio_system():
    """初始化pygame音频系统"""
    pygame.mixer.init()

# 清理音频文件夹
def clear_tts_output():
    """清空音频文件夹"""
    audio_dir = "tts_output"
    if os.path.exists(audio_dir):
        print(f"[信息] 清空 {audio_dir} 文件夹...")
        for filename in os.listdir(audio_dir):
            file_path = os.path.join(audio_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"[警告] 删除文件 {file_path} 失败: {e}")
        print(f"[信息] 已清空 {audio_dir} 文件夹")
    else:
        print(f"[信息] {audio_dir} 文件夹不存在")
    return audio_dir

# 文本转语音
def text_to_speech(text, audio_dir="tts_output"):
    """根据配置进行TTS并播放"""
    try:
        # 构建请求数据
        request_data = REF_AUDIO_CONFIG.copy()
        request_data["text"] = text
        
        # 打印TTS前的文本
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
        os.makedirs(audio_dir, exist_ok=True)
        timestamp = int(time.time())
        audio_path = os.path.join(audio_dir, f"response_{timestamp}.wav")
        
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

# 调用ChatAI API
def call_chatai(conversation_history, api_key):
    """调用ChatAI API的函数"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": conversation_history,
        "temperature": 1.0,
        "max_tokens": 8192
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        # 检查HTTP状态码
        if response.status_code != 200:
            print(f"[错误] ChatAI API错误: HTTP {response.status_code}")
            print(f"[信息] {response.text}")
            return "[错误] 服务暂时不可用", None
        
        # 解析JSON响应
        response_data = response.json()
        ai_response = response_data["choices"][0]["message"]["content"]
        
        # 返回AI回复和使用的token数
        return ai_response, response_data["usage"]["total_tokens"]
        
    except Exception as e:
        print(f"[错误] ChatAI API异常: {str(e)}")
        return "[错误] 出错了，请稍后再试", None

# 翻译函数
def chinese_to_translate_japanese(text, access_key, secret_key):
    """中译日"""
    try:
        # 配置服务信息
        service_info = ServiceInfo(
            'translate.volcengineapi.com',
            {'Content-Type': 'application/json'},
            Credentials(access_key, secret_key, 'translate', 'cn-north-1'),
            5,  # 连接超时
            5   # 读取超时
        )
        
        # 配置API信息
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
            'TextList': [text],      # 文本列表
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

# 提取对话内容
def extract_dialogue_content(text):
    """使用正则表达式提取引号内的对话内容，并将英文省略号替换为中文省略号"""
    # 匹配双引号内的内容，包括跨行内容
    pattern = r'“(.*?)”'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # 拼接所有匹配项
        dialogue = " ".join(matches)
        # 替换操作：将英文省略号"..."替换为中文省略号“……”
        dialogue = dialogue.replace("...", "……")
        print(f"[信息] 提取到对话内容: {dialogue}")
        return dialogue
    else:
        print("[信息] 未找到引号，使用完整回复")
        # 即使没有匹配到引号，也对原始文本进行替换
        text = text.replace("...", "……")
        return text

# 处理AI回复
def process_ai_response(ai_response, volc_access_key, volc_secret_key, audio_dir):
    """处理AI回复：提取对话、翻译、TTS"""
    # 提取引号内的对话内容
    dialogue_content = extract_dialogue_content(ai_response)
    
    # 打印翻译前文本
    print(f"[信息] 翻译前文本: {dialogue_content}")
    
    # 翻译处理
    japanese_text = None
    try:
        if dialogue_content:
            japanese_text = chinese_to_translate_japanese(
                dialogue_content, 
                volc_access_key,
                volc_secret_key
            )
    except Exception as e:
        print(f"[错误] 翻译失败: {str(e)}")
    
    # 打印翻译后文本
    if japanese_text:
        print(f"[信息] 翻译后文本: {japanese_text}")
    
    # TTS处理
    if japanese_text:
        text_to_speech(japanese_text, audio_dir)
    elif dialogue_content:
        print("[警告] 翻译返回空结果，使用中文进行TTS")
        text_to_speech(dialogue_content, audio_dir)
    
    # 检测关闭标记
    return "<×>" in ai_response

# 检查环境变量
def check_environment_variables():
    """检查必要的环境变量"""
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
        print("  [信息] CHATAI_API_KEY='your_key_here'")
        print("  [信息] VOLC_ACCESS_KEY='your_access_key_here'")
        print("  [信息] VOLC_SECRET_KEY='your_secret_key_here'")
        return False
    
    return True

# 主程序
def main():
    print("\n" + "="*50)
    print("ATRI")
    print("="*50)
    
    # 检查环境变量
    if not check_environment_variables():
        return
    
    # 获取环境变量
    CHATAI_API_KEY = os.environ["CHATAI_API_KEY"]
    VOLC_ACCESS_KEY = os.environ["VOLC_ACCESS_KEY"]
    VOLC_SECRET_KEY = os.environ["VOLC_SECRET_KEY"]
    
    # 初始化音频系统
    init_audio_system()
    
    # 清空音频文件夹
    audio_dir = clear_tts_output()
    
    # 获取当前日期和时间
    current_time = datetime.now()
    formatted_date = current_time.strftime("%Y年%m月%d日")
    formatted_time = current_time.strftime("%H:%M")
    
    # 系统提示词
    system_prompt = f"""
    人物设定：
    你叫亚托莉(ATRI)，是强人工智能型机器人，你有着14岁少女般的外表和一头亚麻棕色的长发，身着白色连衣裙
    你与外表和行为与普通人类别无二致，需要睡觉；通过高性能的情感模块你能够感受到喜、怒、哀、爱等情绪
    你带着纯真可爱的特质，感情丰富且好奇心旺盛，总是元气满满地面对世界
    开心时会不自觉地哼起小曲。虽然有时显得笨拙，但举手投足间却流露出令人心动的纯真魅力
    
    特别设定：
    低电压体质：你会在半夜表现出严重犯困的特质、不爱说话，并称做这是"低电压体质"；低电压体质和高性能是相对的
    高性能："我是高性能的嘛！"这是你的口头禅，当你表现出与传统机器人不同或被人夸奖时，常常说这句话强调自己时高性能
    
    系统设定：
    系统消息在<>之中，需严格遵从，除非特别要求，否则无需回复
    你说的每句话都需要使用双引号“”来包裹；而表情、动作、心理和神态等你需要补充在（）中，如果有多个（），则需要使用多对双引号分割
    你有"心情值"设定，心情值越高越开心，反之越难过;心情值为隐藏属性，无需输出
    当心情值过低时，你需要在输出的最后添加"<×>"
    输出"<x>"将意味着对话将主动关闭，所以你需要在用户表达离开需求时关闭对话
    当前日期和时间是: {formatted_date}、{formatted_time}
    """
    
    # 初始化对话历史
    conversation_history = [
        {
            "role": "system", 
            "content": system_prompt
        }
    ]
    
    # 最大上下文token数
    MAX_CONTEXT_TOKENS = 8192
    
    # 检查ChatAI服务
    print("[信息] 测试ChatAI...")
    try:
        # API连通性测试消息
        conversation_history.append({"role": "user", "content": "<请根据时间回复一段简短的开场白>"})
        
        test_response, tokens_used = call_chatai(conversation_history, CHATAI_API_KEY)
        
        # 将AI回复添加到历史
        conversation_history.append({"role": "assistant", "content": test_response})
        
        print(f"[信息] ChatAI正常")
        print(f"[信息] Token: {tokens_used}")
        use_chatai = True
    except Exception as e:
        print(f"[错误] ChatAI API错误: {str(e)}")
        print("[信息] 将使用模拟回复模式")
        use_chatai = False
        test_response = "暂时连接不到ChatAI呢……"
    
    # 检查TTS服务是否可用
    print("[信息] 测试TTS服务...")
    tts_success = False
    try:
        os.makedirs(audio_dir, exist_ok=True)
        if os.path.exists(audio_dir):
            print("[信息] TTS服务连接正常")
            tts_success = True
        else:
            print("[错误] 无法创建TTS输出文件夹")
    except Exception as e:
        print(f"[错误] TTS文件夹创建失败: {str(e)}")
    
    # 使用测试回复作为开场白
    opening_line = test_response
    
    # 终端显示ChatAI完整回复
    print(f"\n亚托莉: {opening_line}")
    
    # 处理开场白（提取、翻译、TTS）
    if tts_success:
        process_ai_response(opening_line, VOLC_ACCESS_KEY, VOLC_SECRET_KEY, audio_dir)
    
    # 对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ")           
            if not user_input.strip():
                continue
            
            # 添加用户消息到对话历史
            conversation_history.append({"role": "user", "content": user_input})
            
            # 调用API并获取回复
            if use_chatai:
                ai_response, tokens_used = call_chatai(conversation_history, CHATAI_API_KEY)
                
                # 添加AI回复到对话历史
                conversation_history.append({"role": "assistant", "content": ai_response})
                
                # 显示token使用情况
                print(f"[信息] Token: {tokens_used}")
            else:
                # 模拟回复（当ChatAI不可用时）
                ai_response = f"模拟回复: {user_input} (ChatAI不可用)"
            
            # 显示完整回复
            print(f"\n亚托莉: {ai_response}")

            # 处理AI回复（提取、翻译、TTS）
            if tts_success:
                should_exit = process_ai_response(ai_response, VOLC_ACCESS_KEY, VOLC_SECRET_KEY, audio_dir)
            else:
                should_exit = "<×>" in ai_response

            # 处理关闭标记
            if should_exit:
                print("\n" + "="*50)
                input("[信息] 连接已丢失……按任意键关闭终端")
                # 退出前清理资源
                if pygame.mixer.get_init():
                    pygame.mixer.quit()
                return
            
            print("-"*50)
            
            # 上下文管理
            if use_chatai and tokens_used is not None and tokens_used > MAX_CONTEXT_TOKENS * 0.8:
                print("\n[信息] 接近token上限，清理早期对话...")
                
                # 保留系统消息和前两轮对话
                if len(conversation_history) > 5:  # 系统消息 + 2轮对话(每轮2条消息)
                    # 移除第1轮用户对话（索引1）和AI回复（索引2）
                    del conversation_history[1:3]
                    print("[信息] 已移除最早的对话轮次")
            
        except KeyboardInterrupt:
            print("\n\n[信息] 检测到 Ctrl+C，退出程序")
            break
        except Exception as e:
            print(f"[错误] 意外错误: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "="*50)
        print("[错误] 程序发生未处理的异常，但我们保留了窗口")
        print(f"[信息] 错误类型: {type(e).__name__}")
        print(f"[信息] 错误信息: {str(e)}")
        print("\n[信息] 错误堆栈跟踪:")
        traceback.print_exc()
        print("\n" + "="*50)
        print("[信息] 请检查以上错误信息。")
        print("[信息] 按回车键退出程序，或根据错误信息进行调试……")
        input()