from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import uuid
import os
import base64
import requests
import json
import time
import tempfile

app = Flask(__name__)
CORS(app)

# ==================== 重要安全提醒 ====================
# 您的API密钥已暴露在网上，请立即撤销并更换！
# 不要将密钥硬编码在代码中，使用环境变量

# API配置 - 改为从环境变量获取
BAIDU_API_KEY = os.environ.get('BAIDU_API_KEY')
BAIDU_SECRET_KEY = os.environ.get('BAIDU_SECRET_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')

# 如果没有设置环境变量，则在启动时显示错误
if not all([BAIDU_API_KEY, BAIDU_SECRET_KEY, DEEPSEEK_API_KEY]):
    print("警告: 请在Render中设置以下环境变量:")
    print("1. BAIDU_API_KEY")
    print("2. BAIDU_SECRET_KEY")
    print("3. DEEPSEEK_API_KEY")
    print("否则应用将无法正常工作！")

# 百度OCR配置
BAIDU_OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
BAIDU_AUTH_URL = "https://aip.baidubce.com/oauth/2.0/token"

# DeepSeek API配置 (已更正为正确的URL)
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

class BaiduOCR:
    """百度OCR服务类"""
    
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = None
        self.token_expires = None
    
    def get_access_token(self):
        """获取百度OCR访问令牌"""
        try:
            if not self.api_key or not self.secret_key:
                raise Exception("百度OCR API密钥未配置")
                
            params = {
                'grant_type': 'client_credentials',
                'client_id': self.api_key,
                'client_secret': self.secret_key
            }
            
            response = requests.post(BAIDU_AUTH_URL, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if 'access_token' in result:
                self.access_token = result['access_token']
                # 令牌有效期通常为30天
                self.token_expires = time.time() + result.get('expires_in', 2592000)
                print(f"成功获取百度OCR令牌")
                return self.access_token
            elif 'error' in result:
                raise Exception(f"获取token失败: {result.get('error_description', result['error'])}")
            else:
                raise Exception(f"获取token失败: {result}")
                
        except requests.exceptions.Timeout:
            raise Exception("获取百度OCR令牌超时")
        except Exception as e:
            raise Exception(f"获取百度OCR访问令牌失败: {str(e)}")
    
    def ocr_image(self, image_path, retry_count=2):
        """对图片进行OCR识别"""
        for attempt in range(retry_count):
            try:
                # 检查API密钥
                if not self.api_key or not self.secret_key:
                    raise Exception("百度OCR API密钥未配置")
                
                # 如果token不存在或已过期，重新获取
                if not self.access_token or (self.token_expires and time.time() > self.token_expires):
                    self.get_access_token()
                
                # 读取并编码图片
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                
                if len(image_data) == 0:
                    raise Exception("图片文件为空")
                
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                if len(image_base64) < 100:
                    raise Exception("图片文件过小或无效")
                
                # 准备请求参数
                headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                params = {'access_token': self.access_token}
                data = {'image': image_base64}
                
                # 调用OCR API
                response = requests.post(BAIDU_OCR_URL, params=params, headers=headers, data=data, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # 处理错误码
                if 'error_code' in result:
                    error_msg = result.get('error_msg', '未知错误')
                    print(f"百度OCR返回错误: {error_msg}")
                    
                    # 如果是token问题，刷新token后重试
                    if result['error_code'] in [110, 111] and attempt < retry_count - 1:
                        print(f"令牌无效，重新获取后重试...")
                        self.get_access_token()
                        continue
                    else:
                        raise Exception(f"OCR识别失败: {error_msg}")
                
                # 提取识别结果
                if 'words_result' in result:
                    text_lines = [item['words'] for item in result['words_result']]
                    ocr_text = '\n'.join(text_lines)
                    print(f"OCR识别成功，识别到{len(ocr_text)}个字符")
                    return ocr_text
                else:
                    raise Exception(f"OCR识别失败: {result}")
                    
            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    print(f"OCR请求超时，重试 {attempt + 1}/{retry_count}...")
                    time.sleep(1)
                    continue
                else:
                    raise Exception("OCR请求超时，请稍后重试")
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"OCR识别失败，重试 {attempt + 1}/{retry_count}...")
                    time.sleep(1)
                    continue
                else:
                    raise Exception(f"OCR处理失败: {str(e)}")

class DeepSeekAI:
    """DeepSeek AI服务类"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = DEEPSEEK_API_URL
    
    def analyze_question(self, question_text, grade, subject, retry_count=2):
        """使用DeepSeek分析题目"""
        for attempt in range(retry_count):
            try:
                # 检查API密钥
                if not self.api_key:
                    raise Exception("DeepSeek API密钥未配置")
                
                # 构建系统提示词
                system_prompt = f"""你是一位经验丰富的{grade}{subject}老师。请分析以下作业题目，并提供详细的解题步骤和答案。
                
                要求：
                1. 分析题目考察的知识点
                2. 提供详细的解题步骤
                3. 给出最终答案
                4. 用中文回答，保持专业和清晰
                5. 如果题目涉及数学公式，请使用LaTeX格式表示
                
                年级：{grade}
                科目：{subject}"""
                
                # 构建用户消息
                user_message = f"""请分析以下题目：
                
                {question_text}
                
                请按照要求提供完整的解析。"""
                
                # 准备请求数据
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                    "stream": False
                }
                
                print(f"正在调用DeepSeek API (尝试 {attempt + 1}/{retry_count})...")
                
                # 调用DeepSeek API
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=data, 
                    timeout=60,
                    verify=True
                )
                
                print(f"DeepSeek API响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 提取AI回复
                    if 'choices' in result and len(result['choices']) > 0:
                        ai_answer = result['choices'][0]['message']['content']
                        print(f"DeepSeek API调用成功，返回字符数: {len(ai_answer)}")
                        return ai_answer
                    else:
                        print(f"DeepSeek API返回异常: {result}")
                        raise Exception(f"AI解析失败: {result}")
                else:
                    error_text = response.text
                    print(f"DeepSeek API错误响应: {error_text}")
                    
                    # 尝试解析错误信息
                    try:
                        error_json = response.json()
                        error_msg = error_json.get('error', {}).get('message', error_text)
                    except:
                        error_msg = error_text
                    
                    # 如果是速率限制，等待后重试
                    if response.status_code == 429 and attempt < retry_count - 1:
                        wait_time = min(30, (attempt + 1) * 10)  # 最多等待30秒
                        print(f"API速率限制，等待{wait_time}秒后重试...")
                        time.sleep(wait_time)
                        continue
                    
                    raise Exception(f"AI分析失败 (状态码 {response.status_code}): {error_msg}")
                    
            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    print(f"DeepSeek API请求超时，重试 {attempt + 1}/{retry_count}...")
                    time.sleep(2)
                    continue
                else:
                    raise Exception("DeepSeek API请求超时，请稍后重试")
            except requests.exceptions.ConnectionError:
                if attempt < retry_count - 1:
                    print(f"DeepSeek API连接错误，重试 {attempt + 1}/{retry_count}...")
                    time.sleep(2)
                    continue
                else:
                    raise Exception("无法连接到DeepSeek API，请检查网络连接")
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"DeepSeek API调用失败，重试 {attempt + 1}/{retry_count}: {str(e)}")
                    time.sleep(2)
                    continue
                else:
                    raise Exception(f"AI分析失败: {str(e)}")

# 初始化服务
baidu_ocr = BaiduOCR(BAIDU_API_KEY, BAIDU_SECRET_KEY)
deepseek_ai = DeepSeekAI(DEEPSEEK_API_KEY)

@app.route('/')
def home():
    """首页"""
    return jsonify({
        "service": "作业助手API",
        "version": "1.0.0",
        "status": "运行正常" if all([BAIDU_API_KEY, BAIDU_SECRET_KEY, DEEPSEEK_API_KEY]) else "API密钥未配置",
        "endpoints": {
            "POST /api/process-homework": "处理作业图片",
            "GET /api/test": "测试接口",
            "GET /api/health": "健康检查",
            "POST /api/test-ocr": "测试OCR功能",
            "POST /api/test-ai": "测试AI功能"
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/process-homework', methods=['POST'])
def process_homework():
    """处理作业图片的API接口"""
    start_time = datetime.now()
    
    try:
        # 检查API密钥
        if not BAIDU_API_KEY or not BAIDU_SECRET_KEY:
            return jsonify({
                "success": False,
                "error": "百度OCR服务未配置"
            }), 503
        
        if not DEEPSEEK_API_KEY:
            return jsonify({
                "success": False,
                "error": "DeepSeek AI服务未配置"
            }), 503
        
        # 获取表单数据
        grade = request.form.get('grade', '高中')
        subject = request.form.get('subject', '数学')
        
        # 检查是否有图片文件
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "没有上传图片文件"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "没有选择文件"
            }), 400
        
        # 检查文件类型
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                "success": False,
                "error": f"不支持的文件类型 {file_ext}，请上传图片文件"
            }), 400
        
        # 检查文件大小（Render免费计划限制100MB，但建议限制更小）
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置指针
        
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                "success": False,
                "error": f"文件太大，最大支持10MB"
            }), 400
        
        # 使用临时文件处理（Render的磁盘是临时的）
        file_id = str(uuid.uuid4())
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            file.save(tmp.name)
            filepath = tmp.name
        
        print(f"文件已保存到临时文件: {filepath}")
        
        try:
            # 步骤1: 调用百度OCR识别文本
            print("开始OCR识别...")
            ocr_text = baidu_ocr.ocr_image(filepath)
            print(f"OCR识别完成，识别到{len(ocr_text)}个字符")
            
            if not ocr_text or len(ocr_text.strip()) < 5:
                return jsonify({
                    "success": False,
                    "error": "图片中未识别到有效文字，请上传清晰的作业图片"
                }), 400
            
            # 步骤2: 调用DeepSeek分析题目
            print("开始AI分析...")
            ai_answer = deepseek_ai.analyze_question(ocr_text, grade, subject)
            print("AI分析完成")
            
            # 计算处理时间
            end_time = datetime.now()
            processing_time = round((end_time - start_time).total_seconds(), 2)
            
            # 返回结果
            return jsonify({
                "success": True,
                "message": "处理成功",
                "data": {
                    "id": file_id,
                    "grade": grade,
                    "subject": subject,
                    "ocr_text": ocr_text,
                    "ai_answer": ai_answer,
                    "processing_time": f"{processing_time}秒",
                    "created_at": datetime.now().isoformat()
                }
            })
            
        finally:
            # 清理临时文件
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"已清理临时文件: {filepath}")
                except:
                    print(f"清理临时文件失败: {filepath}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"处理失败: {error_msg}")
        
        return jsonify({
            "success": False,
            "error": f"处理失败: {error_msg}",
            "suggestion": "请检查网络连接或稍后重试"
        }), 500

@app.route('/api/test', methods=['GET'])
def test_api():
    """测试接口"""
    return jsonify({
        "success": True,
        "message": "服务器运行正常",
        "data": {
            "status": "running",
            "services": {
                "baidu_ocr": "已配置" if BAIDU_API_KEY and BAIDU_SECRET_KEY else "未配置",
                "deepseek_ai": "已配置" if DEEPSEEK_API_KEY else "未配置"
            },
            "timestamp": datetime.now().isoformat()
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 测试百度OCR连接
        if BAIDU_API_KEY and BAIDU_SECRET_KEY:
            baidu_ocr.get_access_token()
            baidu_status = "available"
        else:
            baidu_status = "unavailable: API密钥未配置"
    except Exception as e:
        baidu_status = f"unavailable: {str(e)[:100]}"
    
    try:
        # 测试DeepSeek连接
        if DEEPSEEK_API_KEY:
            # 只发送一个简单的测试请求
            test_response = deepseek_ai.analyze_question("测试", "测试", "测试", retry_count=1)
            deepseek_status = "available"
        else:
            deepseek_status = "unavailable: API密钥未配置"
    except Exception as e:
        deepseek_status = f"unavailable: {str(e)[:100]}"
    
    status = "healthy" if "available" in baidu_status and "available" in deepseek_status else "degraded"
    
    return jsonify({
        "status": status,
        "services": {
            "baidu_ocr": baidu_status,
            "deepseek_ai": deepseek_status
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/test-ocr', methods=['POST'])
def test_ocr():
    """测试OCR功能"""
    try:
        if not BAIDU_API_KEY or not BAIDU_SECRET_KEY:
            return jsonify({
                "success": False,
                "error": "百度OCR服务未配置"
            }), 503
        
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "没有上传图片文件"
            }), 400
        
        file = request.files['image']
        
        # 使用临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            filepath = tmp.name
        
        try:
            ocr_text = baidu_ocr.ocr_image(filepath, retry_count=1)
            
            return jsonify({
                "success": True,
                "message": "OCR测试成功",
                "data": {
                    "text": ocr_text,
                    "char_count": len(ocr_text)
                }
            })
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"OCR测试失败: {str(e)}"
        }), 500

@app.route('/api/test-ai', methods=['POST'])
def test_ai():
    """测试AI功能"""
    try:
        if not DEEPSEEK_API_KEY:
            return jsonify({
                "success": False,
                "error": "DeepSeek AI服务未配置"
            }), 503
        
        data = request.json
        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "error": "请提供question参数"
            }), 400
        
        question = data.get('question', '')
        grade = data.get('grade', '高中')
        subject = data.get('subject', '数学')
        
        ai_answer = deepseek_ai.analyze_question(question, grade, subject, retry_count=1)
        
        return jsonify({
            "success": True,
            "message": "AI测试成功",
            "data": {
                "question": question,
                "answer": ai_answer
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"AI测试失败: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("作业助手服务器启动...")
    print("=" * 60)
    
    # 检查API密钥配置
    api_keys_configured = all([BAIDU_API_KEY, BAIDU_SECRET_KEY, DEEPSEEK_API_KEY])
    
    if not api_keys_configured:
        print("警告: API密钥未完全配置！")
        print("请在Render中设置以下环境变量:")
        print("1. BAIDU_API_KEY")
        print("2. BAIDU_SECRET_KEY")
        print("3. DEEPSEEK_API_KEY")
        print("否则部分功能将无法使用")
    else:
        print("✓ 所有API密钥已配置")
    
    print("\n服务状态:")
    
    # 测试百度OCR连接
    if BAIDU_API_KEY and BAIDU_SECRET_KEY:
        try:
            token = baidu_ocr.get_access_token()
            print(f"✓ 百度OCR: 连接成功 (token: {token[:20]}...)")
        except Exception as e:
            print(f"✗ 百度OCR: 连接失败 - {str(e)[:100]}")
    else:
        print("✗ 百度OCR: 未配置")
    
    # 测试DeepSeek连接
    if DEEPSEEK_API_KEY:
        try:
            # 只发送一个简单的测试请求
            test_response = deepseek_ai.analyze_question("测试连接", "测试", "测试", retry_count=1)
            print(f"✓ DeepSeek: 连接成功")
        except Exception as e:
            print(f"✗ DeepSeek: 连接失败 - {str(e)[:100]}")
    else:
        print("✗ DeepSeek: 未配置")
    
    print("\nAPI接口:")
    print("  GET  /              - 首页")
    print("  POST /api/process-homework - 处理作业图片")
    print("  GET  /api/test      - 测试接口")
    print("  GET  /api/health    - 健康检查")
    print("  POST /api/test-ocr  - 测试OCR功能")
    print("  POST /api/test-ai   - 测试AI功能")
    
    print("\n部署信息:")
    print("  Render会自动设置PORT环境变量")
    print("  使用gunicorn作为生产服务器")
    print("=" * 60)
    
    # 获取端口（Render会设置PORT环境变量）
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)