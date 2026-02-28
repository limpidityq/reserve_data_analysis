import os
import re
import uuid
import asyncio
import httpx
import xmltodict
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from ai_agent import actuary_file_agent  # 导入你之前的精算逻辑函数

app = FastAPI()

# --- 配置区 ---
CONFIG = {
    "CORP_ID": "ww ",
    "SECRET": "k1 ",
    "AGENT_ID": 1000002,
    "UPLOAD_DIR": "./wecom_cache",
    "BASE_FILE": "保费收入.csv" # 默认文件
}

# 存储用户状态：{user_id: {"file": path, "history": []}}
user_sessions = {}
os.makedirs(CONFIG["UPLOAD_DIR"], exist_ok=True)

# --- 工具：获取企业微信 Token ---
async def get_token():
    url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={CONFIG['CORP_ID']}&corpsecret={CONFIG['SECRET']}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        return r.json().get("access_token")

# --- 工具：下载用户上传的文件 ---
async def download_file(media_id):
    token = await get_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        if res.status_code == 200:
            filename = f"{uuid.uuid4()}.csv"
            path = os.path.join(CONFIG["UPLOAD_DIR"], filename)
            with open(path, "wb") as f:
                f.write(res.content)
            return path
    return None

# --- 工具：主动发送消息回用户 ---
async def send_to_user(user_id, text):
    token = await get_token()
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    payload = {
        "touser": user_id,
        "msgtype": "text",
        "agentid": CONFIG["AGENT_ID"],
        "text": {"content": text}
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload)

# --- 异步精算处理任务 ---
async def async_actuary_task(user_id, query):
    session = user_sessions.get(user_id)
    file_to_use = session.get("file", CONFIG["BASE_FILE"])
    
    # 调用你之前整合了 DuckDB 和 Ollama 的函数
    # 修改 actuary_file_agent 让其在内部 print 之外也返回分析文本
    try:
        status = actuary_file_agent(query, file_to_use, session["history"])
        if status == "SUCCESS":
            result_text = session["history"][-1]["assistant"]
            await send_to_user(user_id, f"✅ 分析完成：\n\n{result_text}")
        else:
            await send_to_user(user_id, f"❌ 处理失败：{status}")
    except Exception as e:
        await send_to_user(user_id, f"❌ 系统异常：{str(e)}")

# --- FastAPI 回调接口 ---
@app.post("/wechat_callback")
async def wechat_callback(request: Request, background_tasks: BackgroundTasks):
    # 1. 接收 XML 数据
    body = await request.body()
    data = xmltodict.parse(body).get("xml")
    
    user_id = data.get("FromUserName")
    msg_type = data.get("MsgType")
    
    # 初始化 Session
    if user_id not in user_sessions:
        user_sessions[user_id] = {"file": CONFIG["BASE_FILE"], "history": []}

    # 2. 处理不同类型的消息
    # A. 用户发送的是文件 (CSV/Parquet)
    if msg_type == "file":
        media_id = data.get("MediaId")
        file_path = await download_file(media_id)
        if file_path:
            user_sessions[user_id]["file"] = file_path
            user_sessions[user_id]["history"] = [] # 换表后必须清空记忆
            await send_to_user(user_id, "📁 收到新数据文件！之前的对话记忆已重置，请开始针对此文件提问。")
        return "success"

    # B. 用户发送的是文字指令
    elif msg_type == "text":
        content = data.get("Content").strip()
        
        if content.lower() == "clear":
            user_sessions[user_id]["history"] = []
            return "success" # 这里可以用 XML 返回或主动推送

        # 核心：异步处理精算逻辑，立即返回避免超时
        background_tasks.add_task(async_actuary_task, user_id, content)
        
        # 立即告知用户已受理
        return xmltodict.unparse({"xml": {
            "ToUserName": user_id,
            "FromUserName": data.get("ToUserName"),
            "CreateTime": data.get("CreateTime"),
            "MsgType": "text",
            "Content": "⏳ 正在查询数据库并生成精算报告，请稍候..."
        }})

    return "success"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)