"""支持流式输出的FastAPI应用"""

import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import MAX_HISTORY_ROUNDS, SERVICE_NAME
from rag_service_optimized import OptimizedRAGService
from schemas import ChatRequest, SessionResponse
from session_store import SessionStore


app = FastAPI(title="罕见病智能机器人", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service: OptimizedRAGService | None = None
session_store = SessionStore(max_rounds=MAX_HISTORY_ROUNDS)


@app.on_event("startup")
def startup_event():
    global rag_service
    rag_service = OptimizedRAGService()


@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.post("/api/chat")
def chat(request: ChatRequest):
    """非流式输出接口（兼容旧版本）"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务尚未初始化完成")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="用户问题不能为空")

    history = session_store.get_recent_history_text(request.session_id)
    result = rag_service.answer(question=question, history=history, debug=request.debug)
    session_store.add_turn(request.session_id, question, result["answer"])

    return {
        "session_id": request.session_id,
        "answer": result["answer"],
        "sources": result["sources"],
        "debug_info": result["debug_info"],
    }


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式输出接口"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务尚未初始化完成")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="用户问题不能为空")

    history = session_store.get_recent_history_text(request.session_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. 意图识别阶段
            yield f"data: {json.dumps({'type': 'status', 'content': '正在分析问题意图...'})}\n\n"

            # 2. 获取完整结果（包含意图识别、检索等）
            result = rag_service.answer_streaming(
                question=question,
                history=history,
                debug=request.debug,
            )

            # 3. 发送调试信息
            if result.get("debug_info"):
                yield f"data: {json.dumps({'type': 'debug', 'content': result['debug_info']})}\n\n"

            # 4. 发送来源信息
            if result.get("sources"):
                yield f"data: {json.dumps({'type': 'sources', 'content': result['sources']})}\n\n"

            # 5. 流式发送回答内容
            answer = result.get("answer", "")
            chunk_size = 10  # 每次发送的字符数
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"

            # 6. 发送完成信号
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

            # 7. 保存到会话历史
            session_store.add_turn(request.session_id, question, answer)

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    return {"session_id": session_id, "messages": session_store.get_messages(session_id)}


@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str):
    session_store.clear(session_id)
    return {"session_id": session_id, "cleared": True}
