from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import MAX_HISTORY_ROUNDS, SERVICE_NAME
from rag_service_optimized import OptimizedRAGService as RAGService
from schemas import ChatRequest, ChatResponse, SessionResponse
from session_store import SessionStore

import json
from typing import AsyncGenerator


app = FastAPI(title="罕见病智能机器人", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service: RAGService | None = None
session_store = SessionStore(max_rounds=MAX_HISTORY_ROUNDS)


@app.on_event("startup")
def startup_event():
    global rag_service
    rag_service = RAGService()


@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
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
        full_answer = ""
        try:
            for event in rag_service.answer_streaming(
                question=question,
                history=history,
                debug=request.debug,
            ):
                # 收集完整回答用于保存历史
                if event["type"] == "answer":
                    full_answer += event["content"]

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # 保存会话历史
            if full_answer:
                session_store.add_turn(request.session_id, question, full_answer)

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

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


# ==================== 文件上传和管理接口 ====================

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form(default="用户上传"),
):
    """上传文件到知识库"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务尚未初始化完成")

    # 读取文件内容
    content = await file.read()

    # 检查文件大小（限制10MB）
    max_size = 10 * 1024 * 1024  # 10MB
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="文件大小超过限制（最大10MB）")

    # 上传并添加到知识库
    result = rag_service.upload_and_add_to_knowledge_base(
        file_content=content,
        filename=file.filename,
        category=category,
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "上传失败"))

    return result


@app.get("/api/files")
def list_files():
    """获取已上传的文件列表"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务尚未初始化完成")

    files = rag_service.get_uploaded_files()
    return {"files": files, "total": len(files)}


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str):
    """删除已上传的文件"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 服务尚未初始化完成")

    result = rag_service.delete_uploaded_file(file_id)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "删除失败"))

    return result


@app.get("/api/files/supported-formats")
def get_supported_formats():
    """获取支持的文件格式"""
    from file_parser import FileParser
    parser = FileParser()
    return {"formats": parser.get_supported_formats()}
