const API_BASE = "http://127.0.0.1:8000";

// 窗口管理
let windows = {};
let windowCounter = 0;
let activeWindowId = null;

// DOM元素
const windowTabs = document.getElementById("windowTabs");
const windowsContainer = document.getElementById("windowsContainer");
const addWindowBtn = document.getElementById("addWindowBtn");

// 初始化：创建第一个窗口并绑定添加按钮
document.addEventListener("DOMContentLoaded", () => {
  // 绑定添加窗口按钮
  addWindowBtn.addEventListener("click", () => {
    createNewWindow();
  });

  // 创建第一个窗口
  createNewWindow();
});

// 创建新窗口
function createNewWindow() {
  windowCounter++;
  const windowId = `window-${windowCounter}`;
  const sessionId = `session-${Date.now()}-${windowCounter}`;
  const windowTitle = `会话 ${windowCounter}`;

  // 保存窗口数据
  windows[windowId] = {
    id: windowId,
    sessionId: sessionId,
    title: windowTitle,
  };

  // 创建标签
  const tab = document.createElement("div");
  tab.className = "tab-item";
  tab.dataset.windowId = windowId;
  tab.innerHTML = `
    <span class="tab-title">${windowTitle}</span>
    <button class="tab-close" title="关闭窗口">&times;</button>
  `;
  tab.addEventListener("click", (e) => {
    if (!e.target.classList.contains("tab-close")) {
      switchToWindow(windowId);
    }
  });
  tab.querySelector(".tab-close").addEventListener("click", (e) => {
    e.stopPropagation();
    closeWindow(windowId);
  });
  windowTabs.appendChild(tab);

  // 创建窗口内容
  const windowDiv = document.createElement("div");
  windowDiv.className = "chat-window";
  windowDiv.id = windowId;
  windowDiv.innerHTML = `
    <section class="chat-panel">
      <div class="chat-messages" id="chatMessages-${windowId}">
        <div class="message bot">
          <div class="message-role">机器人</div>
          <div class="message-content">你好，我可以基于罕见病知识库回答科普问题。你可以问：苯丙酮尿症有哪些常见症状？</div>
        </div>
      </div>

      <form class="composer" data-window-id="${windowId}">
        <textarea rows="3" placeholder="请输入你的罕见病科普问题..." id="questionInput-${windowId}"></textarea>
        <div class="composer-actions">
          <label class="debug-toggle">
            <input type="checkbox" checked class="debug-toggle-input" data-window-id="${windowId}">
            返回检索调试信息
          </label>
          <button type="submit">发送</button>
        </div>
      </form>
    </section>

    <aside class="side-panel">
      <section class="card">
        <h2>来源引用</h2>
        <div class="sources-box empty-state" id="sourcesBox-${windowId}">暂无来源。发送问题后会显示检索到的资料来源。</div>
      </section>

      <section class="card">
        <h2>检索调试</h2>
        <div class="debug-box empty-state" id="debugBox-${windowId}">暂无调试信息。</div>
      </section>

      <section class="card compact">
        <h2>服务状态</h2>
        <button class="secondary-button health-button" data-window-id="${windowId}">检查后端服务</button>
        <p class="health-result muted" id="healthResult-${windowId}">尚未检查。</p>
      </section>

      <section class="card">
        <h2>知识库文件</h2>
        <div class="upload-area">
          <input type="file" id="fileInput-${windowId}" multiple accept=".txt,.pdf,.docx,.xlsx,.md" style="display:none">
          <div class="upload-dropzone" id="dropzone-${windowId}">
            <div class="upload-icon">+</div>
            <p>拖拽文件到此处，或 <span class="upload-link">点击选择</span></p>
            <p class="upload-hint">支持 TXT / PDF / Word / Excel，最大 10MB</p>
          </div>
          <div class="upload-progress" id="uploadProgress-${windowId}" style="display:none">
            <div class="progress-bar"><div class="progress-fill" id="progressFill-${windowId}"></div></div>
            <span class="progress-text" id="progressText-${windowId}">上传中...</span>
          </div>
          <div class="upload-result" id="uploadResult-${windowId}" style="display:none"></div>
        </div>
        <div class="file-list" id="fileList-${windowId}">
          <div class="empty-state">暂无上传文件。</div>
        </div>
      </section>
    </aside>
  `;
  windowsContainer.appendChild(windowDiv);

  // 绑定事件
  const form = windowDiv.querySelector("form");
  form.addEventListener("submit", handleSubmit);

  const healthBtn = windowDiv.querySelector(".health-button");
  healthBtn.addEventListener("click", handleHealthCheck);

  // 绑定文件上传事件
  setupUpload(windowId);

  // Enter 发送消息，Shift+Enter 换行
  const textarea = document.getElementById(`questionInput-${windowId}`);
  if (textarea) {
    textarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
      }
    });
  }

  // 切换到新窗口
  switchToWindow(windowId);
}

// 切换窗口
function switchToWindow(windowId) {
  if (!windows[windowId]) return;

  // 更新标签状态
  document.querySelectorAll(".tab-item").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.windowId === windowId);
  });

  // 更新窗口显示
  document.querySelectorAll(".chat-window").forEach((win) => {
    win.classList.toggle("active", win.id === windowId);
  });

  activeWindowId = windowId;
}

// 关闭窗口
function closeWindow(windowId) {
  // 至少保留一个窗口
  if (Object.keys(windows).length <= 1) {
    alert("至少保留一个会话窗口");
    return;
  }

  // 删除标签
  const tab = document.querySelector(`.tab-item[data-window-id="${windowId}"]`);
  if (tab) tab.remove();

  // 删除窗口
  const windowDiv = document.getElementById(windowId);
  if (windowDiv) windowDiv.remove();

  // 删除数据
  delete windows[windowId];

  // 切换到其他窗口
  if (activeWindowId === windowId) {
    const remainingIds = Object.keys(windows);
    if (remainingIds.length > 0) {
      switchToWindow(remainingIds[remainingIds.length - 1]);
    }
  }
}

// 提交问题（流式输出）
async function handleSubmit(event) {
  event.preventDefault();
  const form = event.target;
  const windowId = form.dataset.windowId;
  const windowData = windows[windowId];

  if (!windowData) return;

  const questionInput = document.getElementById(`questionInput-${windowId}`);
  const question = questionInput.value.trim();
  if (!question) {
    appendMessage(windowId, "bot", "请输入问题后再发送。");
    return;
  }

  const debugToggle = form.querySelector(".debug-toggle-input");

  appendMessage(windowId, "user", question);
  questionInput.value = "";

  // 创建用于流式显示的消息元素
  const messageElement = createStreamingMessage(windowId);
  const messageContent = messageElement.querySelector(".message-content");

  try {
    const response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: windowData.sessionId,
        question,
        debug: debugToggle.checked,
      }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullAnswer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
              case "status":
                messageContent.textContent = data.content;
                break;

              case "answer":
                fullAnswer += data.content;
                messageContent.textContent = fullAnswer;
                scrollToBottom(windowId);
                break;

              case "sources":
                renderSources(windowId, data.content);
                break;

              case "debug":
                renderDebug(windowId, data.content);
                break;

              case "error":
                messageContent.textContent = `错误: ${data.content}`;
                break;

              case "done":
                break;
            }
          } catch (e) {
            // 忽略解析错误
          }
        }
      }
    }
  } catch (error) {
    messageContent.textContent = "请求失败，请确认 FastAPI 后端服务已启动。";
  }
}

// 创建流式消息元素
function createStreamingMessage(windowId) {
  const chatMessages = document.getElementById(`chatMessages-${windowId}`);
  if (!chatMessages) return null;

  const message = document.createElement("div");
  message.className = "message bot";
  message.innerHTML = `
    <div class="message-role">机器人</div>
    <div class="message-content">正在思考...</div>
  `;
  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  return message;
}

// 滚动到底部
function scrollToBottom(windowId) {
  const chatMessages = document.getElementById(`chatMessages-${windowId}`);
  if (chatMessages) {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

// 健康检查
async function handleHealthCheck(event) {
  const button = event.target;
  const windowId = button.dataset.windowId;
  const healthResult = document.getElementById(`healthResult-${windowId}`);

  healthResult.textContent = "检查中...";
  try {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    healthResult.textContent = response.ok ? `服务正常：${data.service}` : "服务异常。";
  } catch (error) {
    healthResult.textContent = "无法连接后端服务。";
  }
}

// 追加消息
function appendMessage(windowId, role, content) {
  const chatMessages = document.getElementById(`chatMessages-${windowId}`);
  if (!chatMessages) return;

  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.innerHTML = `
    <div class="message-role">${role === "user" ? "用户" : "机器人"}</div>
    <div class="message-content"></div>
  `;
  message.querySelector(".message-content").textContent = content;
  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 渲染来源
function renderSources(windowId, sources) {
  const sourcesBox = document.getElementById(`sourcesBox-${windowId}`);
  if (!sourcesBox) return;

  if (!sources || sources.length === 0) {
    sourcesBox.className = "sources-box empty-state";
    sourcesBox.textContent = "本次回答没有返回来源。";
    return;
  }

  sourcesBox.className = "sources-box";
  sourcesBox.innerHTML = sources
    .map(
      (source) => `
    <div class="source-item">
      <strong>${source.disease_name_zh || "未知疾病"}</strong><br>
      英文名：${source.disease_name_en || "-"}<br>
      类别：${source.category || "-"}<br>
      来源：${source.source || "-"}，第 ${source.row_index || "-"} 行
    </div>
  `
    )
    .join("");
}

// 渲染调试信息
function renderDebug(windowId, debugInfo) {
  const debugBox = document.getElementById(`debugBox-${windowId}`);
  if (!debugBox) return;

  if (!debugInfo) {
    debugBox.className = "debug-box empty-state";
    debugBox.textContent = "本次请求未开启调试信息。";
    return;
  }

  debugBox.className = "debug-box";
  const docs = debugInfo.retrieved_documents || [];

  // 意图识别信息
  const intentionHtml = debugInfo.intention
    ? `
    <div class="debug-section">
      <strong>意图识别:</strong> ${debugInfo.intention_label || debugInfo.intention}
      ${debugInfo.rag_called ? "(已调用RAG)" : "(未调用RAG)"}
      ${debugInfo.knowledge_source ? `<br><strong>知识来源:</strong> ${debugInfo.knowledge_source}` : ""}
    </div>
  `
    : "";

  // 模型使用信息
  let modelsHtml = "";
  if (debugInfo.models_used) {
    const m = debugInfo.models_used;
    modelsHtml = `
      <div class="debug-section">
        <strong>模型调用:</strong><br>
        意图识别: ${m.intent_recognition || "-"}<br>
        Rerank: ${m.rerank || "-"}<br>
        RAG生成: ${m.rag_generation || "-"}
      </div>
    `;
  }

  // 结构化查询信息
  let structuredQueryHtml = "";
  if (debugInfo.structured_query) {
    const sq = debugInfo.structured_query;
    structuredQueryHtml = `
      <div class="debug-section">
        <strong>结构化查询:</strong><br>
        疾病名: ${sq.disease_name || "未识别"}<br>
        查询类型: ${sq.query_type || "未识别"}<br>
        关键词: ${(sq.keywords || []).join(", ") || "无"}
      </div>
    `;
  }

  // 检索信息
  const retrievalHtml = debugInfo.rag_called
    ? `
    <div class="debug-section">
      <strong>检索信息:</strong> 初始${debugInfo.initial_retrieved || 0}条 → Rerank后${debugInfo.reranked_to || debugInfo.retrieved_count || 0}条
    </div>
  `
    : "";

  // 检索到的文档
  const docsHtml =
    docs.length > 0
      ? `
    <div class="debug-section">
      <strong>检索到的文档:</strong>
      ${docs
        .map(
          (doc) => `
        <div class="debug-item">
          <strong>${doc.metadata.disease_name_zh || "未知疾病"}</strong>
          <span class="debug-meta">(${doc.metadata.index_type || "primary"})</span><br>
          ${doc.content_preview || ""}
        </div>
      `
        )
        .join("")}
    </div>
  `
      : "";

  debugBox.innerHTML = intentionHtml + modelsHtml + structuredQueryHtml + retrievalHtml + docsHtml;
}

// ==================== 文件上传功能 ====================

function setupUpload(windowId) {
  const fileInput = document.getElementById(`fileInput-${windowId}`);
  const dropzone = document.getElementById(`dropzone-${windowId}`);

  if (!fileInput || !dropzone) return;

  // 点击选择文件
  dropzone.addEventListener("click", () => fileInput.click());

  // 文件选择
  fileInput.addEventListener("change", (e) => {
    handleFileUpload(windowId, e.target.files);
  });

  // 拖拽
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    handleFileUpload(windowId, e.dataTransfer.files);
  });

  // 加载已有文件列表
  loadFileList(windowId);
}

async function handleFileUpload(windowId, files) {
  if (!files || files.length === 0) return;

  const progress = document.getElementById(`uploadProgress-${windowId}`);
  const progressFill = document.getElementById(`progressFill-${windowId}`);
  const progressText = document.getElementById(`progressText-${windowId}`);
  const resultDiv = document.getElementById(`uploadResult-${windowId}`);

  for (const file of files) {
    // 检查文件大小
    if (file.size > 10 * 1024 * 1024) {
      showUploadResult(windowId, `文件 ${file.name} 超过10MB限制`, false);
      continue;
    }

    // 显示进度
    progress.style.display = "flex";
    progressFill.style.width = "0%";
    progressText.textContent = `正在上传 ${file.name}...`;
    resultDiv.style.display = "none";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("category", "用户上传");

    try {
      // 模拟进度
      let pct = 0;
      const interval = setInterval(() => {
        pct = Math.min(pct + Math.random() * 20, 90);
        progressFill.style.width = pct + "%";
      }, 300);

      const response = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });

      clearInterval(interval);
      progressFill.style.width = "100%";

      const data = await response.json();

      if (response.ok && data.success) {
        showUploadResult(windowId, `上传成功：${data.filename}，解析为 ${data.chunk_count} 个片段`, true);
        loadFileList(windowId);
      } else {
        showUploadResult(windowId, data.detail || data.error || "上传失败", false);
      }
    } catch (err) {
      showUploadResult(windowId, `上传失败：${err.message}`, false);
    }

    setTimeout(() => { progress.style.display = "none"; }, 1500);
  }

  // 清空 input
  const fileInput = document.getElementById(`fileInput-${windowId}`);
  if (fileInput) fileInput.value = "";
}

function showUploadResult(windowId, message, success) {
  const resultDiv = document.getElementById(`uploadResult-${windowId}`);
  if (!resultDiv) return;
  resultDiv.style.display = "block";
  resultDiv.className = `upload-result ${success ? "upload-success" : "upload-error"}`;
  resultDiv.textContent = message;
}

async function loadFileList(windowId) {
  const fileList = document.getElementById(`fileList-${windowId}`);
  if (!fileList) return;

  try {
    const response = await fetch(`${API_BASE}/api/files`);
    const data = await response.json();
    const files = data.files || [];

    if (files.length === 0) {
      fileList.innerHTML = '<div class="empty-state">暂无上传文件。</div>';
      return;
    }

    fileList.innerHTML = files
      .map(
        (f) => `
      <div class="file-item">
        <div class="file-info">
          <span class="file-name" title="${f.filename}">${f.filename}</span>
          <span class="file-meta">${f.file_type} · ${formatFileSize(f.file_size)} · ${f.chunk_count}片段</span>
        </div>
        <button class="file-delete-btn" onclick="deleteFile('${windowId}', '${f.file_id}')" title="删除">&times;</button>
      </div>
    `
      )
      .join("");
  } catch (err) {
    fileList.innerHTML = '<div class="empty-state">加载文件列表失败。</div>';
  }
}

async function deleteFile(windowId, fileId) {
  if (!confirm("确定要删除该文件吗？")) return;

  try {
    const response = await fetch(`${API_BASE}/api/files/${fileId}`, { method: "DELETE" });
    const data = await response.json();
    if (data.success) {
      loadFileList(windowId);
    } else {
      alert(data.error || "删除失败");
    }
  } catch (err) {
    alert("删除失败：" + err.message);
  }
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + "B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + "KB";
  return (bytes / (1024 * 1024)).toFixed(1) + "MB";
}
