const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

const API_BASE = 'http://localhost:8000';

// Global function for test cases
window.useExample = (text) => {
    console.log('Using example:', text);
    userInput.value = text;
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
    sendMessage();
};

// Auto-resize textarea
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
});

// Send message on Enter (but allow Shift+Enter)
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add user message to UI
    appendMessage('user', text);
    userInput.value = '';
    userInput.style.height = 'auto';

    // Add AI loading message
    const aiMsgDiv = appendMessage('ai', '思考中...', true);

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: text, mode: 'hybrid' })
        });

        if (!response.ok) throw new Error('从 RAG 服务获取回复失败');

        const data = await response.json();

        // Update AI message with typing effect
        typeMessage(aiMsgDiv, data.answer);
    } catch (error) {
        console.error('Error:', error);
        aiMsgDiv.textContent = '抱歉，我遇到了错误。请确保后端服务已启动。';
        aiMsgDiv.classList.remove('typing');
    }
}

function appendMessage(role, text, isTyping = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    if (isTyping) msgDiv.classList.add('typing');

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;

    msgDiv.appendChild(contentDiv);
    chatMessages.appendChild(msgDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return contentDiv;
}

function typeMessage(element, text) {
    element.parentElement.classList.remove('typing');
    element.textContent = '';
    let i = 0;
    const speed = 20; // ms per char

    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            chatMessages.scrollTop = chatMessages.scrollHeight;
            setTimeout(type, speed);
        }
    }
    type();
}

// Fetch system status on load
async function fetchStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        if (response.ok) {
            const data = await response.json();
            document.getElementById('status-corpus').textContent = '医学 (Medical)';
            document.getElementById('status-model').textContent = 'DeepSeek';
            document.getElementById('status-online').textContent = '在线';
        }
    } catch (e) {
        document.getElementById('status-online').textContent = '离线';
        document.getElementById('status-online').style.color = '#ef4444';
        document.getElementById('status-online').classList.remove('pulse');
    }
}

// Fetch graph statistics
async function fetchGraphStats() {
    try {
        const response = await fetch(`${API_BASE}/graph-stats`);
        if (response.ok) {
            const data = await response.json();
            document.getElementById('graph-entities').textContent = data.entities.toLocaleString();
            document.getElementById('graph-relations').textContent = data.relations.toLocaleString();
            document.getElementById('graph-chunks').textContent = data.chunks.toLocaleString();

            // Update settings panel
            document.getElementById('setting-model').textContent = data.model;
            document.getElementById('setting-embed').textContent = data.embedding_model;
            document.getElementById('setting-url').textContent = data.llm_base_url;
            document.getElementById('setting-corpus').textContent = data.corpus;
        }
    } catch (e) {
        console.error('Failed to fetch graph stats:', e);
    }
}

// Panel switching
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();

        // Update active state
        document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');

        // Get panel name from data attribute
        const panelName = item.dataset.panel;

        // Hide all panels
        document.querySelectorAll('.panel').forEach(panel => {
            panel.style.display = 'none';
        });

        // Show selected panel
        const targetPanel = document.getElementById(`panel-${panelName}`);
        if (targetPanel) {
            targetPanel.style.display = 'flex';
        }

        // Load graph stats when switching to graph panel
        if (panelName === 'graph') {
            fetchGraphStats();
        }
    });
});

// Initial fetch
setTimeout(fetchStatus, 2000);
setTimeout(fetchGraphStats, 2500);

