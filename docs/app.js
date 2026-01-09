/**
 * Deleuzian Thinking Machine - Client-side JavaScript
 * Handles SSE streaming, UI updates, and visualizations
 */

// =============================================================================
// DOM Elements
// =============================================================================

const questionForm = document.getElementById('question-form');
const questionInput = document.getElementById('question-input');
const submitBtn = document.getElementById('submit-btn');
const responseSection = document.getElementById('response-section');
const responseText = document.getElementById('response-text');
const cursor = document.getElementById('cursor');
const sourcesPanel = document.getElementById('sources-panel');
const sourcesList = document.getElementById('sources-list');
const loading = document.getElementById('loading');
const copyBtn = document.getElementById('copy-btn');

// Process panel elements
const processPanel = document.getElementById('process-panel');
const processToggle = document.getElementById('process-toggle');
const processContent = document.getElementById('process-content');
const processSummary = document.getElementById('process-summary');
const routingBadge = document.getElementById('routing-badge');
const thinkingText = document.getElementById('thinking-text');
const thinkingTokens = document.getElementById('thinking-tokens');
const flowContainer = document.getElementById('flow-container');

// =============================================================================
// State
// =============================================================================

let isStreaming = false;
let rawResponseContent = '';
let currentSessionId = null;
let citationStore = {};
let toolCount = 0;
let thinkingCharCount = 0;

// Tool icons and labels
const TOOL_CONFIG = {
    'search_corpus': { icon: '📖', label: 'Corpus' },
    'search_concepts': { icon: '🧠', label: 'Graph' },
    'traverse_relationships': { icon: '🕸️', label: 'Rhizome' },
    'search_method_patterns': { icon: '⚡', label: 'Patterns' },
    'get_supporting_quotes': { icon: '💬', label: 'Quotes' }
};

// =============================================================================
// Event Listeners
// =============================================================================

const budgetRange = document.getElementById('budget-range');
const budgetValue = document.getElementById('budget-value');

questionForm.addEventListener('submit', handleSubmit);
if (processToggle) {
    processToggle.addEventListener('click', toggleProcess);
}
copyBtn.addEventListener('click', copyResponse);

budgetRange.addEventListener('input', (e) => {
    budgetValue.textContent = e.target.value;
});

// Allow Cmd/Ctrl + Enter to submit
questionInput.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        questionForm.dispatchEvent(new Event('submit'));
    }
});

// =============================================================================
// Main Submit Handler
// =============================================================================

async function handleSubmit(e) {
    e.preventDefault();

    const question = questionInput.value.trim();
    if (!question || isStreaming) return;

    resetUI();
    showLoading();

    isStreaming = true;
    submitBtn.disabled = true;

    try {
        const model = document.getElementById('model-select').value;
        const budget = document.getElementById('budget-range').value;

        const API_BASE = 'https://wisdomfunction-deleuze-thinking-machine.hf.space';
        const response = await fetch(`${API_BASE}/api/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                model: model,
                thinking_budget: parseInt(budget),
                session_id: currentSessionId
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        hideLoading();
        showResponse();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleEvent(data);
                    } catch (parseError) {
                        console.error('Parse error:', parseError);
                    }
                }
            }
        }

    } catch (error) {
        console.error('Error:', error);
        hideLoading();
        showError(error.message);
    } finally {
        isStreaming = false;
        submitBtn.disabled = false;
        cursor.classList.add('hidden');
        updateProcessSummary();
    }
}

// =============================================================================
// Event Handlers
// =============================================================================

function handleEvent(data) {
    switch (data.type) {
        case 'session':
            currentSessionId = data.session_id;
            break;

        case 'routing':
            showRouting(data.question_type);
            break;

        case 'thinking_start':
            processPanel.classList.remove('hidden');
            break;

        case 'thinking':
            appendThinking(data.content);
            break;

        case 'thinking_progress':
            updateThinkingTokens(data.tokens, data.budget);
            break;

        case 'content_start':
            cursor.classList.remove('hidden');
            break;

        case 'content':
            appendContent(data.content);
            break;

        case 'tool_call_start':
            addToolNode(data.tool, null, true);
            break;

        case 'tool_call':
            addToolNode(data.tool, data.query, false);
            break;

        case 'sources':
            showSources(data.sources);
            break;

        case 'citations':
            storeCitations(data.citations);
            linkInlineCitations();
            break;

        case 'error':
            showError(data.message);
            break;

        case 'done':
            cursor.classList.add('hidden');
            break;
    }
}

// =============================================================================
// UI Updates
// =============================================================================

function resetUI() {
    responseText.innerHTML = '';
    if (thinkingText) thinkingText.textContent = '';
    if (flowContainer) flowContainer.innerHTML = '';
    sourcesList.innerHTML = '';
    rawResponseContent = '';
    citationStore = {};
    toolCount = 0;
    thinkingCharCount = 0;

    // Close any open popover
    const existingPopover = document.querySelector('.citation-popover');
    if (existingPopover) existingPopover.remove();

    responseSection.classList.add('hidden');
    if (processPanel) processPanel.classList.add('hidden');
    sourcesPanel.classList.add('hidden');
    cursor.classList.add('hidden');
    copyBtn.classList.add('hidden');

    // Reset process panel
    if (routingBadge) routingBadge.textContent = '';
    if (processSummary) processSummary.textContent = '';
    if (thinkingTokens) thinkingTokens.textContent = '';
}

function showLoading() {
    loading.classList.remove('hidden');
}

function hideLoading() {
    loading.classList.add('hidden');
}

function showResponse() {
    responseSection.classList.remove('hidden');
}

// =============================================================================
// Process Panel
// =============================================================================

function showRouting(questionType) {
    if (routingBadge) {
        routingBadge.textContent = questionType.replace(/_/g, ' ');
    }
}

function toggleProcess() {
    if (!processContent) return;
    processContent.classList.toggle('collapsed');

    const chevron = processToggle.querySelector('.toggle-chevron');
    const isCollapsed = processContent.classList.contains('collapsed');
    if (chevron) {
        chevron.style.transform = isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
    }
}

function updateProcessSummary() {
    if (!processSummary) return;

    const parts = [];
    if (thinkingCharCount > 0) {
        const tokens = Math.round(thinkingCharCount / 4);
        parts.push(`${tokens.toLocaleString()} thinking tokens`);
    }
    if (toolCount > 0) {
        parts.push(`${toolCount} tool${toolCount > 1 ? 's' : ''} used`);
    }
    processSummary.textContent = parts.join(' · ');
}

function updateThinkingTokens(tokens, budget) {
    if (thinkingTokens) {
        thinkingTokens.textContent = `${tokens.toLocaleString()} / ${budget.toLocaleString()}`;
    }
}

// =============================================================================
// Tool Flow
// =============================================================================

function addToolNode(toolName, query, isLoading = false) {
    if (!flowContainer) return;

    const config = TOOL_CONFIG[toolName] || { icon: '⚡', label: toolName };

    // Check if updating existing loading node
    const existingLoadingNode = flowContainer.querySelector('.tool-chip.loading');
    if (existingLoadingNode && !isLoading) {
        existingLoadingNode.classList.remove('loading');
        if (query) {
            const queryEl = existingLoadingNode.querySelector('.chip-query');
            if (queryEl) queryEl.textContent = parseQuery(query);
        }
        return;
    }

    toolCount++;

    // Create compact chip instead of full node
    const chip = document.createElement('span');
    chip.className = `tool-chip${isLoading ? ' loading' : ''}`;
    chip.setAttribute('data-tool', toolName);
    chip.innerHTML = `<span class="chip-icon">${config.icon}</span><span class="chip-label">${config.label}</span>${query ? `<span class="chip-query">${escapeHtml(parseQuery(query))}</span>` : ''}`;

    flowContainer.appendChild(chip);
    updateProcessSummary();
}

function parseQuery(queryStr) {
    try {
        if (queryStr.startsWith('{')) {
            const parsed = JSON.parse(queryStr);
            return parsed.query || parsed.entity || parsed.source || queryStr;
        }
    } catch (e) { }
    return queryStr;
}

// =============================================================================
// Content & Thinking
// =============================================================================

function appendContent(text) {
    rawResponseContent += text;

    if (typeof marked !== 'undefined') {
        responseText.innerHTML = marked.parse(rawResponseContent);
    } else {
        responseText.textContent = rawResponseContent;
    }

    copyBtn.classList.remove('hidden');
    responseText.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function appendThinking(text) {
    if (!thinkingText) return;
    thinkingText.textContent += text;
    thinkingCharCount += text.length;
    updateProcessSummary();
}

// =============================================================================
// Sources
// =============================================================================

function showSources(sources) {
    if (!sources || sources.length === 0) return;

    sourcesPanel.classList.remove('hidden');
    sourcesList.innerHTML = '';

    const uniqueSources = [...new Set(sources)];
    for (const source of uniqueSources.slice(0, 20)) {
        const li = document.createElement('li');
        li.textContent = source;
        sourcesList.appendChild(li);
    }
}

function showError(message) {
    responseSection.classList.remove('hidden');
    responseText.innerHTML = `<span style="color: #e74c3c;">Error: ${escapeHtml(message)}</span>`;
}

// =============================================================================
// Citations
// =============================================================================

function storeCitations(citations) {
    if (!citations) return;
    citations.forEach(cite => {
        citationStore[cite.id] = cite;
    });
}

function linkInlineCitations() {
    const html = responseText.innerHTML;
    const linkedHtml = html.replace(
        /\[([^\]]+) #(\d+)\]/g,
        (match, title, id) => {
            if (citationStore[id]) {
                return `<span class="citation-link" data-citation-id="${id}" tabindex="0">[${escapeHtml(title)}]</span>`;
            }
            return match;
        }
    );

    responseText.innerHTML = linkedHtml;

    responseText.querySelectorAll('.citation-link').forEach(link => {
        link.addEventListener('click', (e) => showCitationPopover(e.target));
        link.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                showCitationPopover(e.target);
            }
        });
    });
}

function showCitationPopover(element) {
    const existing = document.querySelector('.citation-popover');
    if (existing) existing.remove();

    const citationId = element.dataset.citationId;
    const citation = citationStore[citationId];
    if (!citation) return;

    const popover = document.createElement('div');
    popover.className = 'citation-popover';

    const entitiesHtml = citation.entities && citation.entities.length > 0
        ? `<div class="popover-entities">Entities: ${citation.entities.map(e => escapeHtml(e)).join(', ')}</div>`
        : '';

    popover.innerHTML = `
        <div class="popover-header">
            <span class="popover-title">${escapeHtml(citation.book_title)}</span>
            <button class="popover-close">&times;</button>
        </div>
        <div class="popover-text">${escapeHtml(citation.text)}</div>
        ${entitiesHtml}
    `;

    document.body.appendChild(popover);

    const rect = element.getBoundingClientRect();
    const popoverRect = popover.getBoundingClientRect();

    let left = rect.left + (rect.width / 2) - (popoverRect.width / 2);
    let top = rect.bottom + 8;

    if (left < 10) left = 10;
    if (left + popoverRect.width > window.innerWidth - 10) {
        left = window.innerWidth - popoverRect.width - 10;
    }
    if (top + popoverRect.height > window.innerHeight - 10) {
        top = rect.top - popoverRect.height - 8;
    }

    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;

    const closeBtn = popover.querySelector('.popover-close');
    closeBtn.addEventListener('click', () => popover.remove());

    const closeOnOutside = (e) => {
        if (!popover.contains(e.target) && e.target !== element) {
            popover.remove();
            document.removeEventListener('click', closeOnOutside);
        }
    };
    setTimeout(() => document.addEventListener('click', closeOnOutside), 10);

    const closeOnEscape = (e) => {
        if (e.key === 'Escape') {
            popover.remove();
            document.removeEventListener('keydown', closeOnEscape);
        }
    };
    document.addEventListener('keydown', closeOnEscape);
}

// =============================================================================
// Utilities
// =============================================================================

async function copyResponse() {
    try {
        await navigator.clipboard.writeText(rawResponseContent);
        const originalText = copyBtn.querySelector('.copy-text').textContent;
        copyBtn.querySelector('.copy-text').textContent = 'Copied!';
        copyBtn.classList.add('copied');

        setTimeout(() => {
            copyBtn.querySelector('.copy-text').textContent = originalText;
            copyBtn.classList.remove('copied');
        }, 1500);
    } catch (err) {
        console.error('Failed to copy:', err);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
