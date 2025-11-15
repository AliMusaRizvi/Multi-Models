import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from peft import PeftModel
import time
from pathlib import Path
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Fine Tuned Transformers",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with modern design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }

    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Header Styling */
    .app-header {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        padding: 2rem 3rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .app-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    [data-testid="stSidebar"] .sidebar-content {
        padding: 1.5rem;
    }

    /* New Chat Button */
    .new-chat-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.875rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 1rem 0 2rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        border: none;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }

    .new-chat-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }

    /* History Section */
    .history-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 2rem 0 1rem 0;
        padding: 0 0.5rem;
    }

    .chat-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .chat-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateX(4px);
    }

    .chat-card.active {
        background: rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }

    .chat-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .chat-task-badge {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-sentiment { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
    .badge-code { background: rgba(59, 130, 246, 0.2); color: #93c5fd; }
    .badge-summary { background: rgba(34, 197, 94, 0.2); color: #86efac; }

    .chat-time {
        font-size: 0.7rem;
        color: #64748b;
    }

    .chat-preview {
        color: #cbd5e1;
        font-size: 0.85rem;
        line-height: 1.4;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }

    .chat-output-preview {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .empty-history {
        text-align: center;
        padding: 3rem 1rem;
        color: #64748b;
    }

    .empty-history-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }

    /* Main Content Area */
    .content-card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 2rem;
    }

    .content-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 2px solid rgba(102, 126, 234, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #64748b;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.75rem 1.5rem;
        border-radius: 10px 10px 0 0;
        transition: all 0.3s;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.05);
        color: #667eea;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }

    /* Input Areas */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 0.95rem;
        transition: all 0.3s;
        background: white;
    }

    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    .stButton > button[kind="secondary"] {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
    }

    /* Output Boxes */
    .output-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
    }

    .output-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
    }

    .sentiment-result {
        display: inline-flex;
        align-items: center;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .sentiment-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }

    .sentiment-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }

    .sentiment-neutral {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }

    /* Code Display */
    .stCodeBlock {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    /* Loading State */
    .loading-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }

    .loading-spinner {
        font-size: 3rem;
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        color: #64748b;
        font-size: 1.1rem;
        margin-top: 1rem;
    }

    /* Expandable Section */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 10px;
        font-weight: 600;
        color: #667eea;
    }

    /* Download Button */
    .stDownloadButton > button {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stDownloadButton > button:hover {
        background: #667eea;
        color: white;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Responsive Design */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .app-header {
            padding: 1.5rem;
        }

        .app-title {
            font-size: 1.75rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'tokenizers' not in st.session_state:
    st.session_state.tokenizers = {}
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'base_path' not in st.session_state:
    st.session_state.base_path = "/Users/fahadbaloch/Documents"
if 'auto_load_attempted' not in st.session_state:
    st.session_state.auto_load_attempted = False


def detect_task_type(model_path):
    """Detect which task type based on model files and config"""
    model_path = Path(model_path)
    has_adapter = (model_path / "adapter_config.json").exists()

    if has_adapter:
        adapter_config_file = model_path / "adapter_config.json"
        try:
            with open(adapter_config_file, 'r') as f:
                adapter_config = json.load(f)
                base_model = adapter_config.get('base_model_name_or_path', '').lower()
                if 'gpt' in base_model:
                    return "code_generation", "lora"
                elif 'bert' in base_model:
                    return "sentiment_classification", "lora"
                elif 't5' in base_model:
                    return "summarization", "lora"
        except:
            pass

    has_merges = (model_path / "merges.txt").exists()
    has_vocab_json = (model_path / "vocab.json").exists()
    if has_merges and has_vocab_json:
        return "code_generation", "lora" if has_adapter else "full"

    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                model_type = config.get('model_type', '').lower()
                if 'gpt2' in model_type or 'gpt' in model_type:
                    return "code_generation", "lora" if has_adapter else "full"
                elif 'bert' in model_type:
                    return "sentiment_classification", "lora" if has_adapter else "full"
                elif 't5' in model_type:
                    return "summarization", "lora" if has_adapter else "full"
        except:
            pass

    dir_name = model_path.name.lower()
    if 'task1' in dir_name:
        return "sentiment_classification", "lora" if has_adapter else "full"
    elif 'task2' in dir_name:
        return "code_generation", "lora" if has_adapter else "full"
    elif 'task3' in dir_name:
        return "summarization", "lora" if has_adapter else "full"

    return "unknown", "unknown"


def load_sentiment_model(model_path, is_lora=False):
    try:
        model_path = Path(model_path)
        if is_lora:
            base_model_name = "bert-base-uncased"
            num_labels = 3
            config_file = model_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        num_labels = config.get('num_labels', 3)
                except:
                    pass
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
            model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)


def load_code_generation_model(model_path, is_lora=False):
    try:
        model_path = Path(model_path)
        if is_lora:
            base_model_name = "gpt2-medium"
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            except:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                special_tokens = {
                    'additional_special_tokens': ['<|pseudo|>', '<|code|>'],
                    'pad_token': '<|pad|>'
                }
                tokenizer.add_special_tokens(special_tokens)
            base_model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base_model, str(model_path))
            tokenizer.padding_side = 'left'
        else:
            model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            except:
                tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
            tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)


def load_summarization_model(model_path, is_lora=False):
    try:
        model_path = Path(model_path)
        if is_lora:
            base_model = T5ForConditionalGeneration.from_pretrained("t5-base")
            model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            model = T5ForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True)
        try:
            tokenizer = T5Tokenizer.from_pretrained(str(model_path))
        except:
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer.pad_token_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)


def load_all_models(base_path):
    base_path = Path(base_path)
    models = {}
    tokenizers = {}
    for i in range(1, 4):
        task_name = f"Task{i}"
        task_path = base_path / task_name
        if task_path.exists():
            task_type, model_type = detect_task_type(task_path)
            if task_type != "unknown":
                is_lora = (model_type == "lora")
                try:
                    if task_type == "sentiment_classification":
                        model, tokenizer, error = load_sentiment_model(task_path, is_lora)
                    elif task_type == "code_generation":
                        model, tokenizer, error = load_code_generation_model(task_path, is_lora)
                    elif task_type == "summarization":
                        model, tokenizer, error = load_summarization_model(task_path, is_lora)
                    if not error and model is not None:
                        models[task_type] = model
                        tokenizers[task_type] = tokenizer
                except:
                    pass
    return models, tokenizers


def classify_sentiment(text, model, tokenizer):
    try:
        device = next(model.parameters()).device
        inputs = tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(predicted_class, "Unknown")
        return sentiment, None
    except Exception as e:
        return None, str(e)


def generate_code(pseudocode, model, tokenizer):
    try:
        device = next(model.parameters()).device
        prompt = f"<|pseudo|> {pseudocode.strip()} <|code|>"
        inputs = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get('attention_mask'),
                max_length=512,
                num_beams=5,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        try:
            code_marker = "<|code|>"
            end_marker = "<|endoftext|>"
            if code_marker in generated_text:
                code_start = generated_text.find(code_marker) + len(code_marker)
                code_text = generated_text[code_start:]
                if end_marker in code_text:
                    code_text = code_text[:code_text.find(end_marker)]
                code = code_text.strip()
            else:
                code = generated_text.replace(prompt, "").strip()
        except:
            code = generated_text.replace(prompt, "").strip()
        code = code.replace("<|pseudo|>", "").replace("<|code|>", "").replace("<|endoftext|>", "").strip()
        return code, None
    except Exception as e:
        return None, str(e)


def generate_summary(text, model, tokenizer):
    try:
        device = next(model.parameters()).device
        input_text = "summarize: " + text
        inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary, None
    except Exception as e:
        return None, str(e)


def add_to_history(task_type, input_text, output_text):
    chat_id = f"chat_{len(st.session_state.chat_history)}_{int(time.time())}"
    st.session_state.chat_history.insert(0, {
        'id': chat_id,
        'timestamp': datetime.now(),
        'task': task_type,
        'input': input_text,
        'output': output_text
    })
    st.session_state.current_chat_id = chat_id
    if len(st.session_state.chat_history) > 50:
        st.session_state.chat_history = st.session_state.chat_history[:50]


def new_chat():
    st.session_state.current_chat_id = None


# Auto-load models
if not st.session_state.auto_load_attempted:
    st.session_state.auto_load_attempted = True
    try:
        models, tokenizers = load_all_models(st.session_state.base_path)
        if models:
            st.session_state.models = models
            st.session_state.tokenizers = tokenizers
            st.session_state.models_loaded = True
    except:
        pass

# Sidebar
with st.sidebar:
    if st.button("✨ New Chat", use_container_width=True, key="new_chat_btn"):
        new_chat()
        st.rerun()

    st.markdown('<div class="history-title">Recent Conversations</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        for item in st.session_state.chat_history[:15]:
            time_str = item['timestamp'].strftime("%H:%M")
            date_str = item['timestamp'].strftime("%b %d")

            task_badges = {
                "sentiment_classification": ("Sentiment", "badge-sentiment"),
                "code_generation": ("Code Gen", "badge-code"),
                "summarization": ("Summary", "badge-summary")
            }
            task_label, badge_class = task_badges.get(item['task'], (item['task'], "badge-sentiment"))

            input_preview = item['input'][:80] + "..." if len(item['input']) > 80 else item['input']
            output_preview = item['output'][:50] + "..." if len(item['output']) > 50 else item['output']

            active_class = "active" if item['id'] == st.session_state.current_chat_id else ""

            st.markdown(f"""
            <div class="chat-card {active_class}">
                <div class="chat-meta">
                    <span class="chat-task-badge {badge_class}">{task_label}</span>
                    <span class="chat-time">{time_str}</span>
                </div>
                <div class="chat-preview">{input_preview}</div>
                <div class="chat-output-preview">→ {output_preview}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear History", use_container_width=True, key="clear_history"):
            st.session_state.chat_history = []
            st.session_state.current_chat_id = None
            st.rerun()
    else:
        st.markdown("""
        <div class="empty-history">
            <div class="empty-history-icon">💬</div>
            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">No conversations yet</div>
            <div style="font-size: 0.75rem;">Your chat history will appear here</div>
        </div>
        """, unsafe_allow_html=True)

# Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">Fine Tuned Transformers</h1>
    <p class="app-subtitle">Advanced NLP Models for Sentiment Analysis, Code Generation & Text Summarization</p>
</div>
""", unsafe_allow_html=True)

# Main Content
if not st.session_state.models_loaded or not st.session_state.models:
    st.markdown("""
    <div class="loading-card">
        <div class="loading-spinner">⚡</div>
        <div class="loading-text">Initializing AI Models...</div>
        <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;">This may take a moment</div>
    </div>
    """, unsafe_allow_html=True)
else:
    available_models = list(st.session_state.models.keys())
    task_display_names = {
        "sentiment_classification": "📊 Sentiment Analysis",
        "code_generation": "💻 Code Generation",
        "summarization": "📝 Text Summarization"
    }

    tab_names = [task_display_names.get(task, task) for task in available_models]
    tabs = st.tabs(tab_names)

    for tab, task_type in zip(tabs, available_models):
        with tab:
            model = st.session_state.models[task_type]
            tokenizer = st.session_state.tokenizers[task_type]

            # Sentiment Analysis Tab
            if task_type == "sentiment_classification":
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.markdown('<h2 class="content-title">Analyze Customer Sentiment</h2>', unsafe_allow_html=True)

                input_text = st.text_area(
                    "Enter customer feedback or review",
                    height=150,
                    placeholder="Example: The product quality is excellent and the customer service was outstanding...",
                    key=f"{task_type}_input",
                    label_visibility="collapsed"
                )

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True,
                                            key=f"{task_type}_btn")
                with col2:
                    clear_btn = st.button("Clear", use_container_width=True, key=f"{task_type}_clear")

                if clear_btn:
                    st.rerun()

                if analyze_btn and input_text.strip():
                    with st.spinner("Analyzing sentiment..."):
                        sentiment, error = classify_sentiment(input_text, model, tokenizer)

                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        add_to_history(task_type, input_text, sentiment)
                        sentiment_class = f"sentiment-{sentiment.lower()}"

                        st.markdown(f"""
                        <div class="output-container">
                            <div class="output-title">Sentiment Result</div>
                            <div class="sentiment-result {sentiment_class}">
                                {sentiment}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # Code Generation Tab
            elif task_type == "code_generation":
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.markdown('<h2 class="content-title">Generate Python Code from Pseudocode</h2>',
                            unsafe_allow_html=True)

                with st.expander("💡 View Example Pseudocode"):
                    st.markdown("""
                    **Example 1: Simple Loop**
                    ```
                    create a variable counter and set it to 0
                    while counter is less than 10
                        print the counter value
                        increment counter by 1
                    ```

                    **Example 2: Function Definition**
                    ```
                    define a function calculate_average that takes a list of numbers
                    sum all the numbers in the list
                    divide by the length of the list
                    return the result
                    ```

                    **Example 3: Data Processing**
                    ```
                    create an empty list called even_numbers
                    for each number from 1 to 20
                        if the number is divisible by 2
                            add it to even_numbers
                    print the even_numbers list
                    ```
                    """)

                input_text = st.text_area(
                    "Enter your pseudocode instructions",
                    height=200,
                    placeholder="Example:\ncreate a list of numbers from 1 to 10\nfor each number in the list\n    if number is even\n        print the number",
                    key=f"{task_type}_input",
                    label_visibility="collapsed"
                )

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    generate_btn = st.button("⚡ Generate Code", type="primary", use_container_width=True,
                                             key=f"{task_type}_btn")
                with col2:
                    clear_btn = st.button("Clear", use_container_width=True, key=f"{task_type}_clear")

                if clear_btn:
                    st.rerun()

                if generate_btn and input_text.strip():
                    with st.spinner("Generating Python code..."):
                        code, error = generate_code(input_text, model, tokenizer)

                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        add_to_history(task_type, input_text, code)

                        st.markdown('<div class="output-container">', unsafe_allow_html=True)
                        st.markdown('<div class="output-title">Generated Python Code</div>', unsafe_allow_html=True)
                        st.code(code, language="python", line_numbers=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.download_button(
                            label="💾 Download Python File",
                            data=code,
                            file_name="generated_code.py",
                            mime="text/x-python",
                            use_container_width=True,
                            key=f"{task_type}_download"
                        )

                st.markdown('</div>', unsafe_allow_html=True)

            # Summarization Tab
            elif task_type == "summarization":
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.markdown('<h2 class="content-title">Summarize Long-Form Content</h2>', unsafe_allow_html=True)

                input_text = st.text_area(
                    "Enter text to summarize",
                    height=250,
                    placeholder="Paste your article, document, or long-form text here...",
                    key=f"{task_type}_input",
                    label_visibility="collapsed"
                )

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    summarize_btn = st.button("📝 Generate Summary", type="primary", use_container_width=True,
                                              key=f"{task_type}_btn")
                with col2:
                    clear_btn = st.button("Clear", use_container_width=True, key=f"{task_type}_clear")

                if clear_btn:
                    st.rerun()

                if summarize_btn and input_text.strip():
                    with st.spinner("Generating summary..."):
                        summary, error = generate_summary(input_text, model, tokenizer)

                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        add_to_history(task_type, input_text, summary)

                        st.markdown(f"""
                        <div class="output-container">
                            <div class="output-title">Summary</div>
                            <div style="font-size: 1rem; line-height: 1.7; color: #334155;">
                                {summary}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.download_button(
                            label="💾 Download Summary",
                            data=summary,
                            file_name="summary.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key=f"{task_type}_download"
                        )

                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 3rem 0 2rem 0; color: rgba(255, 255, 255, 0.7); font-size: 0.85rem;">
    <div style="margin-bottom: 0.5rem;">Built with Advanced Transformer Models</div>
    <div style="font-size: 0.75rem;">BERT • GPT-2-Medium • T5 with LoRA Fine-tuning</div>
</div>
""", unsafe_allow_html=True)