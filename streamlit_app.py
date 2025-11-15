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

# Page configuration
st.set_page_config(
    page_title="Fine-tuned Transformer",
    page_icon="🤖",
    layout="wide"
)

# Professional CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 10px 24px;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1557b0;
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.3);
    }
    .output-box {
        background-color: black;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 16px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .code-output {
        background-color: #f5f5f5;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #1a73e8;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 16px;
        margin: 10px 0;
    }
    .sentiment-positive {
        background-color: #e6f4ea;
        color: #1e8e3e;
    }
    .sentiment-negative {
        background-color: #fce8e6;
        color: #d93025;
    }
    .sentiment-neutral {
        background-color: #e8f0fe;
        color: #1a73e8;
    }
    .metric-card {
        background-color: black;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-label {
        color: #5f6368;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #202124;
        font-size: 28px;
        font-weight: 600;
    }
    h1 {
        color: #202124;
        font-weight: 400;
        font-size: 32px;
        margin-bottom: 8px;
    }
    .subtitle {
        color: #5f6368;
        font-size: 14px;
        margin-bottom: 32px;
    }
    .status-success {
        background-color: #e6f4ea;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #1e8e3e;
        margin: 12px 0;
        color: #137333;
    }
    .status-error {
        background-color: #fce8e6;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #d93025;
        margin: 12px 0;
        color: #c5221f;
    }
    .status-info {
        background-color: #e8f0fe;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #1a73e8;
        margin: 12px 0;
        color: #1967d2;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 4px;
        height: 8px;
        margin: 4px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background-color: #1a73e8;
        transition: width 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_task' not in st.session_state:
    st.session_state.current_task = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'base_path' not in st.session_state:
    st.session_state.base_path = None


def detect_task_type(model_path):
    """Detect which task type based on model files and config"""
    model_path = Path(model_path)

    # Check for LoRA adapter
    has_adapter = (model_path / "adapter_config.json").exists()

    # Priority 1: Check adapter_config.json for LoRA models (most reliable)
    if has_adapter:
        adapter_config_file = model_path / "adapter_config.json"
        try:
            import json
            with open(adapter_config_file, 'r') as f:
                adapter_config = json.load(f)

                # Check base_model_name_or_path first (most reliable)
                base_model = adapter_config.get('base_model_name_or_path', '').lower()
                if 'gpt' in base_model:
                    return "code_generation", "lora"
                elif 'bert' in base_model:
                    return "sentiment_classification", "lora"
                elif 't5' in base_model:
                    return "summarization", "lora"

                # Check target_modules as fallback
                target_modules = adapter_config.get('target_modules', [])
                if isinstance(target_modules, str):
                    target_modules = [target_modules]

                # GPT-2 modules: c_attn, c_proj, c_fc, attn.c_attn, mlp.c_fc, etc.
                gpt_modules = ['c_attn', 'c_proj', 'c_fc', 'attn', 'mlp']
                if any(any(gpt_mod in str(m) for gpt_mod in gpt_modules) for m in target_modules):
                    return "code_generation", "lora"

                # BERT modules: query, key, value, dense, attention
                bert_modules = ['query', 'key', 'value', 'dense', 'attention.self']
                if any(any(bert_mod in str(m) for bert_mod in bert_modules) for m in target_modules):
                    return "sentiment_classification", "lora"

                # T5 modules: q, k, v, o, wi, wo, SelfAttention, EncDecAttention
                t5_modules = ['SelfAttention', 'EncDecAttention', 'DenseReluDense']
                if any(any(t5_mod in str(m) for t5_mod in t5_modules) for m in target_modules):
                    return "summarization", "lora"

                # If target_modules contains simple names like 'q', 'k', 'v'
                simple_t5 = ['q', 'k', 'v', 'o', 'wi', 'wo']
                if any(m in simple_t5 for m in target_modules):
                    return "summarization", "lora"

        except Exception as e:
            print(f"Warning: Could not read adapter_config.json: {e}")

    # Priority 2: Check config.json
    config_file = model_path / "config.json"
    if config_file.exists():
        import json
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                model_type = config.get('model_type', '').lower()
                architectures = config.get('architectures', [])

                # Check architectures first (most specific)
                for arch in architectures:
                    arch_lower = arch.lower()
                    if 'gpt' in arch_lower or 'gpt2' in arch_lower:
                        return "code_generation", "lora" if has_adapter else "full"
                    elif 'bert' in arch_lower and 'classification' in arch_lower:
                        return "sentiment_classification", "lora" if has_adapter else "full"
                    elif 't5' in arch_lower and 'conditional' in arch_lower:
                        return "summarization", "lora" if has_adapter else "full"

                # Check model_type
                if 'gpt2' in model_type or 'gpt' in model_type:
                    return "code_generation", "lora" if has_adapter else "full"
                elif 'bert' in model_type:
                    return "sentiment_classification", "lora" if has_adapter else "full"
                elif 't5' in model_type:
                    return "summarization", "lora" if has_adapter else "full"

        except Exception as e:
            print(f"Warning: Could not read config.json: {e}")

    # Priority 3: Check for GPT-2 specific files
    has_merges = (model_path / "merges.txt").exists()
    has_vocab_json = (model_path / "vocab.json").exists()
    if has_merges and has_vocab_json:
        # GPT-2 uses merges.txt and vocab.json
        return "code_generation", "lora" if has_adapter else "full"

    # Priority 4: Check tokenizer files to determine model type
    tokenizer_config = model_path / "tokenizer_config.json"
    if tokenizer_config.exists():
        import json
        try:
            with open(tokenizer_config, 'r') as f:
                tok_config = json.load(f)
                tokenizer_class = tok_config.get('tokenizer_class', '').lower()

                if 'gpt2' in tokenizer_class:
                    return "code_generation", "lora" if has_adapter else "full"
                elif 'bert' in tokenizer_class:
                    return "sentiment_classification", "lora" if has_adapter else "full"
                elif 't5' in tokenizer_class:
                    return "summarization", "lora" if has_adapter else "full"
        except:
            pass

    # Priority 5: Directory name fallback
    dir_name = model_path.name.lower()
    if 'task1' in dir_name:
        return "sentiment_classification", "lora" if has_adapter else "full"
    elif 'task2' in dir_name:
        return "code_generation", "lora" if has_adapter else "full"
    elif 'task3' in dir_name:
        return "summarization", "lora" if has_adapter else "full"

    return "unknown", "unknown"


@st.cache_resource
def load_sentiment_model(model_path, is_lora=False):
    """Load BERT model for sentiment classification (Task 1)"""
    try:
        model_path = Path(model_path)

        if is_lora:
            # Read adapter config to determine base model and num_labels
            adapter_config_file = model_path / "adapter_config.json"
            base_model_name = "bert-base-uncased"
            num_labels = 3  # default: positive, negative, neutral

            # Try to get num_labels from config.json
            config_file = model_path / "config.json"
            if config_file.exists():
                import json
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        num_labels = config.get('num_labels', 3)
                except:
                    pass

            if adapter_config_file.exists():
                import json
                try:
                    with open(adapter_config_file, 'r') as f:
                        adapter_config = json.load(f)
                        base_model_name = adapter_config.get('base_model_name_or_path', 'bert-base-uncased')
                except:
                    pass

            # Load base BERT model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels
            )
            model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            # Load full fine-tuned model
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                local_files_only=True
            )

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except:
            # Fallback to base model tokenizer
            if is_lora:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                except:
                    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            else:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        return model, tokenizer, None
    except Exception as e:
        return None, None, f"Error loading sentiment model: {str(e)}"


@st.cache_resource
def load_code_generation_model(model_path, is_lora=False):
    """Load GPT-2 model for code generation (Task 2)"""
    try:
        model_path = Path(model_path)

        if is_lora:
            # Read adapter config to determine base model
            adapter_config_file = model_path / "adapter_config.json"
            base_model_name = "gpt2-medium"  # Default for Task 2

            if adapter_config_file.exists():
                import json
                try:
                    with open(adapter_config_file, 'r') as f:
                        adapter_config = json.load(f)
                        base_model_name = adapter_config.get('base_model_name_or_path', 'gpt2-medium')
                except:
                    pass

            # Load base GPT-2 model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            # Load tokenizer first to get vocab size
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            except:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                # Add special tokens for SPOC dataset
                special_tokens = {
                    'additional_special_tokens': ['<|pseudo|>', '<|code|>'],
                    'pad_token': '<|pad|>'
                }
                tokenizer.add_special_tokens(special_tokens)

            # Resize embeddings to match tokenizer vocab size
            base_model.resize_token_embeddings(len(tokenizer))

            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, str(model_path))

            # Set padding side to left for causal LM
            tokenizer.padding_side = 'left'

        else:
            # Load full fine-tuned model
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                local_files_only=True
            )

            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            except:
                tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
                special_tokens = {
                    'additional_special_tokens': ['<|pseudo|>', '<|code|>'],
                    'pad_token': '<|pad|>'
                }
                tokenizer.add_special_tokens(special_tokens)

            tokenizer.padding_side = 'left'

        # Set pad token for GPT-2
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        return model, tokenizer, None
    except Exception as e:
        return None, None, f"Error loading code generation model: {str(e)}"


@st.cache_resource
def load_summarization_model(model_path, is_lora=False):
    """Load T5 model for summarization (Task 3)"""
    try:
        model_path = Path(model_path)

        if is_lora:
            # Load base T5 model
            base_model = T5ForConditionalGeneration.from_pretrained("t5-base")
            model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            # Load full fine-tuned model
            model = T5ForConditionalGeneration.from_pretrained(
                str(model_path),
                local_files_only=True
            )

        # Load tokenizer
        try:
            tokenizer = T5Tokenizer.from_pretrained(str(model_path))
        except:
            tokenizer = T5Tokenizer.from_pretrained("t5-base")

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Ensure decoder_start_token_id is set
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer.pad_token_id

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        return model, tokenizer, None
    except Exception as e:
        return None, None, f"Error loading summarization model: {str(e)}"


def load_model_auto(model_path):
    """Automatically detect task type and load appropriate model"""
    task_type, model_type = detect_task_type(model_path)
    is_lora = (model_type == "lora")

    if task_type == "sentiment_classification":
        result = load_sentiment_model(model_path, is_lora)
        return result, task_type
    elif task_type == "code_generation":
        result = load_code_generation_model(model_path, is_lora)
        return result, task_type
    elif task_type == "summarization":
        result = load_summarization_model(model_path, is_lora)
        return result, task_type
    else:
        return (None, None, "Could not detect task type"), task_type


def classify_sentiment(text, model, tokenizer):
    """Classify sentiment of text (Task 1)"""
    try:
        device = next(model.parameters()).device

        inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

        # Map to sentiment labels
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(predicted_class, "Unknown")
        confidence = probabilities[0][predicted_class].item()

        return sentiment, confidence, probabilities[0].cpu().numpy(), None
    except Exception as e:
        return None, None, None, str(e)


def generate_code(pseudocode, model, tokenizer):
    """Generate code from pseudocode (Task 2) using SPOC format"""
    try:
        device = next(model.parameters()).device

        # Format input with special tokens (SPOC format)
        # Format: <|pseudo|> pseudocode <|code|>
        prompt = f"<|pseudo|> {pseudocode.strip()} <|code|>"

        inputs = tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt",
            padding=True
        ).to(device)

        # Get attention mask
        attention_mask = inputs.get('attention_mask', None)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask,
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

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract code between <|code|> and <|endoftext|>
        try:
            # Find the <|code|> marker
            code_marker = "<|code|>"
            end_marker = "<|endoftext|>"

            if code_marker in generated_text:
                code_start = generated_text.find(code_marker) + len(code_marker)
                code_text = generated_text[code_start:]

                # Remove end marker if present
                if end_marker in code_text:
                    code_text = code_text[:code_text.find(end_marker)]

                code = code_text.strip()
            else:
                # Fallback: remove the prompt
                code = generated_text.replace(prompt, "").strip()

        except Exception as e:
            # Fallback: just remove the prompt
            code = generated_text.replace(prompt, "").strip()

        # Clean up any remaining special tokens
        code = code.replace("<|pseudo|>", "").replace("<|code|>", "").replace("<|endoftext|>", "").strip()

        return code, None
    except Exception as e:
        return None, str(e)


def generate_summary(text, model, tokenizer):
    """Generate summary for text (Task 3)"""
    try:
        device = next(model.parameters()).device

        input_text = "summarize: " + text
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
            padding=True
        ).to(device)

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


def find_task_directories(base_path):
    """Find available task directories"""
    base_path = Path(base_path)
    tasks = {}

    for i in range(1, 4):
        task_name = f"Task{i}"
        task_path = base_path / task_name
        if task_path.exists() and task_path.is_dir():
            task_type, model_type = detect_task_type(task_path)
            if task_type != "unknown":
                tasks[task_name] = {
                    'path': str(task_path),
                    'type': task_type,
                    'model_type': model_type
                }

    return tasks


# Main UI Layout
st.title("🤖 Fine-tuned Transformer")
st.markdown('<p class="subtitle">Multi-task transformer models for various NLP tasks</p>', unsafe_allow_html=True)

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")

    default_base_path = "."
    base_path_input = st.text_input(
        "Base Directory Path",
        value=default_base_path,
        help="Path containing Task folders"
    )

    if st.button("Scan Directory", use_container_width=True):
        st.session_state.base_path = base_path_input
        st.rerun()

    st.divider()

    # Display available tasks
    if st.session_state.base_path:
        available_tasks = find_task_directories(st.session_state.base_path)

        if available_tasks:
            st.success(f"Found {len(available_tasks)} model(s)")

            task_options = list(available_tasks.keys())
            selected_task = st.selectbox(
                "Select Model",
                options=task_options,
                help="Choose model to load"
            )

            task_info = available_tasks[selected_task]

            # Display task information
            task_type_display = {
                "sentiment_classification": "Sentiment Classification",
                "code_generation": "Code Generation",
                "summarization": "Text Summarization"
            }

            st.markdown(f"""
            <div class="status-info">
                <strong>Task:</strong> {task_type_display.get(task_info['type'], task_info['type'])}<br>
                <strong>Model:</strong> {task_info['model_type'].upper()}<br>
                <strong>Path:</strong> .../{Path(task_info['path']).name}
            </div>
            """, unsafe_allow_html=True)

            if st.button("Load Model", type="primary", use_container_width=True):
                with st.spinner(f"Loading {selected_task}..."):
                    model_path = task_info['path']

                    # Show detection info
                    task_type_detected, model_type_detected = detect_task_type(model_path)
                    st.info(f"Detected: {task_type_detected} ({model_type_detected})")

                    (model, tokenizer, error), task_type = load_model_auto(model_path)

                    if error:
                        st.error(error)
                        # Show additional debug info
                        st.error(f"Task type detected as: {task_type}")
                        st.error(f"Model path: {model_path}")
                    else:
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.model_loaded = True
                        st.session_state.current_task = selected_task
                        st.session_state.task_type = task_type
                        st.success(f"{selected_task} loaded successfully!")
                        st.rerun()
        else:
            st.warning("No models found in directory")

    st.divider()

    # Model status
    if st.session_state.model_loaded:
        task_type_display = {
            "sentiment_classification": "Sentiment Analysis",
            "code_generation": "Code Generation",
            "summarization": "Text Summarization"
        }

        st.markdown(f"""
        <div class="status-success">
            <strong>✓ Model Active</strong><br>
            {st.session_state.current_task}<br>
            Task: {task_type_display.get(st.session_state.task_type, "Unknown")}<br>
            Device: {"CUDA" if torch.cuda.is_available() else "CPU"}
        </div>
        """, unsafe_allow_html=True)

        if st.button("Unload Model", use_container_width=True):
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.model_loaded = False
            st.session_state.current_task = None
            st.session_state.task_type = None
            st.cache_resource.clear()
            st.rerun()
    else:
        st.info("No model loaded")

    # Help section
    st.divider()
    with st.expander("ℹ️ Help & Information"):
        st.markdown("""
        **Task Descriptions:**

        **Task 1: Sentiment Classification**
        - Model: BERT-based classifier
        - Input: Customer feedback text
        - Output: Positive/Negative/Neutral sentiment with confidence scores

        **Task 2: Code Generation**
        - Model: GPT-2 based generator
        - Input: Pseudocode instructions
        - Output: Executable Python code

        **Task 3: Text Summarization**
        - Model: T5 encoder-decoder
        - Input: Long-form text
        - Output: Concise summary

        **Troubleshooting:**
        - Ensure model files are in correct directories
        - Check that config.json exists in model folder
        - Verify GPU/CUDA availability for faster inference
        """)

# Main content area
if not st.session_state.model_loaded:
    st.markdown("""
    <div class="status-info">
        <h3>Getting Started</h3>
        <ol>
            <li>Enter your base directory path in the sidebar</li>
            <li>Click "Scan Directory" to find available models</li>
            <li>Select a model from the dropdown</li>
            <li>Click "Load Model" to begin</li>
        </ol>
        <p><strong>Supported Tasks:</strong></p>
        <ul>
            <li><strong>Task 1:</strong> Sentiment Classification (BERT)</li>
            <li><strong>Task 2:</strong> Code Generation (GPT-2)</li>
            <li><strong>Task 3:</strong> Text Summarization (T5)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    # Dynamic UI based on task type
    if st.session_state.task_type == "sentiment_classification":
        st.subheader("📊 Sentiment Analysis")
        input_text = st.text_area(
            "Enter customer feedback",
            height=150,
            placeholder="Type or paste customer feedback here..."
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("Clear", use_container_width=True)

        if clear_btn:
            st.rerun()

        if analyze_btn:
            if not input_text.strip():
                st.warning("Please enter text to analyze")
            else:
                with st.spinner("Analyzing sentiment..."):
                    start_time = time.time()
                    sentiment, confidence, probs, error = classify_sentiment(
                        input_text,
                        st.session_state.model,
                        st.session_state.tokenizer
                    )
                    end_time = time.time()

                if error:
                    st.markdown(f"""
                    <div class="status-error">
                        <strong>Error:</strong> {error}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display sentiment
                    sentiment_class = f"sentiment-{sentiment.lower()}"
                    st.markdown(f"""
                    <div class="output-box">
                        <h3>Predicted Sentiment</h3>
                        <div class="sentiment-badge {sentiment_class}">{sentiment}</div>
                        <p style="margin-top: 16px;"><strong>Confidence:</strong> {confidence * 100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show all probabilities
                    st.subheader("Confidence Scores")
                    labels = ["Negative", "Neutral", "Positive"]
                    for i, (label, prob) in enumerate(zip(labels, probs)):
                        st.markdown(f"""
                        <div style="margin: 12px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                <span><strong>{label}</strong></span>
                                <span>{prob * 100:.2f}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {prob * 100}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Processing Time</div>
                            <div class="metric-value">{(end_time - start_time):.3f}s</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Input Length</div>
                            <div class="metric-value">{len(input_text.split())}</div>
                        </div>
                        """, unsafe_allow_html=True)

    elif st.session_state.task_type == "code_generation":
        st.subheader("💻 Pseudocode to Python Code Generation")
        st.markdown("""
        <div class="status-info">
            <strong>Format:</strong> Enter pseudocode instructions in natural language. The model will generate Python code.
        </div>
        """, unsafe_allow_html=True)

        # Add example pseudocode
        with st.expander("📖 View Example Pseudocode"):
            st.markdown("""
            **Example 1: Simple Loop**
            ```
            create a variable i and initialize it to 0
            while i is less than 5
                print the value of i
                increment i by 1
            ```

            **Example 2: Function**
            ```
            define a function called calculate_sum that takes two parameters a and b
            add a and b
            return the result
            ```

            **Example 3: List Operations**
            ```
            create an empty list called numbers
            for each value from 1 to 10
                append the value to numbers
            print the numbers list
            ```
            """)

        input_text = st.text_area(
            "Enter pseudocode instructions",
            height=200,
            placeholder="Example:\ncreate a variable x and set it to 10\nif x is greater than 5\n    print 'x is large'\nelse\n    print 'x is small'",
            help="Enter step-by-step instructions in natural language"
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            generate_btn = st.button("Generate Code", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("Clear", use_container_width=True)

        if clear_btn:
            st.rerun()

        if generate_btn:
            if not input_text.strip():
                st.warning("Please enter pseudocode instructions")
            else:
                with st.spinner("Generating Python code..."):
                    start_time = time.time()
                    code, error = generate_code(
                        input_text,
                        st.session_state.model,
                        st.session_state.tokenizer
                    )
                    end_time = time.time()

                if error:
                    st.markdown(f"""
                    <div class="status-error">
                        <strong>Error:</strong> {error}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.subheader("Generated Python Code")

                    # Display the code with syntax highlighting
                    st.code(code, language="python", line_numbers=True)

                    # Add execution warning
                    st.markdown("""
                    <div class="status-info">
                        <strong>⚠️ Note:</strong> Always review and test generated code before using it in production. 
                        The model may produce code that needs adjustments.
                    </div>
                    """, unsafe_allow_html=True)

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Generation Time</div>
                            <div class="metric-value">{(end_time - start_time):.2f}s</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Lines of Code</div>
                            <div class="metric-value">{len(code.split(chr(10)))}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Characters</div>
                            <div class="metric-value">{len(code)}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Download button
                    st.download_button(
                        label="💾 Download Python Code",
                        data=code,
                        file_name="generated_code.py",
                        mime="text/x-python",
                        use_container_width=True
                    )

    elif st.session_state.task_type == "summarization":
        st.subheader("📝 Text Summarization")
        input_text = st.text_area(
            "Enter text to summarize",
            height=200,
            placeholder="Paste your text here..."
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            generate_btn = st.button("Summarize", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("Clear", use_container_width=True)

        if clear_btn:
            st.rerun()

        if generate_btn:
            if not input_text.strip():
                st.warning("Please enter text to summarize")
            else:
                with st.spinner("Generating summary..."):
                    start_time = time.time()
                    summary, error = generate_summary(
                        input_text,
                        st.session_state.model,
                        st.session_state.tokenizer
                    )
                    end_time = time.time()

                if error:
                    st.markdown(f"""
                    <div class="status-error">
                        <strong>Error:</strong> {error}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.subheader("Summary")
                    st.markdown(f"""
                    <div class="output-box">
                        {summary}
                    </div>
                    """, unsafe_allow_html=True)

                    # Metrics
                    input_words = len(input_text.split())
                    output_words = len(summary.split())
                    compression = input_words / output_words if output_words > 0 else 0

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Input</div>
                            <div class="metric-value">{input_words}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Output</div>
                            <div class="metric-value">{output_words}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Ratio</div>
                            <div class="metric-value">{compression:.1f}x</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Time</div>
                            <div class="metric-value">{(end_time - start_time):.2f}s</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )