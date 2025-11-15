#!/usr/bin/env python3
"""
Unified Multi-Task Transformer App Launcher
Supports: Sentiment Classification (BERT), Code Generation (GPT-2), Text Summarization (T5)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'streamlit': 'streamlit',
        'torch': 'torch',
        'transformers': 'transformers',
        'peft': 'peft'
    }

    missing_packages = []
    installed_packages = {}

    print("🔍 Checking dependencies...")
    print("=" * 70)

    for package, pip_name in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            installed_packages[package] = version
            print(f"✅ {package:<20} {version}")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"❌ {package:<20} NOT INSTALLED")

    print("=" * 70)

    if missing_packages:
        print("\n❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True


def get_task_info(task_path):
    """Get detailed information about a task directory"""
    task_path = Path(task_path)

    if not task_path.exists():
        return None

    info = {
        'path': str(task_path),
        'exists': True,
        'files': {},
        'task_type': 'unknown',
        'model_type': 'unknown'
    }

    # Check for key files
    key_files = [
        'config.json',
        'model.safetensors',
        'pytorch_model.bin',
        'adapter_config.json',
        'adapter_model.safetensors',
        'adapter_model.bin',
        'tokenizer_config.json',
        'vocab.txt',
        'vocab.json',
        'merges.txt',  # GPT-2 specific
        'added_tokens.json',
        'special_tokens_map.json'
    ]

    for file in key_files:
        file_path = task_path / file
        info['files'][file] = file_path.exists()

    # Determine model type
    has_adapter = info['files'].get('adapter_config.json', False)
    has_full_model = info['files'].get('model.safetensors', False) or info['files'].get('pytorch_model.bin', False)

    if has_adapter:
        info['model_type'] = 'lora'
    elif has_full_model:
        info['model_type'] = 'full'

    # Check for GPT-2 specific files (merges.txt + vocab.json = GPT-2)
    has_merges = info['files'].get('merges.txt', False)
    has_vocab_json = info['files'].get('vocab.json', False)
    if has_merges and has_vocab_json:
        info['task_type'] = 'code_generation'
        info['base_model'] = 'GPT-2'

    # Try to determine task type from adapter_config.json
    if has_adapter and info['task_type'] == 'unknown':
        adapter_config = task_path / 'adapter_config.json'
        try:
            with open(adapter_config, 'r') as f:
                adapter_conf = json.load(f)
                base_model = adapter_conf.get('base_model_name_or_path', '').lower()

                if 'gpt' in base_model:
                    info['task_type'] = 'code_generation'
                    info['base_model'] = 'GPT-2'
                elif 'bert' in base_model:
                    info['task_type'] = 'sentiment_classification'
                    info['base_model'] = 'BERT'
                elif 't5' in base_model:
                    info['task_type'] = 'summarization'
                    info['base_model'] = 'T5'
        except:
            pass

    # Try to determine task type from config.json
    if info['task_type'] == 'unknown':
        config_path = task_path / 'config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_type = config.get('model_type', '').lower()
                    architectures = config.get('architectures', [])

                    # Check architectures
                    for arch in architectures:
                        arch_lower = arch.lower()
                        if 'gpt' in arch_lower:
                            info['task_type'] = 'code_generation'
                            info['base_model'] = 'GPT-2'
                            break
                        elif 'bert' in arch_lower:
                            info['task_type'] = 'sentiment_classification'
                            info['base_model'] = 'BERT'
                            break
                        elif 't5' in arch_lower:
                            info['task_type'] = 'summarization'
                            info['base_model'] = 'T5'
                            break

                    # Fallback to model_type
                    if info['task_type'] == 'unknown':
                        if 'gpt' in model_type or 'gpt2' in model_type:
                            info['task_type'] = 'code_generation'
                            info['base_model'] = 'GPT-2'
                        elif 'bert' in model_type:
                            info['task_type'] = 'sentiment_classification'
                            info['base_model'] = 'BERT'
                        elif 't5' in model_type:
                            info['task_type'] = 'summarization'
                            info['base_model'] = 'T5'

                    info['num_labels'] = config.get('num_labels', 'N/A')
            except Exception as e:
                info['config_error'] = str(e)

    # Check tokenizer to determine model type
    if info['task_type'] == 'unknown':
        tokenizer_config = task_path / 'tokenizer_config.json'
        if tokenizer_config.exists():
            try:
                with open(tokenizer_config, 'r') as f:
                    tok_config = json.load(f)
                    tokenizer_class = tok_config.get('tokenizer_class', '').lower()

                    if 'gpt2' in tokenizer_class:
                        info['task_type'] = 'code_generation'
                        info['base_model'] = 'GPT-2'
                    elif 'bert' in tokenizer_class:
                        info['task_type'] = 'sentiment_classification'
                        info['base_model'] = 'BERT'
                    elif 't5' in tokenizer_class:
                        info['task_type'] = 'summarization'
                        info['base_model'] = 'T5'
            except:
                pass

    # Fallback task type detection from directory name
    if info['task_type'] == 'unknown':
        dir_name = task_path.name.lower()
        if 'task1' in dir_name:
            info['task_type'] = 'sentiment_classification'
            info['base_model'] = 'BERT (assumed)'
        elif 'task2' in dir_name:
            info['task_type'] = 'code_generation'
            info['base_model'] = 'GPT-2 (assumed)'
        elif 'task3' in dir_name:
            info['task_type'] = 'summarization'
            info['base_model'] = 'T5 (assumed)'

    return info


def scan_base_directory(base_path):
    """Scan base directory for all task folders"""
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"❌ Base directory not found: {base_path}")
        return {}

    print(f"\n📁 Scanning base directory: {base_path}")
    print("=" * 80)

    tasks = {}
    task_type_names = {
        'sentiment_classification': 'Sentiment Classification (BERT)',
        'code_generation': 'Code Generation (GPT-2)',
        'summarization': 'Text Summarization (T5)',
        'unknown': 'Unknown Task Type'
    }

    # Check for Task1, Task2, Task3
    for i in range(1, 4):
        task_name = f"Task{i}"
        task_path = base_path / task_name

        print(f"\n{'─' * 80}")
        print(f"📂 {task_name}")
        print(f"{'─' * 80}")

        if task_path.exists():
            info = get_task_info(task_path)
            if info:
                tasks[task_name] = info

                # Display information
                print(f"✅ Directory found: {task_path}")
                print(f"   Task Type: {task_type_names.get(info['task_type'], info['task_type'])}")
                print(f"   Model Type: {info['model_type'].upper()}")
                if 'base_model' in info:
                    print(f"   Base Model: {info['base_model']}")
                if 'num_labels' in info and info['num_labels'] != 'N/A':
                    print(f"   Num Labels: {info['num_labels']}")

                print(f"\n   Key Files:")
                critical_files = {
                    'config.json': 'Model configuration',
                    'model.safetensors': 'Full model weights',
                    'adapter_config.json': 'LoRA adapter config',
                    'adapter_model.safetensors': 'LoRA adapter weights',
                    'tokenizer_config.json': 'Tokenizer configuration',
                    'vocab.json': 'Vocabulary (GPT-2)',
                    'merges.txt': 'BPE merges (GPT-2)',
                    'special_tokens_map.json': 'Special tokens'
                }

                for file, desc in critical_files.items():
                    if file in info['files']:
                        status = "✅" if info['files'][file] else "❌"
                        print(f"   {status} {file:<30} - {desc}")

                # Validation
                is_valid = False
                if info['model_type'] == 'lora':
                    is_valid = (
                            info['files'].get('adapter_config.json', False) and
                            (info['files'].get('adapter_model.safetensors', False) or
                             info['files'].get('adapter_model.bin', False))
                    )
                elif info['model_type'] == 'full':
                    is_valid = (
                            info['files'].get('config.json', False) and
                            (info['files'].get('model.safetensors', False) or
                             info['files'].get('pytorch_model.bin', False))
                    )

                # Special validation for Task 2 (GPT-2 with special tokens)
                if task_name == 'Task2' and info['task_type'] == 'code_generation':
                    print(f"\n   🔍 Task 2 Special Validation:")

                    # Check for GPT-2 specific files
                    has_merges = info['files'].get('merges.txt', False)
                    has_vocab = info['files'].get('vocab.json', False)
                    print(f"   {'✅' if has_merges else '⚠️'} merges.txt (required for GPT-2)")
                    print(f"   {'✅' if has_vocab else '⚠️'} vocab.json (required for GPT-2)")

                    # Check for special tokens
                    special_tokens_file = task_path / 'special_tokens_map.json'
                    if special_tokens_file.exists():
                        try:
                            with open(special_tokens_file, 'r') as f:
                                special_tokens = json.load(f)
                                additional = special_tokens.get('additional_special_tokens', [])
                                has_pseudo = '<|pseudo|>' in str(additional)
                                has_code = '<|code|>' in str(additional)
                                print(f"   {'✅' if has_pseudo else '⚠️'} <|pseudo|> token")
                                print(f"   {'✅' if has_code else '⚠️'} <|code|> token")
                        except:
                            print(f"   ⚠️  Could not read special_tokens_map.json")

                    # Check tokenizer config for vocab size
                    tokenizer_config_file = task_path / 'tokenizer_config.json'
                    if tokenizer_config_file.exists():
                        try:
                            with open(tokenizer_config_file, 'r') as f:
                                tok_config = json.load(f)
                                model_max_length = tok_config.get('model_max_length', 'N/A')
                                print(f"   Max Length: {model_max_length}")
                        except:
                            pass

                if is_valid:
                    print(f"\n   ✅ Model is valid and ready to load")
                else:
                    print(f"\n   ⚠️  Model may be incomplete - missing required files")

                tasks[task_name]['valid'] = is_valid
            else:
                print(f"⚠️  Directory exists but could not read model info")
        else:
            print(f"❌ Directory not found")

    print(f"\n{'=' * 80}")

    valid_tasks = sum(1 for t in tasks.values() if t.get('valid', False))
    print(f"\n📊 Summary: {valid_tasks}/{len(tasks) if tasks else 0} valid task(s) found\n")

    return tasks


def validate_environment():
    """Validate the complete environment"""
    print("\n🔬 Environment Validation")
    print("=" * 70)

    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python Version: {python_version}")

    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA Available: Yes (v{torch.version.cuda})")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"⚠️  CUDA Available: No (CPU mode only)")
    except:
        print(f"❌ PyTorch not installed")

    print("=" * 70)


def run_streamlit_app(port=8501, base_path=None):
    """Launch the unified Streamlit app"""

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    print("\n✅ All dependencies installed\n")

    # Validate environment
    validate_environment()

    # Check base directory if provided
    if base_path:
        tasks = scan_base_directory(base_path)
        valid_tasks = sum(1 for t in tasks.values() if t.get('valid', False))

        if valid_tasks == 0:
            print("\n⚠️  No valid task models found!")
            print("   The app will still launch, but you'll need to configure paths manually.")
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    # App file name - try different possible names
    possible_names = ['streamlit_app.py', 'unified_summarizer_app.py', 'app.py']
    app_file = None

    for name in possible_names:
        potential_file = script_dir / name
        if potential_file.exists():
            app_file = potential_file
            break

    if not app_file:
        print(f"❌ Error: Streamlit app file not found in {script_dir}")
        print("\n💡 Looking for one of these files:")
        for name in possible_names:
            print(f"   - {name}")
        sys.exit(1)

    # Streamlit configuration
    config_args = [
        'streamlit', 'run',
        str(app_file),
        '--server.port', str(port),
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
        '--theme.primaryColor', '#1a73e8',
        '--theme.backgroundColor', '#f8f9fa',
        '--theme.secondaryBackgroundColor', '#ffffff',
        '--theme.textColor', '#202124',
        '--theme.font', 'sans serif'
    ]

    # Print startup info
    print("\n" + "=" * 70)
    print("🚀 Starting Unified Multi-Task Transformer App")
    print("=" * 70)
    print(f"\n📍 Local URL: http://localhost:{port}")
    print(f"🌐 Network URL: http://<your-ip>:{port}\n")
    print("💡 Supported Tasks:")
    print("   • Task 1: Sentiment Classification (BERT)")
    print("   • Task 2: Code Generation (GPT-2)")
    print("   • Task 3: Text Summarization (T5)")

    if base_path:
        print(f"\n📂 Base directory: {base_path}")

    print("\n💡 Features:")
    print("   • Automatic task type detection")
    print("   • Support for both full models and LoRA adapters")
    print("   • Dynamic UI based on task type")
    print("   • Real-time inference with metrics")

    print("\n⌨️  Press Ctrl+C to stop the server")
    print("\n" + "=" * 70 + "\n")

    try:
        # Run streamlit
        subprocess.run(config_args)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
        print("✅ Server stopped successfully")
    except Exception as e:
        print(f"\n❌ Error running Streamlit: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run the Unified Multi-Task Transformer App',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_unified_app.py
  python run_unified_app.py --port 8080
  python run_unified_app.py --base-path /path/to/models
  python run_unified_app.py --check-only
  python run_unified_app.py --scan

Task Structure:
  Base Directory/
  ├── Task1/          (BERT - Sentiment Classification)
  │   ├── config.json
  │   ├── model.safetensors (or adapter files)
  │   └── tokenizer files
  ├── Task2/          (GPT-2-Medium + LoRA - Pseudocode to Code)
  │   ├── config.json (model_type: gpt2)
  │   ├── adapter_config.json (target_modules: c_attn, c_proj, c_fc)
  │   ├── adapter_model.safetensors
  │   ├── tokenizer files
  │   ├── vocab.json (GPT-2 vocabulary)
  │   ├── merges.txt (GPT-2 BPE merges)
  │   └── special_tokens_map.json (includes <|pseudo|>, <|code|>)
  └── Task3/          (T5 - Text Summarization)
      ├── config.json
      ├── model.safetensors (or adapter files)
      └── tokenizer files

Notes:
  - Task 2 uses GPT-2-Medium as base model (not gpt2-small)
  - Task 2 requires special tokens: <|pseudo|>, <|code|>, <|endoftext|>, <|pad|>
  - Task 2 input format: "<|pseudo|> [pseudocode] <|code|>"
  - Task 2 tokenizer must have vocab_size matching model embeddings
        """
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port number for Streamlit server (default: 8501)'
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default='.',
        help='Base directory containing Task folders'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Check dependencies and environment without starting the app'
    )

    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan base directory and show detailed task information'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run full environment validation'
    )

    args = parser.parse_args()

    # Handle validation
    if args.validate:
        print("🔬 Running full validation...\n")
        deps_ok = check_dependencies()
        validate_environment()
        tasks = scan_base_directory(args.base_path)

        valid_tasks = sum(1 for t in tasks.values() if t.get('valid', False))

        print("\n" + "=" * 70)
        if deps_ok and valid_tasks > 0:
            print("✅ Validation passed! System is ready.")
        else:
            if not deps_ok:
                print("❌ Dependency check failed")
            if valid_tasks == 0:
                print("⚠️  No valid tasks found")
        print("=" * 70)
        sys.exit(0 if deps_ok else 1)

    # Handle scan
    if args.scan:
        scan_base_directory(args.base_path)
        sys.exit(0)

    # Handle check only
    if args.check_only:
        print("🔍 Running system checks...\n")
        deps_ok = check_dependencies()

        if deps_ok:
            print("\n✅ All dependencies installed")
            print("\n💡 To start the app, run:")
            print(f"   python {sys.argv[0]} --base-path {args.base_path}")
        else:
            print("\n❌ Please install missing dependencies first")
        sys.exit(0 if deps_ok else 1)

    # Run the app
    run_streamlit_app(port=args.port, base_path=args.base_path)


if __name__ == "__main__":
    main()