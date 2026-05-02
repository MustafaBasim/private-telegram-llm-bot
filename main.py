import os
import sys
import json
import subprocess
import time
import base64
import requests
import psutil
import logging
import re
import html
import asyncio
from telegramify_markdown import markdownify
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
ALLOWED_USER_IDS = [int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()]
LMSTUDIO_DIR = r"D:\LLM\lmstudio"
CONFIG_FILE = "config.json"

LLAMA_SERVER_PROCESS = None
CURRENT_MODEL = None
CHAT_HISTORY = {}  # {user_id: [{"role": "user", "content": "..."}]}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": [], "system_messages": {}}

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def scan_models():
    models = []
    for root, dirs, files in os.walk(LMSTUDIO_DIR):
        gguf_files = [f for f in files if f.endswith(".gguf")]
        
        # Check if there is any mmproj file in this directory
        mmproj_files = [f for f in gguf_files if "mmproj" in f.lower()]
        has_mmproj = len(mmproj_files) > 0
        mmproj_path = os.path.join(root, mmproj_files[0]) if has_mmproj else None

        for f in gguf_files:
            if "mmproj" in f.lower():
                continue # Skip the mmproj files themselves as primary models
            
            model_path = os.path.join(root, f)
            name = f.replace(".gguf", "")
            
            models.append({
                "name": name,
                "path": model_path,
                "ctx_len": 8192, # Default context
                "multimodal": has_mmproj,
                "mmproj_path": mmproj_path,
                "alias": name
            })
    
    config = load_config()
    # Keep existing system messages and context lengths if model already exists
    existing_models = {m["path"]: m for m in config.get("models", [])}
    
    updated_models = []
    for new_m in models:
        if new_m["path"] in existing_models:
            ext_m = existing_models[new_m["path"]]
            new_m["ctx_len"] = ext_m.get("ctx_len", 8192)
            new_m["alias"] = ext_m.get("alias", new_m["name"])
        updated_models.append(new_m)
        
    config["models"] = updated_models
    save_config(config)
    return updated_models

def restricted(func):
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USER_IDS:
            print(f"Unauthorized access denied for {user_id}.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to your local Llama Server bot!\nUse /models to see available models.\nUse /update_models to scan for new models.")

@restricted
async def update_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Scanning for models...")
    models = scan_models()
    await update.message.reply_text(f"Found {len(models)} models. Use /models to select one.")

@restricted
async def list_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    config = load_config()
    models = config.get("models", [])
    if not models:
        await update.message.reply_text("No models found. Run /update_models first.")
        return

    keyboard = []
    for i, model in enumerate(models):
        keyboard.append([InlineKeyboardButton(model["alias"], callback_data=f"load_{i}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select a model to load:", reply_markup=reply_markup)

@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    if data.startswith("load_"):
        index = int(data.split("_")[1])
        config = load_config()
        models = config.get("models", [])
        if index < len(models):
            model = models[index]
            await load_model(query.message, model, update.effective_user.id)

async def load_model(message, model, user_id):
    global LLAMA_SERVER_PROCESS, CURRENT_MODEL, CHAT_HISTORY
    
    if LLAMA_SERVER_PROCESS is not None:
        await message.reply_text("Unloading current model...")
        LLAMA_SERVER_PROCESS.terminate()
        LLAMA_SERVER_PROCESS.wait()
        LLAMA_SERVER_PROCESS = None

    await message.reply_text(f"Starting model: {model['alias']}...\nPlease wait.")
    
    cmd = [
        "llama-server",
        "-m", model["path"],
        "-c", str(model["ctx_len"]),
        "-fa", "on",
        "--jinja",
        "--port", "8080"
    ]
    if model.get("multimodal") and model.get("mmproj_path"):
        cmd.extend(["--mmproj", model["mmproj_path"]])
        
    if "thinking" in model:
        is_thinking = model["thinking"]
        model_name = model["name"].lower()
        
        kwargs = {}
        # Apply model-specific kwargs for thinking
        if "qwen" in model_name:
            kwargs["enable_thinking"] = is_thinking
            kwargs["preserve_thinking"] = is_thinking
        elif "gemma" in model_name:
            kwargs["enable_thinking"] = is_thinking
        elif "gpt-oss" in model_name:
            kwargs["enable_thinking"] = is_thinking
            kwargs["preserve_thinking"] = is_thinking
            kwargs["reasoning_effort"] = "high" if is_thinking else "none"
        else:
            kwargs["enable_thinking"] = is_thinking
            kwargs["preserve_thinking"] = is_thinking
            
        cmd.extend(["--chat-template-kwargs", json.dumps(kwargs)])
        
    try:
        LLAMA_SERVER_PROCESS = subprocess.Popen(cmd)
    except FileNotFoundError:
        await message.reply_text("Error: llama-server not found. Make sure it's in your system PATH.")
        return

    # Wait for server to be up
    ready = False
    for _ in range(30):
        try:
            resp = await asyncio.to_thread(requests.get, "http://127.0.0.1:8080/v1/models")
            if resp.status_code == 200:
                ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        await asyncio.sleep(2)

    if ready:
        CURRENT_MODEL = model
        CHAT_HISTORY[user_id] = []
        cfg = load_config()
        cfg["last_loaded_model"] = model["path"]
        save_config(cfg)
        await message.reply_text(f"Model {model['alias']} is ready! You can now chat with it.")
    else:
        LLAMA_SERVER_PROCESS.terminate()
        LLAMA_SERVER_PROCESS = None
        await message.reply_text("Failed to start server within 60 seconds.")

@restricted
async def unload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LLAMA_SERVER_PROCESS, CURRENT_MODEL
    if LLAMA_SERVER_PROCESS:
        LLAMA_SERVER_PROCESS.terminate()
        LLAMA_SERVER_PROCESS.wait()
        LLAMA_SERVER_PROCESS = None
        CURRENT_MODEL = None
        await update.message.reply_text("Model unloaded.")
    else:
        await update.message.reply_text("No model is currently loaded.")

@restricted
async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LLAMA_SERVER_PROCESS, CURRENT_MODEL
    if not CURRENT_MODEL:
        await update.message.reply_text("No model is currently loaded to reload.")
        return
        
    model_to_reload = CURRENT_MODEL.copy()
    await update.message.reply_text(f"Reloading {model_to_reload['alias']}...")
    
    if LLAMA_SERVER_PROCESS:
        LLAMA_SERVER_PROCESS.terminate()
        LLAMA_SERVER_PROCESS.wait()
        LLAMA_SERVER_PROCESS = None
        
    CURRENT_MODEL = None
    CHAT_HISTORY[update.effective_user.id] = []
    
    await load_model(update.message, model_to_reload, update.effective_user.id)

@restricted
async def shutdown_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LLAMA_SERVER_PROCESS
    if LLAMA_SERVER_PROCESS:
        LLAMA_SERVER_PROCESS.terminate()
        LLAMA_SERVER_PROCESS.wait()
    await update.message.reply_text("Shutting down the bot and server...")
    os._exit(0)

@restricted
async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    CHAT_HISTORY[user_id] = []
    await update.message.reply_text("Chat history cleared.")

@restricted
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    vm = psutil.virtual_memory()
    ram_total_gb = vm.total / (1024**3)
    ram_used_gb = vm.used / (1024**3)
    ram_pct = vm.percent
    ram_str = f"RAM: {ram_used_gb:.1f} GB / {ram_total_gb:.1f} GB ({ram_pct}%)"
    
    vram_str = "VRAM: Unknown"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            text=True
        )
        lines = output.strip().split("\n")
        parts = lines[0].replace(" MiB", "").split(",")
        if len(parts) == 2:
            used_mib = float(parts[0].strip())
            total_mib = float(parts[1].strip())
            used_gb = used_mib / 1024
            total_gb = total_mib / 1024
            pct = (used_mib / total_mib) * 100 if total_mib > 0 else 0
            vram_str = f"VRAM: {used_gb:.1f} GB / {total_gb:.1f} GB ({pct:.1f}%)"
        else:
            vram_str = f"VRAM: {lines[0]}"
    except Exception:
        pass
        
    status = f"System Status:\n{ram_str}\n{vram_str}\n"
    if CURRENT_MODEL:
        status += f"\nLoaded Model: {CURRENT_MODEL['alias']}"
        status += f"\nContext Length: {CURRENT_MODEL.get('ctx_len')}"
        status += f"\nMultimodal: {CURRENT_MODEL.get('multimodal')}"
        if "thinking" in CURRENT_MODEL:
            status += f"\nThinking Enabled: {CURRENT_MODEL['thinking']}"
        status += f"\nSend Thinking Block: {CURRENT_MODEL.get('send_thinking', True)}"
        
        cfg = load_config()
        sys_msg = cfg.get("system_messages", {}).get(CURRENT_MODEL["path"], "None")
        status += f"\nSystem Message: {sys_msg}"
        status += f"\nPath: {CURRENT_MODEL.get('path')}"
    else:
        status += "\nNo model loaded."
        
    await update.message.reply_text(status)

@restricted
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 Bot Commands List:\n\n"
        "/start - Welcome message\n"
        "/models - List available local models\n"
        "/update_models - Scan for new models in D:\\LLM\\lmstudio\n"
        "/load - Load the last loaded model\n"
        "/unload - Unload the currently running model\n"
        "/shutdown - Shut down the bot and the llama server entirely\n"
        "/clear - Clear the current chat history\n"
        "/status - Show system RAM and VRAM usage\n"
        "/set_context <value> - Set the context length for the active model (e.g., 8192)\n"
        "/set_thinking <true/false> - Enable or disable thinking mode for the active model\n"
        "/set_send_thinking <true/false> - Send the thinking block in chat or not\n"
        "/set_chat_status <true/false> - Show generation stats at the bottom of messages\n"
        "/set_system_message <message> - Set a system prompt for the active model\n"
        "/get_system_message - View the active model's system prompt\n"
        "/clear_system_message - Clear the active model's system prompt\n"
        "/help - Show this commands list\n"
    )
    await update.message.reply_text(help_text)

@restricted
async def load_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = load_config()
    last_path = cfg.get("last_loaded_model")
    if not last_path:
        await update.message.reply_text("No model was previously loaded. Use /models to select one.")
        return
        
    models = cfg.get("models", [])
    target_model = next((m for m in models if m["path"] == last_path), None)
    
    if target_model:
        await load_model(update.message, target_model, update.effective_user.id)
    else:
        await update.message.reply_text("The previously loaded model was not found in the current configuration. Please use /models.")

@restricted
async def set_context(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /set_context <value>")
        return
    try:
        ctx_val = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Please provide a valid integer.")
        return
        
    if not CURRENT_MODEL:
        await update.message.reply_text("Please load a model first before setting its context.")
        return
        
    cfg = load_config()
    for m in cfg.get("models", []):
        if m["path"] == CURRENT_MODEL["path"]:
            m["ctx_len"] = ctx_val
            CURRENT_MODEL["ctx_len"] = ctx_val
            break
            
    save_config(cfg)
    await update.message.reply_text(f"Context length for {CURRENT_MODEL['alias']} set to {ctx_val}. Please /unload and reload for changes to take effect.")

@restricted
async def toggle_thinking(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CURRENT_MODEL:
        await update.message.reply_text("Please load a model first before toggling its thinking mode.")
        return
        
    current_val = CURRENT_MODEL.get("thinking", True)
    new_val = not current_val
        
    cfg = load_config()
    for m in cfg.get("models", []):
        if m["path"] == CURRENT_MODEL["path"]:
            m["thinking"] = new_val
            CURRENT_MODEL["thinking"] = new_val
            break
            
    save_config(cfg)
    await update.message.reply_text(f"Thinking mode for {CURRENT_MODEL['alias']} toggled to {new_val}. Please /reload for changes to take effect.")

@restricted
async def toggle_send_thinking(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CURRENT_MODEL:
        await update.message.reply_text("Please load a model first before toggling this preference.")
        return
        
    current_val = CURRENT_MODEL.get("send_thinking", True)
    new_val = not current_val
        
    cfg = load_config()
    for m in cfg.get("models", []):
        if m["path"] == CURRENT_MODEL["path"]:
            m["send_thinking"] = new_val
            CURRENT_MODEL["send_thinking"] = new_val
            break
            
    save_config(cfg)
    await update.message.reply_text(f"Send thinking blocks for {CURRENT_MODEL['alias']} toggled to {new_val}.")

@restricted
async def toggle_chat_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CURRENT_MODEL:
        await update.message.reply_text("Please load a model first before toggling this preference.")
        return
        
    current_val = CURRENT_MODEL.get("send_chat_status", False)
    new_val = not current_val
        
    cfg = load_config()
    for m in cfg.get("models", []):
        if m["path"] == CURRENT_MODEL["path"]:
            m["send_chat_status"] = new_val
            CURRENT_MODEL["send_chat_status"] = new_val
            break
            
    save_config(cfg)
    await update.message.reply_text(f"Chat status display for {CURRENT_MODEL['alias']} toggled to {new_val}.")

@restricted
async def set_system_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CURRENT_MODEL:
        await update.message.reply_text("Please load a model first to set its system message.")
        return
        
    msg = " ".join(context.args)
    if not msg:
        await update.message.reply_text("Usage: /set_system_message <message>")
        return
        
    cfg = load_config()
    cfg.setdefault("system_messages", {})[CURRENT_MODEL["path"]] = msg
    save_config(cfg)
    
    # Also update history to include it at the start if clear
    user_id = update.effective_user.id
    if user_id in CHAT_HISTORY:
         # remove old system messages from active history to prevent conflicts, though standard is prepending it per request.
         CHAT_HISTORY[user_id] = [m for m in CHAT_HISTORY[user_id] if m["role"] != "system"]
         
    await update.message.reply_text(f"System message set for {CURRENT_MODEL['alias']}.")

@restricted
async def get_system_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CURRENT_MODEL:
        await update.message.reply_text("Please load a model first.")
        return
    
    cfg = load_config()
    msg = cfg.get("system_messages", {}).get(CURRENT_MODEL["path"], "No custom system message set.")
    await update.message.reply_text(f"System message for {CURRENT_MODEL['alias']}:\n{msg}")

@restricted
async def clear_system_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CURRENT_MODEL:
        await update.message.reply_text("Please load a model first.")
        return
        
    cfg = load_config()
    if CURRENT_MODEL["path"] in cfg.get("system_messages", {}):
        del cfg["system_messages"][CURRENT_MODEL["path"]]
        save_config(cfg)
        
    user_id = update.effective_user.id
    if user_id in CHAT_HISTORY:
         CHAT_HISTORY[user_id] = [m for m in CHAT_HISTORY[user_id] if m["role"] != "system"]

    await update.message.reply_text(f"System message cleared for {CURRENT_MODEL['alias']}.")

@restricted
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CURRENT_MODEL:
        cfg = load_config()
        last_path = cfg.get("last_loaded_model")
        if last_path:
            models = cfg.get("models", [])
            target_model = next((m for m in models if m["path"] == last_path), None)
            if target_model:
                await load_model(update.message, target_model, update.effective_user.id)
                if not CURRENT_MODEL:
                    return
            else:
                await update.message.reply_text("Please load a model first using /models.")
                return
        else:
            await update.message.reply_text("Please load a model first using /models.")
            return
        
    user_id = update.effective_user.id
    if user_id not in CHAT_HISTORY:
        CHAT_HISTORY[user_id] = []
        
    text = update.message.text or update.message.caption or ""
    
    # Handle document/file if present
    if update.message.document:
        doc = update.message.document
        file_name = doc.file_name
        
        file = await doc.get_file()
        file_bytes = await file.download_as_bytearray()
        
        try:
            file_text = file_bytes.decode('utf-8')
            text = f"--- Content of {file_name} ---\n{file_text}\n--- End of {file_name} ---\n\n{text}"
        except UnicodeDecodeError:
            await update.message.reply_text(f"Sorry, I can only read plain-text files right now. {file_name} seems to be binary or an unsupported format.")
            return

    content = text
    
    # Handle image if present
    if update.message.photo:
        if not CURRENT_MODEL.get("multimodal"):
            await update.message.reply_text("Current model does not support images.")
            return
            
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_img = base64.b64encode(photo_bytes).decode('utf-8')
        
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        ]
        
    CHAT_HISTORY[user_id].append({"role": "user", "content": content})
    
    # Prepare payload
    cfg = load_config()
    sys_msg = cfg.get("system_messages", {}).get(CURRENT_MODEL["path"])
    
    messages = []
    
    # If thinking is explicitly disabled, forcefully inject a system instruction
    if "thinking" in CURRENT_MODEL and not CURRENT_MODEL["thinking"]:
        disable_think_msg = "You are a direct assistant. Do NOT output reasoning or <think> blocks."
        if sys_msg:
            sys_msg += "\n\n" + disable_think_msg
        else:
            sys_msg = disable_think_msg

    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
        
    messages.extend(CHAT_HISTORY[user_id])
    
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "stream": False
    }

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        response = await asyncio.to_thread(requests.post, "http://127.0.0.1:8080/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        
        reasoning_content = data["choices"][0]["message"].get("reasoning_content", "")
        reply_text = data["choices"][0]["message"].get("content", "")
        
        # Fallback for models that embed <think> in the main content
        if not reasoning_content:
            match = re.search(r'<think>(.*?)</think>\n?', reply_text, flags=re.DOTALL)
            if match:
                reasoning_content = match.group(1).strip()
                reply_text = re.sub(r'<think>.*?</think>\n?', '', reply_text, flags=re.DOTALL).strip()
        
        if reasoning_content and CURRENT_MODEL.get("send_thinking", True):
            reasoning_msg = f"<b>Reasoning:</b>\n{html.escape(reasoning_content.strip())}"
            await update.message.reply_text(reasoning_msg, parse_mode="HTML")
            
            # Send typing action again for the main response
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
            await asyncio.sleep(0.5)
            
        # If user wants it hidden, and the resulting message is empty
        if not CURRENT_MODEL.get("send_thinking", True) and not reply_text and reasoning_content:
            reply_text = "*(Thinking block hidden)*"
                
        CHAT_HISTORY[user_id].append({"role": "assistant", "content": reply_text})
        
        try:
            md_text = markdownify(reply_text)
            await update.message.reply_text(md_text, parse_mode="MarkdownV2")
        except Exception:
            await update.message.reply_text(reply_text)
        
        # Send chat status if enabled
        if CURRENT_MODEL.get("send_chat_status", False):
            usage = data.get("usage", {})
            timings = data.get("timings", {})
            pp = timings.get("prompt_per_second", 0)
            tg = timings.get("predicted_per_second", 0)
            total_tokens = usage.get("total_tokens", 0)
            ctx_len = CURRENT_MODEL.get("ctx_len", 8192)
            ctx_pct = (total_tokens / ctx_len * 100) if ctx_len > 0 else 0
            
            status_text = f"PP: {pp:.2f} t/s | TG: {tg:.2f} t/s | Ctx: {total_tokens}/{ctx_len} ({ctx_pct:.1f}%)"
            await update.message.reply_text(status_text)
    except Exception as e:
        await update.message.reply_text(f"Error communicating with model: {e}")
        # remove the last user message on error so they can retry
        if CHAT_HISTORY[user_id]:
            CHAT_HISTORY[user_id].pop()

async def post_init(application: Application):
    commands = [
        BotCommand("start", "Welcome message"),
        BotCommand("models", "List available local models"),
        BotCommand("update_models", "Scan for new models"),
        BotCommand("load", "Load the last loaded model"),
        BotCommand("unload", "Unload the currently running model"),
        BotCommand("reload", "Quickly reload the current model"),
        BotCommand("shutdown", "Shut down the bot and server"),
        BotCommand("clear", "Clear the current chat history"),
        BotCommand("status", "Show system RAM and VRAM usage"),
        BotCommand("set_context", "Set context length (e.g., 8192)"),
        BotCommand("toggle_thinking", "Toggle thinking mode on/off"),
        BotCommand("toggle_send_thinking", "Toggle sending thinking block in chat"),
        BotCommand("toggle_chat_status", "Toggle showing generation stats"),
        BotCommand("set_system_message", "Set system prompt"),
        BotCommand("get_system_message", "View system prompt"),
        BotCommand("clear_system_message", "Clear system prompt"),
        BotCommand("help", "Show commands list")
    ]
    await application.bot.set_my_commands(commands)

def main():
    if not BOT_TOKEN:
        print("BOT_TOKEN is not set in .env")
        return
        
    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("update_models", update_models))
    application.add_handler(CommandHandler("models", list_models))
    application.add_handler(CommandHandler("load", load_cmd))
    application.add_handler(CommandHandler("unload", unload_cmd))
    application.add_handler(CommandHandler("reload", reload_cmd))
    application.add_handler(CommandHandler("shutdown", shutdown_cmd))
    application.add_handler(CommandHandler("clear", clear_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("set_context", set_context))
    application.add_handler(CommandHandler("toggle_thinking", toggle_thinking))
    application.add_handler(CommandHandler("toggle_send_thinking", toggle_send_thinking))
    application.add_handler(CommandHandler("toggle_chat_status", toggle_chat_status))
    application.add_handler(CommandHandler("set_system_message", set_system_message))
    application.add_handler(CommandHandler("get_system_message", get_system_message))
    application.add_handler(CommandHandler("clear_system_message", clear_system_message))
    
    application.add_handler(CallbackQueryHandler(button_callback))
    
    application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.Document.ALL, handle_message))

    print("Bot is starting...")
    application.run_polling()

if __name__ == "__main__":
    main()
