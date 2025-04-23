# --- START OF FILE bot.py ---

import asyncio
import sys
import os
import json
import logging
import datetime
import time
from collections import defaultdict, deque
import math # NaNチェック用に追加
import functools # run_in_executor用に追加

import discord
from discord import app_commands, Embed
from discord.ext import commands, tasks # tasksを追加
from dotenv import load_dotenv
import aiohttp
try:
    import aiofiles # 非同期ファイルI/O用
except ImportError:
    aiofiles = None # インポート失敗時のフラグ

# --- Windows用イベントループポリシーの設定 ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # logging.FileHandler("bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger('ollama_bot')

# --- 環境変数の読み込み ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL')
try:
    CHAT_CHANNEL_ID = int(os.getenv('CHAT_CHANNEL_ID'))
except (TypeError, ValueError):
    logger.error("環境変数 'CHAT_CHANNEL_ID' が設定されていないか、整数値ではありません。BOTを終了します。")
    sys.exit(1)
try:
    HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', '50'))
except ValueError:
    logger.warning("環境変数 'HISTORY_LIMIT' の値が不正です。デフォルト値の50を使用します。")
    HISTORY_LIMIT = 50
# プロンプトリロード間隔（分）
try:
    PROMPT_RELOAD_INTERVAL_MINUTES = float(os.getenv('PROMPT_RELOAD_INTERVAL', '5.0'))
except ValueError:
    logger.warning("環境変数 'PROMPT_RELOAD_INTERVAL' の値が不正です。デフォルト値の5.0分を使用します。")
    PROMPT_RELOAD_INTERVAL_MINUTES = 5.0
# モデルリスト更新間隔（分） - オートコンプリート用
try:
    MODEL_UPDATE_INTERVAL_MINUTES = float(os.getenv('MODEL_UPDATE_INTERVAL', '15.0'))
except ValueError:
    logger.warning("環境変数 'MODEL_UPDATE_INTERVAL' の値が不正です。デフォルト値の15.0分を使用します。")
    MODEL_UPDATE_INTERVAL_MINUTES = 15.0


# --- BOT設定 ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# --- グローバル変数 & 定数 ---
active_model = DEFAULT_MODEL
system_prompts: dict[str, str | None] = defaultdict(lambda: None)

PROMPT_DIR_NAME = "prompts"
available_prompts: dict[str, str] = {}
SYSTEM_PROMPT_TXT_CONTENT: str | None = None

available_ollama_models: list[str] = [] # モデルリストキャッシュ用

PROMPT_NAME_DEFAULT = "[デフォルト]"
PROMPT_NAME_SYSTEM_TXT = "[System prompt.txt]"

channel_data = defaultdict(lambda: {
    "history": deque(maxlen=HISTORY_LIMIT),
    "params": {"temperature": 0.7},
    "stats": deque(maxlen=50), # 統計情報の最大保持数
    "is_generating": False,
    "stop_generation_requested": False,
})

STREAM_UPDATE_INTERVAL = 1.5
STREAM_UPDATE_CHARS = 75

script_dir = os.path.dirname(os.path.abspath(__file__))
default_prompt_file_path = os.path.join(script_dir, "System prompt.txt")
prompts_dir_path = os.path.join(script_dir, PROMPT_DIR_NAME)

DEFAULT_SYSTEM_PROMPT_TEXT = " "

# --- ヘルパー関数 ---
def get_prompt_name_from_content(prompt_content: str | None) -> str:
    """プロンプトの内容から、対応する表示名を返す"""
    if prompt_content is None:
        return PROMPT_NAME_DEFAULT
    if SYSTEM_PROMPT_TXT_CONTENT is not None and prompt_content == SYSTEM_PROMPT_TXT_CONTENT:
        return PROMPT_NAME_SYSTEM_TXT
    for name, content in available_prompts.items():
        if prompt_content == content:
            return name
    if prompt_content == DEFAULT_SYSTEM_PROMPT_TEXT:
        return PROMPT_NAME_DEFAULT
    return "[カスタム設定]"

async def load_system_prompt_from_file(file_path: str = default_prompt_file_path) -> str | None:
    """指定されたファイルパスからシステムプロンプトを読み込む (非同期)"""
    global SYSTEM_PROMPT_TXT_CONTENT
    if not aiofiles:
        logger.warning("aiofiles がないため、ファイル読み込みをスキップします: %s", os.path.basename(file_path))
        return None

    logger.info(f"システムプロンプトファイル '{os.path.basename(file_path)}' の非同期読み込み試行...")
    content = None
    try:
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            content = await f.read()
            content = content.strip()
            if not content:
                logger.warning(f"システムプロンプトファイル '{os.path.basename(file_path)}' は空です。")
                content = None
            else:
                logger.info(f"システムプロンプトファイル '{os.path.basename(file_path)}' の読み込み成功。")
    except FileNotFoundError:
        logger.warning(f"システムプロンプトファイル '{os.path.basename(file_path)}' が見つかりませんでした。")
        content = None
    except PermissionError:
        logger.error(f"システムプロンプトファイル '{os.path.basename(file_path)}' の読み取り権限がありません。")
        content = None
    except UnicodeDecodeError:
        logger.error(f"システムプロンプトファイル '{os.path.basename(file_path)}' の読み込み中にエンコーディングエラー。UTF-8で保存してください。")
        content = None
    except Exception as e:
        logger.error(f"システムプロンプトファイル '{os.path.basename(file_path)}' の読み込み中に予期せぬエラー: {e}", exc_info=True)
        content = None

    if file_path == default_prompt_file_path:
        SYSTEM_PROMPT_TXT_CONTENT = content

    return content

def _load_prompts_sync(dir_path: str, current_system_prompt_file_basename: str) -> dict[str, str]:
    """指定されたディレクトリから同期的にプロンプトファイルを読み込む"""
    loaded_prompts = {}
    logger.debug(f"_load_prompts_sync: プロンプトディレクトリ '{dir_path}' の同期読み込みを開始...")

    if not os.path.isdir(dir_path):
        logger.warning(f"_load_prompts_sync: プロンプトディレクトリ '{dir_path}' が見つかりません。")
        return {}

    try:
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".txt") and filename != current_system_prompt_file_basename:
                file_path = os.path.join(dir_path, filename)
                prompt_name = os.path.splitext(filename)[0]
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            if prompt_name in [PROMPT_NAME_DEFAULT, PROMPT_NAME_SYSTEM_TXT]:
                                logger.warning(f"  - _sync: プロンプト名 '{prompt_name}' ({filename}) は予約語のためスキップ。")
                                continue
                            loaded_prompts[prompt_name] = content
                            logger.debug(f"  - _sync: プロンプト '{prompt_name}' を読み込み。")
                        else:
                            logger.warning(f"  - _sync: プロンプトファイル '{filename}' は空。スキップ。")
                except Exception as e:
                    logger.error(f"  - _sync: プロンプトファイル '{filename}' 読込エラー: {e}", exc_info=False)
    except Exception as e:
        logger.error(f"_sync: ディレクトリ '{dir_path}' リスト取得エラー: {e}", exc_info=True)

    logger.debug(f"_load_prompts_sync: 同期読み込み完了: {len(loaded_prompts)} 個。")
    return loaded_prompts

async def get_available_models() -> list[str]:
    """Ollama APIから利用可能なモデルの一覧を取得する"""
    url = f"{OLLAMA_API_URL}/api/tags"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    return sorted([model['name'] for model in models])
                else:
                    logger.warning(f"モデル一覧取得APIエラー - ステータス: {response.status}, URL: {url}")
                    return []
    except asyncio.TimeoutError:
        logger.error(f"Ollama API ({url}) への接続がタイムアウトしました (モデル取得時)。")
        return []
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Ollama APIへの接続に失敗しました (モデル取得時): {e}. URL: {url}")
        return []
    except Exception as e:
        logger.error(f"モデル一覧の取得中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return []

async def fetch_channel_history(channel: discord.TextChannel, limit: int = 100):
    """指定されたチャンネルの過去メッセージを取得し、内部履歴に追加する"""
    if not isinstance(channel, discord.TextChannel):
        logger.warning(f"指定されたチャンネルが無効です: {channel}")
        return

    channel_id = channel.id
    logger.info(f"チャンネル '{channel.name}' (ID: {channel_id}) の履歴取得を開始 (最大{limit}件)...")
    try:
        messages_to_add = []
        count = 0
        async for message in channel.history(limit=limit):
            if not message.author.bot or message.author.id == bot.user.id:
                if message.content:
                    messages_to_add.append({
                        "author_name": message.author.display_name,
                        "author_id": message.author.id,
                        "content": message.content,
                        "timestamp": message.created_at.isoformat(),
                        "is_bot": message.author.bot
                    })
                    count += 1

        added_count = 0
        history_deque = channel_data[channel_id]["history"]
        existing_timestamps_contents = { (msg["timestamp"], msg["content"]) for msg in history_deque }

        for msg in reversed(messages_to_add):
            if (msg["timestamp"], msg["content"]) not in existing_timestamps_contents:
                 history_deque.append(msg)
                 existing_timestamps_contents.add((msg["timestamp"], msg["content"]))
                 added_count += 1

        logger.info(f"チャンネル '{channel.name}' から {count} 件のメッセージを調査し、{added_count} 件を履歴に追加しました。現在の履歴数: {len(history_deque)}")

    except discord.Forbidden:
        logger.error(f"チャンネル '{channel.name}' の履歴読み取り権限がありません。")
    except discord.HTTPException as e:
        logger.error(f"チャンネル '{channel.name}' の履歴取得中にDiscord APIエラーが発生しました: {e}")
    except Exception as e:
        logger.error(f"チャンネル '{channel.name}' の履歴取得中に予期せぬエラーが発生しました: {e}", exc_info=True)

def build_chat_context(channel_id: int) -> list[dict]:
    """指定されたチャンネルIDの内部履歴から、Ollama API用のメッセージリストを構築する"""
    history_deque = channel_data[channel_id]["history"]
    messages = []
    for msg in history_deque:
        role = "assistant" if msg["is_bot"] and msg["author_id"] == bot.user.id else "user"
        messages.append({"role": role, "content": msg["content"]})
    return messages

async def generate_response_stream(
    prompt: str,
    channel_id: int,
    message_to_edit: discord.Message,
    model: str = None,
) -> tuple[str | None, dict | None, str | None]:
    """
    Ollama APIにリクエストを送信し、ストリーミングで応答を生成・表示する。
    停止リクエストがあれば中断する。
    """
    current_model = model or active_model
    if not current_model:
        logger.error(f"チャンネル {channel_id}: 応答生成不可 - モデル未設定")
        return None, None, "エラー: 使用するモデルが設定されていません。"

    channel_params = channel_data[channel_id]["params"]
    system_prompt_content = system_prompts.get(current_model) or DEFAULT_SYSTEM_PROMPT_TEXT
    prompt_name = get_prompt_name_from_content(system_prompts.get(current_model))
    using_custom = system_prompts.get(current_model) is not None

    data = {
        "model": current_model,
        "prompt": prompt,
        "system": system_prompt_content,
        "stream": True,
        "options": channel_params,
    }
    url = f"{OLLAMA_API_URL}/api/generate"
    logger.info(f"チャンネル {channel_id}: モデル '{current_model}' (プロンプト: {prompt_name}) に生成リクエスト送信中...")
    response_key = "response"
    done_key = "done"

    full_response = ""
    last_update_time = time.monotonic()
    last_update_len = 0
    performance_metrics = None
    error_message = None
    stopped_by_user = False
    start_time = time.monotonic()

    async def update_footer(status: str):
        if message_to_edit and message_to_edit.embeds:
            try:
                embed = message_to_edit.embeds[0]
                footer_text = f"Model: {current_model} | Prompt: {prompt_name} | {status}"
                embed.set_footer(text=footer_text)
                await message_to_edit.edit(embed=embed)
            except (discord.NotFound, discord.HTTPException):
                pass

    await update_footer("思考中...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"チャンネル {channel_id}: Ollama APIエラー ({response.status}): {error_text}")
                    return None, None, f"Ollama APIエラー ({response.status})。サーバーログを確認してください。"

                async for line in response.content:
                    if channel_data[channel_id]["stop_generation_requested"]:
                        logger.info(f"チャンネル {channel_id}: ユーザーリクエストにより応答生成停止")
                        stopped_by_user = True
                        error_message = "ユーザーにより応答生成が停止されました。"
                        break

                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            chunk_response_content = ""

                            if response_key in chunk and not chunk.get(done_key, False):
                                chunk_response_content = chunk[response_key]

                            if chunk_response_content:
                                full_response += chunk_response_content
                                current_time = time.monotonic()
                                if (current_time - last_update_time > STREAM_UPDATE_INTERVAL or
                                        len(full_response) - last_update_len > STREAM_UPDATE_CHARS):

                                    if message_to_edit and message_to_edit.embeds:
                                        display_response = full_response
                                        # Embed Description 文字数制限 (4096未満)
                                        if len(display_response) > 4000: # より安全マージン
                                            display_response = display_response[:4000] + "..."

                                        embed = message_to_edit.embeds[0]
                                        embed.description = display_response + " ▌"
                                        footer_text = f"Model: {current_model} | Prompt: {prompt_name} | 生成中..."
                                        embed.set_footer(text=footer_text)
                                        try:
                                            await message_to_edit.edit(embed=embed)
                                            last_update_time = current_time
                                            last_update_len = len(full_response)
                                        except discord.NotFound:
                                            logger.warning(f"チャンネル {channel_id}: ストリーミング編集失敗 - メッセージ消失")
                                            message_to_edit = None
                                            error_message = "ストリーミング中に内部エラー (メッセージ消失)。"
                                            break
                                        except discord.HTTPException as e:
                                            logger.warning(f"チャンネル {channel_id}: ストリーミング編集失敗: {e}")

                            if chunk.get(done_key, False):
                                end_time = time.monotonic()
                                total_duration = end_time - start_time
                                metrics_data = {
                                    "total_duration": total_duration,
                                    "load_duration_sec": chunk.get('load_duration', 0) / 1e9,
                                    "prompt_eval_count": chunk.get('prompt_eval_count', 0),
                                    "prompt_eval_duration_sec": chunk.get('prompt_eval_duration', 0) / 1e9,
                                    "eval_count": chunk.get('eval_count', 0),
                                    "eval_duration_sec": chunk.get('eval_duration', 0) / 1e9
                                }
                                eval_duration = metrics_data["eval_duration_sec"]
                                eval_count = metrics_data["eval_count"]
                                if eval_duration > 0 and eval_count > 0:
                                    tps = eval_count / eval_duration
                                    metrics_data["tokens_per_second"] = tps if not math.isnan(tps) else 0.0
                                else:
                                    metrics_data["tokens_per_second"] = 0.0
                                metrics_data["total_tokens"] = eval_count if not math.isnan(eval_count) else 0

                                performance_metrics = metrics_data
                                logger.info(f"チャンネル {channel_id}: 生成完了 ({total_duration:.2f}s). メトリクス: {performance_metrics.get('tokens_per_second', 0):.2f} tok/s, {performance_metrics.get('total_tokens', 0)} tokens.")
                                break

                        except json.JSONDecodeError as e:
                            logger.error(f"チャンネル {channel_id}: JSON解析失敗: {e}. Line: {line.decode('utf-8', errors='ignore')}")
                        except Exception as e:
                             logger.error(f"チャンネル {channel_id}: ストリーミング処理中エラー: {e}", exc_info=True)
                             error_message = "ストリーミング処理中にエラー発生。"
                             break

                if stopped_by_user:
                    performance_metrics = None # 不完全なメトリクスはクリア

    except asyncio.TimeoutError:
        logger.error(f"チャンネル {channel_id}: Ollama APIタイムアウト")
        error_message = "リクエストタイムアウト。Ollamaサーバー確認要。"
    except aiohttp.ClientConnectorError as e:
        logger.error(f"チャンネル {channel_id}: Ollama API接続失敗: {e}")
        error_message = f"Ollama API ({OLLAMA_API_URL}) 接続不可。サーバー確認要。"
    except Exception as e:
        logger.error(f"チャンネル {channel_id}: Ollama APIリクエスト中エラー: {e}", exc_info=True)
        error_message = f"応答生成中エラー: {str(e)}"

    return full_response.strip(), performance_metrics, error_message

# --- 定期実行タスク ---

@tasks.loop(minutes=PROMPT_RELOAD_INTERVAL_MINUTES)
async def reload_prompts_task():
    global available_prompts
    logger.info("プロンプト定期リロード実行...")
    try:
        new_prompts = await bot.loop.run_in_executor(
            None,
            functools.partial(_load_prompts_sync, prompts_dir_path, os.path.basename(default_prompt_file_path))
        )
        if new_prompts != available_prompts:
            added = list(set(new_prompts.keys()) - set(available_prompts.keys()))
            removed = list(set(available_prompts.keys()) - set(new_prompts.keys()))
            updated = [k for k, v in new_prompts.items() if k in available_prompts and available_prompts[k] != v]
            available_prompts = new_prompts
            log_msg = f"プロンプトリスト更新 ({len(available_prompts)}個)。"
            if added: log_msg += f" 追加: {added}"
            if removed: log_msg += f" 削除: {removed}"
            if updated: log_msg += f" 更新: {updated}"
            logger.info(log_msg)
        else:
            logger.info("プロンプト変更なし。")
    except Exception as e:
        logger.error(f"プロンプトリロードタスクエラー: {e}", exc_info=True)

@reload_prompts_task.before_loop
async def before_reload_prompts():
    await bot.wait_until_ready()
    logger.info(f"プロンプトリロードタスク準備完了 ({PROMPT_RELOAD_INTERVAL_MINUTES}分ごと)。")
    # 初回実行は start() 呼び出し後のループに任せる

@tasks.loop(minutes=MODEL_UPDATE_INTERVAL_MINUTES)
async def update_models_task():
    global available_ollama_models
    logger.info("モデルリスト定期更新実行...")
    try:
        models = await get_available_models()
        if models != available_ollama_models:
            available_ollama_models = models
            logger.info(f"モデルリスト更新 ({len(available_ollama_models)}個): {available_ollama_models}")
        else:
            logger.info("モデルリスト変更なし。")
    except Exception as e:
        logger.error(f"モデルリスト更新タスクエラー: {e}", exc_info=True)

@update_models_task.before_loop
async def before_update_models():
    await bot.wait_until_ready()
    logger.info(f"モデルリスト更新タスク準備完了 ({MODEL_UPDATE_INTERVAL_MINUTES}分ごと)。初回更新実行...")
    await update_models_task() # 初回実行

# --- Discord イベントハンドラ ---

@bot.event
async def on_ready():
    """BOT起動時の処理"""
    logger.info(f'{bot.user} (ID: {bot.user.id}) としてログインしました')

    # System prompt.txt 読み込み
    await load_system_prompt_from_file(default_prompt_file_path) # ★ ここで呼び出し ★

    global active_model
    if not active_model:
        logger.info("デフォルトモデル未設定。利用可能なモデルから自動選択試行...")
        initial_models = await get_available_models()
        if initial_models:
            active_model = initial_models[0]
            available_ollama_models[:] = initial_models
            logger.info(f"アクティブモデルを '{active_model}' に設定しました。")
        else:
            logger.error("利用可能なOllamaモデルが見つかりませんでした。`/model` コマンドで手動設定が必要です。")
    elif not available_ollama_models:
         available_ollama_models[:] = await get_available_models()
         logger.info(f"初回モデルリストキャッシュ更新 ({len(available_ollama_models)}個)。")

    if active_model and active_model not in system_prompts:
         system_prompts[active_model] = None

    chat_channel = bot.get_channel(CHAT_CHANNEL_ID)
    if chat_channel and isinstance(chat_channel, discord.TextChannel):
        logger.info(f"チャットチャンネル '{chat_channel.name}' (ID: {CHAT_CHANNEL_ID}) 認識。履歴読み込み開始...")
        await fetch_channel_history(chat_channel, limit=HISTORY_LIMIT * 2)
    else:
        logger.error(f"指定チャットチャンネルID ({CHAT_CHANNEL_ID}) が見つからないか、テキストチャンネルではありません。")

    try:
        synced = await bot.tree.sync()
        logger.info(f'{len(synced)}個のスラッシュコマンド同期完了: {[cmd.name for cmd in synced]}')
    except Exception as e:
        logger.error(f"スラッシュコマンド同期エラー: {e}")

    if not reload_prompts_task.is_running():
        reload_prompts_task.start()
    if not update_models_task.is_running():
        update_models_task.start()

    logger.info("BOTの準備が完了しました。")


@bot.event
async def on_message(message: discord.Message):
    """メッセージ受信時の処理"""
    if message.author == bot.user or message.channel.id != CHAT_CHANNEL_ID or message.content.startswith('/') or message.content.startswith(bot.command_prefix):
        return

    channel_id = message.channel.id

    if channel_data[channel_id]["is_generating"]:
        try:
            # fail_if_not_exists を削除
            await message.reply("⏳ 他の応答を生成中です。しばらくお待ちください。", mention_author=False, delete_after=10)
        except discord.HTTPException:
            pass
        logger.warning(f"チャンネル {channel_id}: 応答生成中に新規メッセージ受信、スキップ: {message.content[:50]}...")
        return

    current_model = active_model
    if not current_model:
        try:
            await message.reply("⚠️ モデル未選択。`/model` コマンドで選択してください。", mention_author=False)
        except discord.HTTPException: pass
        return

    user_message_data = {
        "author_name": message.author.display_name,
        "author_id": message.author.id,
        "content": message.content,
        "timestamp": message.created_at.isoformat(),
        "is_bot": False
    }
    channel_data[channel_id]["history"].append(user_message_data)
    logger.debug(f"チャンネル {channel_id} 履歴追加 (User): {user_message_data['author_name']} - {user_message_data['content'][:50]}...")

    reply_message = None
    try:
        channel_data[channel_id]["is_generating"] = True
        channel_data[channel_id]["stop_generation_requested"] = False

        prompt_name = get_prompt_name_from_content(system_prompts.get(current_model))
        placeholder_embed = Embed(description="思考中... 🤔", color=discord.Color.light_gray())
        placeholder_embed.set_footer(text=f"Model: {current_model} | Prompt: {prompt_name}")
        reply_message = await message.reply(embed=placeholder_embed, mention_author=False)

        final_response_text, metrics, error_msg = await generate_response_stream(
            prompt=message.content,
            channel_id=channel_id,
            message_to_edit=reply_message,
            model=current_model
        )

    except discord.HTTPException as e:
        logger.error(f"チャンネル {channel_id}: メッセージ処理/プレースホルダー送信エラー: {e}")
        error_embed = Embed(title="エラー", description=f"処理中にエラー発生:\n```\n{e}\n```", color=discord.Color.red())
        try:
            if reply_message: await reply_message.edit(embed=error_embed, content=None)
            else: await message.channel.send(embed=error_embed, reference=message, mention_author=False)
        except discord.HTTPException: pass
        # finally は実行される
    except Exception as e:
        logger.error(f"チャンネル {channel_id}: メッセージ処理中予期せぬエラー: {e}", exc_info=True)
        error_embed = Embed(title="予期せぬエラー", description="処理中に予期せぬエラー発生。", color=discord.Color.red())
        try:
            if reply_message: await reply_message.edit(embed=error_embed, content=None)
            else: await message.channel.send(embed=error_embed, reference=message, mention_author=False)
        except discord.HTTPException: pass
        # finally は実行される
    finally:
        channel_data[channel_id]["is_generating"] = False
        channel_data[channel_id]["stop_generation_requested"] = False
        logger.debug(f"チャンネル {channel_id}: is_generating フラグを False にリセット。")

    # finally ブロックの外で最終メッセージ編集を行う
    # try ブロックで return した場合、ここは実行されない
    if reply_message:
        # try ブロックで final_response_text などが未定義になる可能性があるので、
        # エラーが発生した場合は早期リターンするように修正済みのため、ここでは変数が存在すると仮定できる
        try:
            final_embed = reply_message.embeds[0] if reply_message.embeds else Embed()
            prompt_name = get_prompt_name_from_content(system_prompts.get(current_model))

            if error_msg:
                if "ユーザーにより応答生成が停止されました" in error_msg:
                    final_embed.title = "⏹️ 停止"
                    stopped_text = final_response_text if final_response_text else '(応答なし)'
                    if len(stopped_text) > 4000: stopped_text = stopped_text[:4000] + "...(途中省略)"
                    final_embed.description = f"ユーザーリクエストにより応答停止。\n\n**生成途中内容:**\n{stopped_text}"
                    final_embed.color = discord.Color.orange()
                    logger.info(f"チャンネル {channel_id}: ユーザー停止後のメッセージ表示。")
                else:
                    final_embed.title = "⚠️ エラー"
                    final_embed.description = f"応答生成エラー:\n\n{error_msg}"
                    final_embed.color = discord.Color.red()
                final_embed.set_footer(text=f"Model: {current_model} | Prompt: {prompt_name}")

            elif final_response_text:
                final_embed.title = None
                display_final_text = final_response_text
                if len(display_final_text) > 4000: display_final_text = display_final_text[:4000] + "\n...(文字数上限)"
                final_embed.description = display_final_text
                final_embed.color = discord.Color.blue()

                footer_text = f"Model: {current_model} | Prompt: {prompt_name}"
                if metrics:
                    channel_data[channel_id]["stats"].append(metrics)
                    tok_sec = metrics.get("tokens_per_second", 0)
                    total_tokens = metrics.get("total_tokens", 0)
                    duration = metrics.get("total_duration", 0)
                    if tok_sec > 0: footer_text += f" | {tok_sec:.2f} tok/s"
                    if total_tokens > 0: footer_text += f" | {int(total_tokens)} tokens"
                    if duration > 0: footer_text += f" | {duration:.2f}s"
                final_embed.set_footer(text=footer_text)

                bot_message_data = {
                    "author_name": bot.user.display_name,
                    "author_id": bot.user.id,
                    "content": final_response_text,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "is_bot": True
                }
                channel_data[channel_id]["history"].append(bot_message_data)
                logger.debug(f"チャンネル {channel_id} 履歴追加 (Bot): {bot_message_data['author_name']} - {bot_message_data['content'][:50]}...")
            else:
                final_embed.title = "❓ 無応答"
                final_embed.description = "応答生成失敗。入力確認またはモデル変更試行要。"
                final_embed.color = discord.Color.orange()
                final_embed.set_footer(text=f"Model: {current_model} | Prompt: {prompt_name}")
                logger.warning(f"チャンネル {channel_id}: 応答テキストもエラーメッセージもなし。")

            await reply_message.edit(embed=final_embed)

        except discord.NotFound:
             logger.warning(f"チャンネル {channel_id}: 最終メッセージ編集失敗 - メッセージ消失 (ID: {reply_message.id})")
        except discord.HTTPException as e:
            logger.error(f"チャンネル {channel_id}: 最終メッセージ編集失敗: {e}")
            err_code = getattr(e, 'code', 'N/A')
            err_text = str(e)[:100]
            try:
                await message.channel.send(f"エラー: 応答最終表示失敗 (Code: {err_code}) - {err_text}", reference=message, mention_author=False)
            except discord.HTTPException: pass
        except IndexError:
             logger.error(f"チャンネル {channel_id}: 最終メッセージ編集失敗 - Embedなし")
             try: await reply_message.edit(content="エラー: 応答表示準備失敗。", embed=None)
             except discord.HTTPException: pass
        except Exception as e:
            logger.error(f"チャンネル {channel_id}: 最終メッセージ編集中予期せぬエラー: {e}", exc_info=True)
            try: await reply_message.edit(content="エラー: 応答最終表示中エラー発生。", embed=None)
            except discord.HTTPException: pass

# --- スラッシュコマンド ---

# --- オートコンプリート ---
async def model_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = [
        app_commands.Choice(name=model, value=model)
        for model in available_ollama_models if current.lower() in model.lower()
    ]
    return choices[:25]

async def prompt_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = []
    special_choices = {PROMPT_NAME_DEFAULT: None, PROMPT_NAME_SYSTEM_TXT: SYSTEM_PROMPT_TXT_CONTENT}
    for name, content in special_choices.items():
        if name == PROMPT_NAME_SYSTEM_TXT and content is None: continue
        if current.lower() in name.lower(): choices.append(app_commands.Choice(name=name, value=name))
    custom_choices = [
        app_commands.Choice(name=name, value=name)
        for name in sorted(available_prompts.keys()) if current.lower() in name.lower()
    ]
    choices.extend(custom_choices)
    return choices[:25]

# --- コマンド本体 ---

@bot.tree.command(name="stop", description="現在このチャンネルで生成中のAIの応答を停止します。")
async def stop_generation(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    if channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    # defer() は不要（すぐに完了するため）
    if channel_data[channel_id]["is_generating"]:
        if not channel_data[channel_id]["stop_generation_requested"]:
            channel_data[channel_id]["stop_generation_requested"] = True
            logger.info(f"チャンネル {channel_id}: ユーザー {interaction.user} (ID: {interaction.user.id}) により停止リクエスト。")
            await interaction.response.send_message("⏹️ 応答の停止を試みています...", ephemeral=True)
            try:
                await interaction.channel.send(f"⚠️ {interaction.user.mention} が応答生成の停止を試みています。")
            except discord.HTTPException as e:
                 logger.warning(f"チャンネル {channel_id}: 停止試行の公開ログ送信失敗: {e}")
        else:
            await interaction.response.send_message("ℹ️ 既に停止リクエストが送信されています。", ephemeral=True)
    else:
        await interaction.response.send_message("ℹ️ 現在このチャンネルで生成中の応答はありません。", ephemeral=True)


@bot.tree.command(name="model", description="使用するAIモデルとシステムプロンプトを設定します。")
@app_commands.describe(
    model="利用可能なモデル名を選択してください。",
    prompt_name=f"適用するシステムプロンプト ('{PROMPT_DIR_NAME}'内のファイル名、{PROMPT_NAME_DEFAULT}、{PROMPT_NAME_SYSTEM_TXT})"
)
@app_commands.autocomplete(model=model_autocomplete, prompt_name=prompt_autocomplete)
async def select_model(interaction: discord.Interaction, model: str, prompt_name: str = None):
    global active_model
    channel_id = interaction.channel_id
    if channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    # defer() を先に行う
    await interaction.response.defer(ephemeral=True, thinking=False)

    if model not in available_ollama_models:
        model_list_str = "\n- ".join(available_ollama_models) if available_ollama_models else "キャッシュにモデルがありません。"
        await interaction.followup.send(
            f"❌ エラー: モデル '{model}' は利用できません (キャッシュ参照)。\n利用可能なモデル (キャッシュ):\n- {model_list_str}",
            ephemeral=True
        )
        return

    previous_model = active_model
    previous_prompt_content = system_prompts.get(previous_model)
    previous_prompt_name = get_prompt_name_from_content(previous_prompt_content)

    active_model = model
    model_changed = previous_model != active_model

    prompt_actually_changed = False
    selected_prompt_name_for_log = None
    error_occurred = False
    ephemeral_message_lines = []

    if prompt_name:
        new_prompt_content: str | None = None
        valid_prompt_selection = False
        selected_prompt_name_for_log = prompt_name

        if prompt_name == PROMPT_NAME_DEFAULT: new_prompt_content = None; valid_prompt_selection = True
        elif prompt_name == PROMPT_NAME_SYSTEM_TXT:
            if SYSTEM_PROMPT_TXT_CONTENT is not None: new_prompt_content = SYSTEM_PROMPT_TXT_CONTENT; valid_prompt_selection = True
            else: ephemeral_message_lines.append(f"⚠️ `{os.path.basename(default_prompt_file_path)}` 未検出/空。プロンプト '{prompt_name}' 設定不可。"); error_occurred = True
        elif prompt_name in available_prompts: new_prompt_content = available_prompts[prompt_name]; valid_prompt_selection = True
        else: ephemeral_message_lines.append(f"❌ エラー: 不明なプロンプト名 '{prompt_name}'。プロンプト設定スキップ。"); selected_prompt_name_for_log = None; error_occurred = True

        if valid_prompt_selection:
            current_prompt_for_new_model = system_prompts.get(active_model)
            if new_prompt_content != current_prompt_for_new_model:
                system_prompts[active_model] = new_prompt_content
                logger.info(f"チャンネル {channel_id}: モデル '{active_model}' プロンプト設定 -> '{prompt_name}'")
                prompt_actually_changed = True
                ephemeral_message_lines.append(f"📄 システムプロンプトを **{prompt_name}** に設定。")
            else: ephemeral_message_lines.append(f"ℹ️ モデル **{active_model}** プロンプトは既に **{prompt_name}**。")
    else:
        maintained_prompt_content = system_prompts.get(active_model)
        selected_prompt_name_for_log = get_prompt_name_from_content(maintained_prompt_content)
        ephemeral_message_lines.append(f"ℹ️ システムプロンプト **{selected_prompt_name_for_log}** 維持。")
        if active_model not in system_prompts: system_prompts[active_model] = None

    final_ephemeral_message = []
    if model_changed: final_ephemeral_message.append(f"✅ モデル変更 -> **{active_model}**。")
    else: final_ephemeral_message.append(f"ℹ️ モデルは **{active_model}** のまま。")
    final_ephemeral_message.extend(ephemeral_message_lines)

    await interaction.followup.send("\n".join(final_ephemeral_message), ephemeral=True)

    if not error_occurred and (model_changed or prompt_actually_changed):
        log_parts = []
        current_model_display = f"**{active_model}**"
        current_prompt_display = f"**{selected_prompt_name_for_log}**"
        if model_changed and prompt_actually_changed: log_parts.append(f"モデル: **{previous_model}** → {current_model_display}, プロンプト: **{previous_prompt_name}** → {current_prompt_display}")
        elif model_changed: log_parts.append(f"モデル: **{previous_model}** → {current_model_display} (プロンプト: {current_prompt_display})")
        elif prompt_actually_changed: log_parts.append(f"モデル {current_model_display} プロンプト: **{previous_prompt_name}** → {current_prompt_display}")

        if log_parts:
            public_log_message = f"🔧 {interaction.user.mention} 設定変更: {' '.join(log_parts)}"
            try:
                if interaction.channel: await interaction.channel.send(public_log_message)
                else: logger.warning(f"チャンネル {channel_id}: 公開ログ送信失敗 - interaction.channel is None")
            except discord.HTTPException as e: logger.error(f"チャンネル {channel_id}: モデル変更公開ログ送信失敗: {e}")


@bot.tree.command(name="set_prompt", description="現在アクティブなモデルのシステムプロンプトを設定します。")
@app_commands.describe(
    prompt_name=f"適用するシステムプロンプト ('{PROMPT_DIR_NAME}'内のファイル名、{PROMPT_NAME_DEFAULT}、{PROMPT_NAME_SYSTEM_TXT})"
)
@app_commands.autocomplete(prompt_name=prompt_autocomplete)
async def set_prompt(interaction: discord.Interaction, prompt_name: str):
    channel_id = interaction.channel_id
    if channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return
    if not active_model:
        await interaction.response.send_message("⚠️ モデル未選択。`/model` コマンドで選択してください。", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=False)

    previous_prompt_content = system_prompts.get(active_model)
    previous_prompt_name = get_prompt_name_from_content(previous_prompt_content)
    new_prompt_content: str | None = None
    valid_prompt = False
    error_message = None

    if prompt_name == PROMPT_NAME_DEFAULT: new_prompt_content = None; valid_prompt = True
    elif prompt_name == PROMPT_NAME_SYSTEM_TXT:
        if SYSTEM_PROMPT_TXT_CONTENT is not None: new_prompt_content = SYSTEM_PROMPT_TXT_CONTENT; valid_prompt = True
        else: error_message = f"⚠️ `{os.path.basename(default_prompt_file_path)}` 未検出/空。設定不可。"
    elif prompt_name in available_prompts: new_prompt_content = available_prompts[prompt_name]; valid_prompt = True
    else: error_message = f"❌ エラー: 不明なプロンプト名 '{prompt_name}'。"

    if error_message:
        await interaction.followup.send(error_message, ephemeral=True)
        return

    if valid_prompt:
        if new_prompt_content != previous_prompt_content:
            system_prompts[active_model] = new_prompt_content
            logger.info(f"チャンネル {channel_id}: モデル '{active_model}' プロンプト変更 -> '{prompt_name}'")
            await interaction.followup.send(f"✅ モデル **{active_model}** プロンプト設定 -> **{prompt_name}**。", ephemeral=True)

            public_log_message = f"🔧 {interaction.user.mention} がモデル **{active_model}** のプロンプト変更: **{previous_prompt_name}** → **{prompt_name}**"
            try:
                if interaction.channel: await interaction.channel.send(public_log_message)
                else: logger.warning(f"チャンネル {channel_id}: 公開ログ送信失敗 - interaction.channel is None")
            except discord.HTTPException as e: logger.error(f"チャンネル {channel_id}: プロンプト変更公開ログ送信失敗: {e}")
        else:
            await interaction.followup.send(f"ℹ️ モデル **{active_model}** プロンプトは既に **{prompt_name}**。", ephemeral=True)


@bot.tree.command(name="clear_history", description="このチャンネルの会話履歴と応答統計を消去します。")
async def clear_history(interaction: discord.Interaction):
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=False)

    if target_channel_id in channel_data:
        channel_data[target_channel_id]["history"].clear()
        channel_data[target_channel_id]["stats"].clear()
        logger.info(f"チャンネルID {target_channel_id} 会話履歴/統計クリア完了。")
        await interaction.followup.send("✅ このチャンネルの会話履歴と応答統計をクリアしました。", ephemeral=True)
    else:
        logger.warning(f"クリア対象チャンネルID {target_channel_id} データなし。")
        await interaction.followup.send("ℹ️ クリア対象の会話履歴が見つかりませんでした。", ephemeral=True)


@bot.tree.command(name="show_history", description="このチャンネルの直近の会話履歴を表示します。")
@app_commands.describe(count=f"表示する履歴の件数 (デフォルト10, 最大 {HISTORY_LIMIT})")
async def show_history(interaction: discord.Interaction, count: app_commands.Range[int, 1, None] = 10):
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=False)

    history = channel_data[target_channel_id]["history"]
    if not history:
        await interaction.followup.send("表示できる会話履歴がありません。", ephemeral=True)
        return

    actual_count = min(count, HISTORY_LIMIT)
    history_list = list(history)
    start_index = max(0, len(history_list) - actual_count)
    display_history = history_list[start_index:]

    embed = Embed(title=f"直近会話履歴 ({len(display_history)}/{len(history_list)}件)", color=discord.Color.light_gray())
    history_text = ""
    for i, msg in enumerate(display_history):
        prefix = "🤖" if msg["is_bot"] else "👤"
        author_name_safe = discord.utils.escape_markdown(msg['author_name'])
        author_str = f"{prefix} **{'Assistant' if msg['is_bot'] else author_name_safe}**"
        content_short = (msg['content'][:150] + '...') if len(msg['content']) > 150 else msg['content']
        content_safe = discord.utils.escape_markdown(content_short).replace('`', '\\`')
        entry_text = f"`{start_index + i + 1}`. {author_str}:\n{content_safe}\n\n"

        if len(history_text) + len(entry_text) > 4000:
             history_text += "... (表示数上限のため省略)"
             break
        history_text += entry_text

    embed.description = history_text if history_text else "履歴内容空。"
    embed.set_footer(text=f"最大保持数: {HISTORY_LIMIT}件")
    await interaction.followup.send(embed=embed, ephemeral=True)


@bot.tree.command(name="set_param", description="Ollamaの生成パラメータ (temperature, top_k, top_p) を調整します。")
@app_commands.describe(parameter="調整するパラメータ名", value="設定する値 (例: 0.7, 50)")
@app_commands.choices(parameter=[
    app_commands.Choice(name="temperature", value="temperature"),
    app_commands.Choice(name="top_k", value="top_k"),
    app_commands.Choice(name="top_p", value="top_p"),
])
async def set_parameter(interaction: discord.Interaction, parameter: app_commands.Choice[str], value: str):
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=False)

    param_name = parameter.value
    current_params = channel_data[target_channel_id]["params"]
    response_message = ""

    try:
        original_value = current_params.get(param_name)
        new_value = None

        if param_name == "temperature":
            float_value = float(value); new_value = float_value if 0.0 <= float_value <= 2.0 else (_ for _ in ()).throw(ValueError("Temperature は 0.0 ～ 2.0"))
        elif param_name == "top_k":
            int_value = int(value); new_value = int_value if int_value >= 0 else (_ for _ in ()).throw(ValueError("Top K は 0 以上"))
        elif param_name == "top_p":
            float_value = float(value); new_value = float_value if 0.0 <= float_value <= 1.0 else (_ for _ in ()).throw(ValueError("Top P は 0.0 ～ 1.0"))

        if new_value is not None:
            is_changed = not (isinstance(original_value, (int, float)) and isinstance(new_value, (int, float)) and math.isclose(original_value, new_value, rel_tol=1e-9)) and original_value != new_value
            if is_changed:
                 current_params[param_name] = new_value
                 logger.info(f"チャンネル {target_channel_id}: パラメータ '{param_name}' 設定 -> '{new_value}'")
                 response_message = f"✅ パラメータ **{param_name}** 設定 -> **{new_value}**。"
            else: response_message = f"ℹ️ パラメータ **{param_name}** は既に **{new_value}**。"
        else: raise ValueError("内部エラー: 値処理失敗。")

    except ValueError as e:
        logger.warning(f"チャンネル {target_channel_id}: パラメータ設定エラー ({param_name}={value}): {e}")
        response_message = f"⚠️ 設定値エラー: {e}"
    except Exception as e:
        logger.error(f"チャンネル {target_channel_id}: パラメータ設定中エラー: {e}", exc_info=True)
        response_message = "❌ パラメータ設定中エラー発生。"

    await interaction.followup.send(response_message, ephemeral=True)


@bot.tree.command(name="stats", description="現在の設定と直近の応答生成統計を表示します。")
async def show_stats(interaction: discord.Interaction):
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=False)

    stats_deque = channel_data[target_channel_id]["stats"]
    total_count = len(stats_deque)
    stats_max_len = channel_data[target_channel_id]["stats"].maxlen or 50

    embed = Embed(title="📊 BOTステータス & 応答統計", color=discord.Color.green())

    current_model_str = f"**{active_model}**" if active_model else "未設定"
    current_prompt_content = system_prompts.get(active_model)
    current_prompt_name = get_prompt_name_from_content(current_prompt_content)
    current_prompt_str = f"**{current_prompt_name}**"
    current_params = channel_data[target_channel_id]["params"]
    params_str = ", ".join([f"{k}={v}" for k, v in sorted(current_params.items())]) if current_params else "デフォルト"

    embed.add_field(
        name="現在の設定",
        value=f"モデル: {current_model_str}\nプロンプト: {current_prompt_str}\nパラメータ: `{params_str}`",
        inline=False
    )

    if not stats_deque:
        embed.add_field(name=f"応答統計 (直近 0/{stats_max_len} 回)", value="記録なし。", inline=False)
    else:
        total_duration, total_tokens, total_tps, valid_tps_count = 0.0, 0, 0.0, 0
        for stat in stats_deque:
            duration = stat.get("total_duration", 0.0)
            tokens = stat.get("total_tokens", 0)
            tps = stat.get("tokens_per_second", 0.0)
            if duration > 0 and duration < 600: total_duration += duration
            if tokens > 0: total_tokens += tokens
            if tps > 0 and tps < 10000: total_tps += tps; valid_tps_count += 1

        avg_duration = total_duration / total_count if total_count > 0 else 0
        avg_tokens = total_tokens / total_count if total_count > 0 else 0
        avg_tps = total_tps / valid_tps_count if valid_tps_count > 0 else 0

        stats_summary = (
            f"平均応答時間: **{avg_duration:.2f} 秒**\n"
            f"平均生成トークン数: **{avg_tokens:.1f} トークン**\n"
            f"平均TPS: **{avg_tps:.2f} tok/s**"
        )
        embed.add_field(name=f"応答統計 (直近 {total_count}/{stats_max_len} 回)", value=stats_summary, inline=False)

    embed.set_footer(text=f"履歴保持数: {HISTORY_LIMIT} | Ollama API: {OLLAMA_API_URL}")
    await interaction.followup.send(embed=embed, ephemeral=True)

# --- BOT起動 ---
if __name__ == "__main__":
    if not TOKEN: logger.critical("環境変数 'DISCORD_TOKEN' 未設定。"); sys.exit(1)
    if CHAT_CHANNEL_ID is None: logger.critical("環境変数 'CHAT_CHANNEL_ID' 無効。"); sys.exit(1)
    if aiofiles is None: logger.warning("`aiofiles` 未インストール。一部機能制限あり。")

    logger.info("--- Ollama Discord BOT 起動プロセス開始 ---")
    logger.info(f"監視チャンネルID: {CHAT_CHANNEL_ID}")
    logger.info(f"デフォルトモデル: {DEFAULT_MODEL or '未設定'}")
    logger.info(f"履歴保持数: {HISTORY_LIMIT}")
    logger.info(f"Ollama API URL: {OLLAMA_API_URL}")
    logger.info(f"システムプロンプトパス: {default_prompt_file_path}")
    logger.info(f"カスタムプロンプトDir: {prompts_dir_path}")
    logger.info(f"プロンプトリロード間隔: {PROMPT_RELOAD_INTERVAL_MINUTES} 分")
    logger.info(f"モデルリスト更新間隔: {MODEL_UPDATE_INTERVAL_MINUTES} 分")
    logger.info("-------------------------------------------")

    try:
        bot.run(TOKEN, log_handler=None)
    except discord.LoginFailure: logger.critical("Discordログイン失敗。トークン確認要。")
    except discord.PrivilegedIntentsRequired: logger.critical("Message Content Intent 無効。Developer Portal確認要。")
    except ImportError as e: logger.critical(f"ライブラリ不足: {e}")
    except Exception as e: logger.critical(f"BOT起動中致命的エラー: {e}", exc_info=True)

# --- END OF FILE bot.py ---