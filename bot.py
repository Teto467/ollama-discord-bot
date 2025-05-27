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

# Gemini API 関連のインポート
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions # Gemini APIエラー処理用
    from google.generativeai.types import HarmCategory, HarmBlockThreshold # セーフティ設定用
except ImportError:
    genai = None
    google_exceptions = None
    # loggerはまだ定義されていないので、ここではprintする
    # この警告は後ほどloggerが初期化された後にも出す可能性があります
    print("警告: 'google-generativeai' ライブラリが見つかりません。Gemini API機能は利用できません。")


# --- Windows用イベントループポリシーの設定 ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # logging.FileHandler("bot.log", encoding="utf-8") # 必要に応じて有効化
    ]
)
logger = logging.getLogger('discord_llm_bot') # BOT名をdiscord_llm_botに変更

# --- 環境変数の読み込み ---
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # Gemini APIキー
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL') # 例: "ollama:llama3" や "gemini:gemini-1.5-flash-latest"

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
try:
    PROMPT_RELOAD_INTERVAL_MINUTES = float(os.getenv('PROMPT_RELOAD_INTERVAL', '5.0'))
except ValueError:
    logger.warning("環境変数 'PROMPT_RELOAD_INTERVAL' の値が不正です。デフォルト値の5.0分を使用します。")
    PROMPT_RELOAD_INTERVAL_MINUTES = 5.0
try:
    MODEL_UPDATE_INTERVAL_MINUTES = float(os.getenv('MODEL_UPDATE_INTERVAL', '15.0'))
except ValueError:
    logger.warning("環境変数 'MODEL_UPDATE_INTERVAL' の値が不正です。デフォルト値の15.0分を使用します。")
    MODEL_UPDATE_INTERVAL_MINUTES = 15.0

# --- Gemini APIクライアント設定 ---
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini APIクライアント設定完了。")
    except Exception as e:
        logger.error(f"Gemini APIクライアント設定失敗: {e}", exc_info=True)
        genai = None # 設定失敗時はgenaiをNoneにして機能無効化
elif genai and not GEMINI_API_KEY:
    logger.warning("環境変数 'GEMINI_API_KEY' が設定されていません。Gemini API機能は無効化されます。")
    genai = None
elif not genai: # インポート失敗時の再警告 (loggerが利用可能になったため)
    logger.warning("'google-generativeai' ライブラリのインポートに失敗したため、Gemini API機能は利用できません。")


# --- BOT設定 ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents) # コマンドプレフィックスは現状ほぼ使わないが一応残す

# --- グローバル変数 & 定数 ---
active_model: str | None = DEFAULT_MODEL # 現在選択中のモデル (プレフィックス付き: "ollama:model" or "gemini:model")

# 各モデルごとのシステムプロンプトを保持 (None はデフォルトプロンプトを使用)
# キーはプレフィックス付きモデル名 (例: "ollama:llama3", "gemini:gemini-1.5-pro-latest")
system_prompts: dict[str, str | None] = defaultdict(lambda: None)

PROMPT_DIR_NAME = "prompts"
available_prompts: dict[str, str] = {} # prompts ディレクトリ内のカスタムプロンプト

# モデルリストキャッシュ用 (プレフィックス付きモデル名: "ollama:model" or "gemini:model")
available_bot_models: list[str] = []

PROMPT_NAME_DEFAULT = "[デフォルト]"

channel_data = defaultdict(lambda: {
    "history": deque(maxlen=HISTORY_LIMIT),
    "params": {"temperature": 0.7, "top_k": None, "top_p": None}, # top_k, top_p も追加 (Noneは未設定)
    "stats": deque(maxlen=50),
    "is_generating": False,
    "stop_generation_requested": False,
})

STREAM_UPDATE_INTERVAL = 1.5 # ストリーミング時のDiscordメッセージ更新間隔（秒）
STREAM_UPDATE_CHARS = 75    # ストリーミング時のDiscordメッセージ更新文字数間隔

script_dir = os.path.dirname(os.path.abspath(__file__))
prompts_dir_path = os.path.join(script_dir, PROMPT_DIR_NAME)

DEFAULT_SYSTEM_PROMPT_TEXT = "" # API側のデフォルトを使う意図

# Gemini用のセーフティセッティング (必要に応じて調整)
GEMINI_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


# --- ヘルパー関数 ---
def get_model_type_and_name(model_identifier: str | None) -> tuple[str | None, str | None]:
    """
    モデル識別子 (例: "ollama:llama3", "gemini:gemini-pro") からタイプと実際のモデル名を分離する。
    プレフィックスがない場合は、タイプをNone、名前をそのまま返す（後方互換性またはエラーケース）。
    """
    if not model_identifier:
        return None, None
    if ":" in model_identifier:
        parts = model_identifier.split(":", 1)
        if len(parts) == 2:
            return parts[0].lower(), parts[1]
    return None, model_identifier # プレフィックスなし (Ollama単独時代の名残やエラーの可能性)

def get_prompt_name_from_content(prompt_content: str | None) -> str:
    if prompt_content is None or prompt_content == DEFAULT_SYSTEM_PROMPT_TEXT:
        return PROMPT_NAME_DEFAULT
    for name, content in available_prompts.items():
        if prompt_content == content:
            return name
    return "[カスタム設定]"

def _load_prompts_sync(dir_path: str) -> dict[str, str]:
    loaded_prompts = {}
    logger.debug(f"_load_prompts_sync: プロンプトディレクトリ '{dir_path}' の同期読み込みを開始...")
    if not os.path.isdir(dir_path):
        logger.warning(f"_load_prompts_sync: プロンプトディレクトリ '{dir_path}' が見つかりません。")
        return {}
    try:
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                prompt_name = os.path.splitext(filename)[0]
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            if prompt_name == PROMPT_NAME_DEFAULT: # 予約語チェック
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

async def fetch_and_update_available_models() -> list[str]:
    """OllamaとGemini APIから利用可能なモデルの一覧を取得し、キャッシュを更新する"""
    global available_bot_models
    new_model_list = []

    # 1. Ollamaモデルの取得
    ollama_url = f"{OLLAMA_API_URL}/api/tags"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ollama_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    for model_info in models:
                        model_name = model_info.get('name')
                        if model_name:
                            new_model_list.append(f"ollama:{model_name}")
                    logger.info(f"Ollamaから {len(models)} 個のモデルを取得。")
                else:
                    logger.warning(f"Ollamaモデル一覧取得APIエラー - ステータス: {response.status}, URL: {ollama_url}")
    except asyncio.TimeoutError:
        logger.error(f"Ollama API ({ollama_url}) への接続がタイムアウトしました (モデル取得時)。")
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Ollama APIへの接続に失敗しました (モデル取得時): {e}. URL: {ollama_url}")
    except Exception as e:
        logger.error(f"Ollamaモデル一覧の取得中に予期せぬエラー: {e}", exc_info=True)

    # 2. Geminiモデルの取得
    if genai: # Gemini APIが利用可能な場合のみ
        try:
            gemini_models_found = 0
            for model_info in genai.list_models():
                # 'generateContent' (テキスト生成) をサポートし、かつ名前に 'embedding' を含まないモデルをリストアップ
                if 'generateContent' in model_info.supported_generation_methods and 'embedding' not in model_info.name:
                    # モデル名は通常 "models/gemini-1.5-pro-latest" のような形式なので、"gemini-" より後を取得
                    name_part = model_info.name.split('/')[-1]
                    new_model_list.append(f"gemini:{name_part}")
                    gemini_models_found +=1
            logger.info(f"Gemini APIから {gemini_models_found} 個の生成モデルを取得。")
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Gemini APIからのモデル一覧取得に失敗: {e}")
        except Exception as e:
            logger.error(f"Geminiモデル一覧の取得中に予期せぬエラー: {e}", exc_info=True)
    else:
        logger.info("Gemini APIが無効なため、Geminiモデルの取得はスキップされました。")

    sorted_models = sorted(list(set(new_model_list))) # 重複除去とソート

    if sorted_models != available_bot_models:
        logger.info(f"モデルリスト更新 ({len(sorted_models)}個): {sorted_models}")
        available_bot_models = sorted_models
    else:
        logger.info(f"モデルリスト変更なし。現在のキャッシュ: {len(available_bot_models)}個。")
    return available_bot_models


async def fetch_channel_history(channel: discord.TextChannel, limit: int = 100):
    if not isinstance(channel, discord.TextChannel):
        logger.warning(f"指定されたチャンネルが無効です: {channel}")
        return

    channel_id = channel.id
    logger.info(f"チャンネル '{channel.name}' (ID: {channel_id}) の履歴取得を開始 (最大{limit}件)...")
    try:
        messages_to_add = []
        count = 0
        async for message in channel.history(limit=limit):
            if not message.author.bot or message.author.id == bot.user.id: # 自分自身のメッセージは含める
                if message.content: # コンテンツがあるメッセージのみ
                    messages_to_add.append({
                        "author_name": message.author.display_name,
                        "author_id": message.author.id,
                        "content": message.content,
                        "timestamp": message.created_at.isoformat(),
                        "is_bot": message.author.bot # BOT自身の発言かどうかのフラグ
                    })
                    count += 1
        
        added_count = 0
        history_deque = channel_data[channel_id]["history"]
        # 既存のメッセージを識別するためのセット (timestampとcontentのタプル)
        existing_timestamps_contents = { (msg["timestamp"], msg["content"]) for msg in history_deque }

        for msg in reversed(messages_to_add): # 新しいものから順に追加するために逆順で処理
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


def build_ollama_chat_context(channel_id: int) -> list[dict]:
    """Ollama API用のメッセージリストを構築する"""
    history_deque = channel_data[channel_id]["history"]
    messages = []
    for msg in history_deque:
        # BOT自身の発言は 'assistant', それ以外は 'user'
        role = "assistant" if msg["is_bot"] and msg["author_id"] == bot.user.id else "user"
        messages.append({"role": role, "content": msg["content"]})
    return messages

def build_gemini_chat_history(channel_id: int) -> list[dict]:
    """Gemini API用のチャット履歴リストを構築する"""
    history_deque = channel_data[channel_id]["history"]
    gemini_history = []
    for msg in history_deque:
        role = "model" if msg["is_bot"] and msg["author_id"] == bot.user.id else "user"
        # Geminiは'parts'の中にテキストをリストとして持つ
        gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
    return gemini_history


async def generate_response_stream(
    user_prompt: str, # "prompt" から "user_prompt" に変更 (Geminiのプロンプトとの混同を避ける)
    channel_id: int,
    message_to_edit: discord.Message,
    model_identifier: str, # "model" から "model_identifier" に変更 (プレフィックス付きモデル名)
) -> tuple[str | None, dict | None, str | None]: # (full_response, performance_metrics, error_message)

    model_type, actual_model_name = get_model_type_and_name(model_identifier)

    if not model_type or not actual_model_name:
        logger.error(f"チャンネル {channel_id}: 応答生成不可 - モデル識別子が無効 ({model_identifier})")
        return None, None, f"エラー: 使用するモデル ({model_identifier}) の指定が正しくありません。"

    channel_params = channel_data[channel_id]["params"]
    system_prompt_content = system_prompts.get(model_identifier, None) or DEFAULT_SYSTEM_PROMPT_TEXT
    current_prompt_display_name = get_prompt_name_from_content(system_prompts.get(model_identifier))

    full_response = ""
    last_update_time = time.monotonic()
    last_update_len = 0
    performance_metrics = None
    error_message = None
    stopped_by_user = False
    start_time = time.monotonic() # 生成開始時間

    async def update_footer(status: str):
        if message_to_edit and message_to_edit.embeds:
            try:
                embed = message_to_edit.embeds[0]
                footer_text = f"Model: {actual_model_name} ({model_type}) | Prompt: {current_prompt_display_name} | {status}"
                embed.set_footer(text=footer_text)
                await message_to_edit.edit(embed=embed)
            except (discord.NotFound, discord.HTTPException):
                pass # メッセージが消えたり編集できない場合は無視

    await update_footer("思考中...")

    if model_type == "ollama":
        # --- Ollama API呼び出し ---
        # /api/chat を使うので、最新のユーザープロンプトも含めてmessagesに含める
        ollama_messages_context = build_ollama_chat_context(channel_id) 
        # ollama_messages_contextの最後の要素が最新のユーザープロンプトのはずなので、
        # generate_response_streamに渡された user_prompt と一致するか確認してもよい

        data = {
            "model": actual_model_name,
            "messages": ollama_messages_context, # ユーザープロンプトを含む全履歴
            "system": system_prompt_content,
            "stream": True,
            "options": { 
                "temperature": channel_params.get("temperature"),
                "top_k": channel_params.get("top_k"),
                "top_p": channel_params.get("top_p"),
            }
        }
        data["options"] = {k: v for k, v in data["options"].items() if v is not None}
        if not data["system"]: del data["system"] # 空なら送らない

        api_url = f"{OLLAMA_API_URL}/api/chat" 
        logger.info(f"チャンネル {channel_id}: Ollamaモデル '{actual_model_name}' (プロンプト: {current_prompt_display_name}) に生成リクエスト送信中...")
        response_key_ollama = "message" 
        content_key_ollama = "content"
        done_key_ollama = "done"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=data, timeout=aiohttp.ClientTimeout(total=600)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"チャンネル {channel_id}: Ollama APIエラー ({response.status}): {error_text}")
                        return None, None, f"Ollama APIエラー ({response.status})。サーバーログを確認してください。"

                    async for line in response.content:
                        if channel_data[channel_id]["stop_generation_requested"]:
                            logger.info(f"チャンネル {channel_id}: ユーザーリクエストによりOllama応答生成停止")
                            stopped_by_user = True
                            error_message = "ユーザーにより応答生成が停止されました。"
                            break
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                chunk_response_content = ""

                                if response_key_ollama in chunk and isinstance(chunk[response_key_ollama], dict) and \
                                   content_key_ollama in chunk[response_key_ollama] and \
                                   not chunk.get(done_key_ollama, False): # API仕様上done=falseでもcontentは来る
                                    chunk_response_content = chunk[response_key_ollama][content_key_ollama]

                                if chunk_response_content:
                                    full_response += chunk_response_content
                                    current_time = time.monotonic()
                                    if (current_time - last_update_time > STREAM_UPDATE_INTERVAL or
                                            len(full_response) - last_update_len > STREAM_UPDATE_CHARS):
                                        if message_to_edit and message_to_edit.embeds:
                                            display_response = full_response
                                            if len(display_response) > 4000:
                                                display_response = display_response[:4000] + "..."
                                            embed = message_to_edit.embeds[0]
                                            embed.description = display_response + " ▌"
                                            await update_footer("生成中...")
                                            try:
                                                await message_to_edit.edit(embed=embed)
                                                last_update_time = current_time
                                                last_update_len = len(full_response)
                                            except discord.NotFound:
                                                logger.warning(f"チャンネル {channel_id}: Ollamaストリーミング編集失敗 - メッセージ消失")
                                                message_to_edit = None; error_message = "ストリーミング中に内部エラー (メッセージ消失)。"; break
                                            except discord.HTTPException as e:
                                                logger.warning(f"チャンネル {channel_id}: Ollamaストリーミング編集失敗: {e}")
                                
                                # /api/chat のストリームの最後は done: true のレスポンスでメトリクスを含む
                                if chunk.get(done_key_ollama, False): 
                                    end_time = time.monotonic()
                                    total_duration = end_time - start_time
                                    metrics_data = {
                                        "total_duration": total_duration, # これはクライアント側計測の総時間
                                        "api_total_duration_sec": chunk.get('total_duration', 0) / 1e9 if chunk.get('total_duration') else 0, # APIが返す総時間
                                        "load_duration_sec": chunk.get('load_duration', 0) / 1e9 if chunk.get('load_duration') else 0,
                                        "prompt_eval_count": chunk.get('prompt_eval_count', 0),
                                        "prompt_eval_duration_sec": chunk.get('prompt_eval_duration', 0) / 1e9 if chunk.get('prompt_eval_duration') else 0,
                                        "eval_count": chunk.get('eval_count', 0), # これが生成されたトークン数に相当
                                        "eval_duration_sec": chunk.get('eval_duration', 0) / 1e9 if chunk.get('eval_duration') else 0 # 生成時間
                                    }
                                    eval_duration = metrics_data["eval_duration_sec"]
                                    eval_count = metrics_data["eval_count"]
                                    if eval_duration > 0 and eval_count > 0:
                                        tps = eval_count / eval_duration
                                        metrics_data["tokens_per_second"] = tps if not math.isnan(tps) else 0.0
                                    else: # eval_durationが0の場合でも、クライアント側総時間で概算TPSを計算
                                        if total_duration > 0 and eval_count > 0 :
                                            metrics_data["tokens_per_second"] = eval_count / total_duration
                                        else:
                                            metrics_data["tokens_per_second"] = 0.0

                                    metrics_data["total_tokens"] = eval_count if not math.isnan(eval_count) else 0 # 生成トークン数
                                    performance_metrics = metrics_data
                                    logger.info(f"チャンネル {channel_id}: Ollama生成完了 ({total_duration:.2f}s client / {metrics_data['api_total_duration_sec']:.2f}s api). メトリクス: {performance_metrics.get('tokens_per_second', 0):.2f} tok/s, {performance_metrics.get('total_tokens', 0)} tokens.")
                                    break
                            except json.JSONDecodeError as e:
                                logger.error(f"チャンネル {channel_id}: Ollama JSON解析失敗: {e}. Line: {line.decode('utf-8', errors='ignore')}")
                            except Exception as e:
                                logger.error(f"チャンネル {channel_id}: Ollamaストリーミング処理中エラー: {e}", exc_info=True)
                                error_message = "Ollamaストリーミング処理中にエラー発生。"; break
                    if stopped_by_user: performance_metrics = None
        except asyncio.TimeoutError:
            logger.error(f"チャンネル {channel_id}: Ollama APIタイムアウト")
            error_message = "Ollama API リクエストタイムアウト。Ollamaサーバー確認要。"
        except aiohttp.ClientConnectorError as e:
            logger.error(f"チャンネル {channel_id}: Ollama API接続失敗: {e}")
            error_message = f"Ollama API ({OLLAMA_API_URL}) 接続不可。サーバー確認要。"
        except Exception as e:
            logger.error(f"チャンネル {channel_id}: Ollama APIリクエスト中エラー: {e}", exc_info=True)
            error_message = f"Ollama応答生成中エラー: {str(e)}"

    elif model_type == "gemini" and genai:
        # --- Gemini API呼び出し ---
        logger.info(f"チャンネル {channel_id}: Geminiモデル '{actual_model_name}' (プロンプト: {current_prompt_display_name}) に生成リクエスト送信中...")
        try:
            gemini_model_obj = genai.GenerativeModel(
                model_name=actual_model_name, 
                system_instruction=system_prompt_content if system_prompt_content else None,
                safety_settings=GEMINI_SAFETY_SETTINGS 
            )
            
            # Geminiのチャット履歴は、ユーザーの最新プロンプトを含まない形で構築
            chat_history_for_gemini = build_gemini_chat_history(channel_id)
            # 最後のメッセージが現在のユーザープロンプトなので、それを取り除くか、
            # もしくはbuild_gemini_chat_historyが最新のユーザープロンプトを含まないように修正する。
            # 現状のbuild_gemini_chat_historyは全履歴を作るので、最後の要素が現在のユーザー入力。
            # generate_contentのcontentsには、[履歴(ユーザー入力なし), 最新のユーザー入力]の形か、
            # [全履歴(ユーザー入力あり)]のどちらかで渡す。SDKは後者を推奨。
            # そのため、build_gemini_chat_historyで作成した全履歴をそのまま渡す。
            # ただし、user_promptは別で渡さない。
            
            contents_for_gemini = chat_history_for_gemini # build_gemini_chat_historyが最新のユーザー入力を含む

            generation_config_params = {
                "temperature": channel_params.get("temperature"),
                "top_k": channel_params.get("top_k"),
                "top_p": channel_params.get("top_p"),
            }
            generation_config_params = {k: v for k, v in generation_config_params.items() if v is not None}
            gemini_generation_config = genai.types.GenerationConfig(**generation_config_params) if generation_config_params else None

            # SDKの generate_content は、contents に会話履歴全体 (最新のユーザープロンプトも含む) を渡す
            response_stream = await bot.loop.run_in_executor( 
                None,
                functools.partial(
                    gemini_model_obj.generate_content,
                    contents=contents_for_gemini, # ここに最新の user_prompt を含む全履歴
                    stream=True,
                    generation_config=gemini_generation_config
                )
            )
            
            usage_metadata_final = None 

            for chunk in response_stream:
                if channel_data[channel_id]["stop_generation_requested"]:
                    logger.info(f"チャンネル {channel_id}: ユーザーリクエストによりGemini応答生成停止")
                    stopped_by_user = True; error_message = "ユーザーにより応答生成が停止されました。"; break
                
                # chunk.text が存在し、内容がある場合のみ処理 (空のチャンクが来ることがある)
                if hasattr(chunk, 'text') and chunk.text:
                    full_response += chunk.text
                    current_time = time.monotonic()
                    if (current_time - last_update_time > STREAM_UPDATE_INTERVAL or
                            len(full_response) - last_update_len > STREAM_UPDATE_CHARS):
                        if message_to_edit and message_to_edit.embeds:
                            display_response = full_response
                            if len(display_response) > 4000: display_response = display_response[:4000] + "..."
                            embed = message_to_edit.embeds[0]
                            embed.description = display_response + " ▌"
                            await update_footer("生成中...")
                            try:
                                await message_to_edit.edit(embed=embed)
                                last_update_time = current_time; last_update_len = len(full_response)
                            except discord.NotFound:
                                logger.warning(f"チャンネル {channel_id}: Geminiストリーミング編集失敗 - メッセージ消失")
                                message_to_edit = None; error_message = "ストリーミング中に内部エラー (メッセージ消失)。"; break
                            except discord.HTTPException as e:
                                logger.warning(f"チャンネル {channel_id}: Geminiストリーミング編集失敗: {e}")
                
                # ストリームの最後にusage_metadataが含まれるか確認
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage_metadata_final = chunk.usage_metadata # 最後のチャンクのメタデータを保持


            if not stopped_by_user: 
                end_time = time.monotonic()
                total_duration = end_time - start_time
                
                # ストリーム完了後、メインのresponseオブジェクトからusage_metadataを取得
                # response_streamがイテレータの場合、通常はイテレーション完了後に情報を持つ
                if not usage_metadata_final and hasattr(response_stream, 'usage_metadata'):
                    usage_metadata_final = response_stream.usage_metadata
                
                # もしresolveメソッドがあれば試す (非同期SDKのパターン)
                if not usage_metadata_final and hasattr(response_stream, 'resolve'):
                    try:
                        final_resolved_response = await bot.loop.run_in_executor(None, response_stream.resolve)
                        if hasattr(final_resolved_response, 'usage_metadata'):
                            usage_metadata_final = final_resolved_response.usage_metadata
                    except Exception as e_resolve:
                        logger.warning(f"Gemini ストリーム解決中のエラー: {e_resolve}")


                prompt_tokens = 0
                candidate_tokens = 0 # 生成されたトークン (Geminiではcandidates_token_count)

                if usage_metadata_final:
                    prompt_tokens = getattr(usage_metadata_final, 'prompt_token_count', 0)
                    candidate_tokens = getattr(usage_metadata_final, 'candidates_token_count', 0) 
                
                metrics_data = {
                    "total_duration": total_duration, # クライアント側計測
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": candidate_tokens, 
                    "total_tokens": prompt_tokens + candidate_tokens, # Geminiはprompt+generatedをtotalとしない場合があるので自前で計算
                    "tokens_per_second": 0.0
                }
                if total_duration > 0 and candidate_tokens > 0:
                    tps = candidate_tokens / total_duration
                    metrics_data["tokens_per_second"] = tps if not math.isnan(tps) else 0.0
                
                performance_metrics = metrics_data
                logger.info(f"チャンネル {channel_id}: Gemini生成完了 ({total_duration:.2f}s). メトリクス: {performance_metrics.get('tokens_per_second',0):.2f} tok/s, {performance_metrics.get('generated_tokens',0)} gen tokens, {performance_metrics.get('prompt_tokens',0)} prompt tokens.")

        except google_exceptions.DeadlineExceeded as e:
            logger.error(f"チャンネル {channel_id}: Gemini APIタイムアウト: {e}")
            error_message = "Gemini API リクエストタイムアウト。"
        except google_exceptions.ResourceExhausted as e: 
            logger.error(f"チャンネル {channel_id}: Gemini APIリソース上限超過: {e}")
            error_message = f"Gemini API リソース上限超過。レート制限の可能性があります。" # 詳細メッセージは省略
        except google_exceptions.InvalidArgument as e: 
            logger.error(f"チャンネル {channel_id}: Gemini API不正な引数: {e}")
            error_message = f"Gemini API へのリクエスト引数が不正です (プロンプトが長すぎる、形式が違う等)。"
        except google_exceptions.FailedPrecondition as e: # APIキーが無効など
            logger.error(f"チャンネル {channel_id}: Gemini API事前条件エラー: {e}")
            error_message = f"Gemini APIの事前条件エラー（APIキーが無効、プロジェクト設定不備など）。"
        except google_exceptions.GoogleAPIError as e: # その他のGoogle APIエラー
            logger.error(f"チャンネル {channel_id}: Gemini APIエラー: {e}")
            error_message = f"Gemini APIでエラーが発生しました。"
        except Exception as e:
            logger.error(f"チャンネル {channel_id}: Gemini APIリクエスト中予期せぬエラー: {e}", exc_info=True)
            error_message = f"Gemini応答生成中エラー: {str(e)}"

        if stopped_by_user: performance_metrics = None

    else: 
        logger.error(f"チャンネル {channel_id}: 未知のモデルタイプ '{model_type}' またはGemini APIが無効です。")
        error_message = f"エラー: 未知のモデルタイプ '{model_type}'、または該当APIが設定されていません。"

    return full_response.strip(), performance_metrics, error_message


# --- 定期実行タスク ---
@tasks.loop(minutes=PROMPT_RELOAD_INTERVAL_MINUTES)
async def reload_prompts_task():
    global available_prompts
    logger.info("プロンプト定期リロード実行...")
    try:
        new_prompts = await bot.loop.run_in_executor(
            None,
            functools.partial(_load_prompts_sync, prompts_dir_path)
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
    # on_ready で初回実行するのでここでは不要
    # await reload_prompts_task() 

@tasks.loop(minutes=MODEL_UPDATE_INTERVAL_MINUTES)
async def update_models_task():
    logger.info("モデルリスト定期更新実行...")
    try:
        await fetch_and_update_available_models() # 新しい関数でモデルリストを更新
    except Exception as e:
        logger.error(f"モデルリスト更新タスクエラー: {e}", exc_info=True)

@update_models_task.before_loop
async def before_update_models():
    await bot.wait_until_ready()
    logger.info(f"モデルリスト更新タスク準備完了 ({MODEL_UPDATE_INTERVAL_MINUTES}分ごと)。")
    # on_ready で初回実行するのでここでは不要
    # await update_models_task() 


# --- Discord イベントハンドラ ---
@bot.event
async def on_ready():
    global active_model, available_bot_models, available_prompts

    logger.info(f'{bot.user} (ID: {bot.user.id}) としてログインしました')

    if not TOKEN: logger.critical("環境変数 'DISCORD_TOKEN' 未設定。"); sys.exit(1)
    if CHAT_CHANNEL_ID is None: logger.critical("環境変数 'CHAT_CHANNEL_ID' 無効。"); sys.exit(1)
    if genai is None and GEMINI_API_KEY : 
        logger.warning("Gemini APIキーは設定されていますが、'google-generativeai'ライブラリのインポートに失敗したため、Gemini機能は利用できません。")
    elif genai is None and not GEMINI_API_KEY: 
         logger.info("Gemini APIキーが設定されていない、またはライブラリがありません。Gemini機能は無効です。")
    if HISTORY_LIMIT < 1:
        logger.critical(f"環境変数 'HISTORY_LIMIT' の値 ({HISTORY_LIMIT}) が不正です。1以上の値を設定してください。BOTの動作に深刻な支障が出ます。")
    # 初回モデルリスト取得
    logger.info("初回モデルリスト取得開始...")
    await fetch_and_update_available_models() 

    # 初回カスタムプロンプト読み込み
    try:
        available_prompts = await bot.loop.run_in_executor(
            None,
            functools.partial(_load_prompts_sync, prompts_dir_path)
        )
        logger.info(f"初回カスタムプロンプト読み込み完了 ({len(available_prompts)}個): {list(available_prompts.keys())}")
    except Exception as e:
        logger.error(f"初回カスタムプロンプト読み込みエラー: {e}", exc_info=True)


    # デフォルトモデルの決定とアクティブモデルの設定
    if not active_model: 
        logger.info("デフォルトモデル(.env)未設定。利用可能なモデルから自動選択試行...")
        if available_bot_models:
            # 優先順位: 1. Ollamaモデル, 2. Gemini 2.5系, 3. その他Gemini
            ollama_models_on_ready = [m for m in available_bot_models if m.startswith("ollama:")]
            gemini_2_5_on_ready = [m for m in available_bot_models if m.startswith("gemini:") and "2.5" in m]
            
            if ollama_models_on_ready: active_model = ollama_models_on_ready[0]
            elif gemini_2_5_on_ready: active_model = gemini_2_5_on_ready[0]
            else: active_model = available_bot_models[0] # 上記がなければリストの先頭
            logger.info(f"アクティブモデルを '{active_model}' に設定しました。")
        else:
            logger.error("利用可能なモデルが見つかりませんでした。`/model` コマンドで手動設定が必要です。")
    elif active_model not in available_bot_models: 
        logger.warning(f"環境変数で指定されたデフォルトモデル '{active_model}' は利用可能リストにありません。")
        if available_bot_models:
            logger.info(f"代わりにリストの先頭モデル '{available_bot_models[0]}' を使用します。")
            active_model = available_bot_models[0]
        else:
            logger.error("利用可能なモデルがリストにないため、アクティブモデルを設定できません。")
            active_model = None 
    else: # DEFAULT_MODELが設定され、かつリストにも存在する場合
        logger.info(f"環境変数からデフォルトモデル '{active_model}' をアクティブモデルに設定しました。")


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

    # タスクの開始は on_ready で行う
    if not reload_prompts_task.is_running(): 
        await before_reload_prompts() # before_loop を呼び出してから開始
        reload_prompts_task.start()
    if not update_models_task.is_running(): 
        await before_update_models() # before_loop を呼び出してから開始
        update_models_task.start()

    logger.info("BOTの準備が完了しました。")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.channel.id != CHAT_CHANNEL_ID or \
       message.content.startswith('/') or message.content.startswith(bot.command_prefix or ' unlikely_prefix '):
        return

    channel_id = message.channel.id

    if channel_data[channel_id]["is_generating"]:
        try:
            await message.reply("⏳ 他の応答を生成中です。しばらくお待ちください。", mention_author=False, delete_after=10)
        except discord.HTTPException: pass
        logger.warning(f"チャンネル {channel_id}: 応答生成中に新規メッセージ受信、スキップ: {message.content[:50]}...")
        return

    current_active_model = active_model 
    if not current_active_model:
        try:
            await message.reply("⚠️ モデル未選択。`/model` コマンドで選択してください。", mention_author=False)
        except discord.HTTPException: pass
        return

    model_type, _ = get_model_type_and_name(current_active_model)
    if model_type == "gemini" and not genai:
        try:
            await message.reply("⚠️ 現在選択中のモデルはGeminiモデルですが、Gemini APIが設定されていません。管理者に連絡するか、別のモデルを選択してください。", mention_author=False)
        except discord.HTTPException: pass
        logger.error(f"チャンネル {channel_id}: Geminiモデル '{current_active_model}' が選択されていますが、Gemini APIは無効です。")
        return


    user_message_data = {
        "author_name": message.author.display_name, "author_id": message.author.id,
        "content": message.content, "timestamp": message.created_at.isoformat(), "is_bot": False
    }
    channel_data[channel_id]["history"].append(user_message_data)
    
    reply_message = None
    final_response_text = None
    metrics = None
    error_msg_from_generation = None 

    try:
        channel_data[channel_id]["is_generating"] = True
        channel_data[channel_id]["stop_generation_requested"] = False

        model_type_disp, model_name_disp = get_model_type_and_name(current_active_model)
        prompt_name_disp = get_prompt_name_from_content(system_prompts.get(current_active_model))
        
        placeholder_embed = Embed(description="思考中... 🤔", color=discord.Color.light_gray())
        placeholder_embed.set_footer(text=f"Model: {model_name_disp} ({model_type_disp}) | Prompt: {prompt_name_disp}")
        reply_message = await message.reply(embed=placeholder_embed, mention_author=False)

        final_response_text, metrics, error_msg_from_generation = await generate_response_stream(
            user_prompt=message.content, # generate_response_stream側で履歴に含めるか判断
            channel_id=channel_id,
            message_to_edit=reply_message,
            model_identifier=current_active_model
        )

    except discord.HTTPException as e:
        logger.error(f"チャンネル {channel_id}: メッセージ処理/プレースホルダー送信エラー: {e}")
        error_msg_from_generation = f"処理中にDiscord APIエラー発生:\n```\n{e}\n```"
    except Exception as e:
        logger.error(f"チャンネル {channel_id}: メッセージ処理中予期せぬエラー: {e}", exc_info=True)
        error_msg_from_generation = "処理中に予期せぬエラーが発生しました。"
    finally:
        channel_data[channel_id]["is_generating"] = False
        channel_data[channel_id]["stop_generation_requested"] = False
        
    if reply_message:
        try:
            final_embed = reply_message.embeds[0] if reply_message.embeds else Embed()
            model_type_final, model_name_final = get_model_type_and_name(current_active_model)
            prompt_name_final = get_prompt_name_from_content(system_prompts.get(current_active_model))

            if error_msg_from_generation:
                if "ユーザーにより応答生成が停止されました" in error_msg_from_generation:
                    final_embed.title = "⏹️ 停止"
                    stopped_text = final_response_text if final_response_text else '(応答なし)'
                    if len(stopped_text) > 3900: stopped_text = stopped_text[:3900] + "...(途中省略)" 
                    final_embed.description = f"ユーザーリクエストにより応答停止。\n\n**生成途中内容:**\n{stopped_text}"
                    final_embed.color = discord.Color.orange()
                else:
                    final_embed.title = "⚠️ エラー"
                    final_embed.description = f"応答生成エラー:\n\n{error_msg_from_generation}"
                    final_embed.color = discord.Color.red()
                final_embed.set_footer(text=f"Model: {model_name_final} ({model_type_final}) | Prompt: {prompt_name_final}")

            elif final_response_text is not None:
                final_embed.title = None
                display_final_text = final_response_text
                if len(display_final_text) > 4000: display_final_text = display_final_text[:4000] + "\n...(文字数上限)"
                final_embed.description = display_final_text if display_final_text else "(空の応答)"
                final_embed.color = discord.Color.blue()

                footer_text_parts = [f"Model: {model_name_final} ({model_type_final})", f"Prompt: {prompt_name_final}"]
                if metrics:
                    channel_data[channel_id]["stats"].append(metrics) 
                    duration = metrics.get("total_duration", 0)
                    if duration > 0: footer_text_parts.append(f"{duration:.2f}s")

                    if model_type_final == "ollama":
                        tok_sec = metrics.get("tokens_per_second", 0)
                        total_tok = metrics.get("total_tokens", 0) # Ollamaではeval_count (生成トークン数)
                        if tok_sec > 0: footer_text_parts.append(f"{tok_sec:.2f} tok/s")
                        if total_tok > 0: footer_text_parts.append(f"{int(total_tok)} genTk")
                        # prompt_eval_count も表示するなら追加
                        # p_eval_c = metrics.get("prompt_eval_count", 0)
                        # if p_eval_c > 0: footer_text_parts.append(f"{int(p_eval_c)} prmTk")
                    elif model_type_final == "gemini":
                        tok_sec = metrics.get("tokens_per_second", 0)
                        gen_tok = metrics.get("generated_tokens", 0)
                        p_tok = metrics.get("prompt_tokens", 0)
                        if tok_sec > 0: footer_text_parts.append(f"{tok_sec:.2f} tok/s")
                        if gen_tok > 0: footer_text_parts.append(f"{int(gen_tok)} genTk")
                        if p_tok > 0: footer_text_parts.append(f"{int(p_tok)} prmTk")
                
                final_embed.set_footer(text=" | ".join(footer_text_parts))

                bot_message_data = {
                    "author_name": bot.user.display_name, "author_id": bot.user.id,
                    "content": final_response_text, 
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "is_bot": True
                }
                channel_data[channel_id]["history"].append(bot_message_data)
            else: 
                final_embed.title = "❓ 無応答"
                final_embed.description = "応答生成に失敗しました。入力内容を確認するか、モデルまたはプロンプトの変更をお試しください。"
                final_embed.color = discord.Color.dark_orange()
                final_embed.set_footer(text=f"Model: {model_name_final} ({model_type_final}) | Prompt: {prompt_name_final}")
                logger.warning(f"チャンネル {channel_id}: 応答テキストもエラーメッセージもなし。 model: {current_active_model}")
            
            await reply_message.edit(embed=final_embed)

        except discord.NotFound:
            logger.warning(f"チャンネル {channel_id}: 最終メッセージ編集失敗 - メッセージ消失 (ID: {reply_message.id})")
        except discord.HTTPException as e:
            logger.error(f"チャンネル {channel_id}: 最終メッセージ編集失敗: {e}")
            err_code = getattr(e, 'code', 'N/A'); err_text = str(e)[:100]
            try: await message.channel.send(f"エラー: 応答最終表示失敗 (Code: {err_code}) - {err_text}", reference=message, mention_author=False)
            except discord.HTTPException: pass
        except IndexError: 
            logger.error(f"チャンネル {channel_id}: 最終メッセージ編集失敗 - Embedなし")
            try: await reply_message.edit(content="エラー: 応答表示準備に失敗しました。", embed=None)
            except discord.HTTPException: pass
        except Exception as e:
            logger.error(f"チャンネル {channel_id}: 最終メッセージ編集中予期せぬエラー: {e}", exc_info=True)
            try: await reply_message.edit(content="エラー: 応答の最終表示中に予期せぬエラーが発生しました。", embed=None)
            except discord.HTTPException: pass


# --- スラッシュコマンド ---
# --- オートコンプリート ---#
async def model_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    ollama_models = []
    gemini_2_5_models = []
    gemini_2_0_models = []
    gemini_1_5_models = []
    gemini_1_0_models = []
    gemini_gemma_models = []  # GeminiがホストするGemmaモデル用
    other_gemini_models = []
    unknown_prefix_models = []

    # モデルをカテゴリに分類
    # available_bot_models は既にアルファベット順にソートされています
    for model_id in available_bot_models:
        if model_id.startswith("ollama:"):
            ollama_models.append(model_id)
        elif model_id.startswith("gemini:"):
            # "gemini:" プレフィックスを除いたモデル名部分を取得
            name_after_prefix = model_id[len("gemini:"):] 

            if name_after_prefix.startswith("gemini-2.5"):
                gemini_2_5_models.append(model_id)
            elif name_after_prefix.startswith("gemini-2.0"):
                gemini_2_0_models.append(model_id)
            elif name_after_prefix.startswith("gemini-1.5"):
                gemini_1_5_models.append(model_id)
            elif name_after_prefix.startswith("gemini-1.0"):
                gemini_1_0_models.append(model_id)
            elif name_after_prefix.startswith("gemma"): #例: "gemini:gemma-3-12b-it"
                gemini_gemma_models.append(model_id)
            else:
                other_gemini_models.append(model_id)
        else:
            unknown_prefix_models.append(model_id)

    # 指定された優先順位でリストを結合
    # 各カテゴリ内のモデルは、元のアルファベット順を維持します
    prioritized_model_list = (
        ollama_models +          # 1. Ollama
        gemini_2_5_models +      # 2. Gemini 2.5系
        gemini_2_0_models +      # 3. Gemini 2.0系
        gemini_1_5_models +      # 4. Gemini 1.5系
        gemini_1_0_models +      # 5. Gemini 1.0系
        gemini_gemma_models +    # 6. Gemini Gemma系
        other_gemini_models +    # 7. その他のGeminiモデル
        unknown_prefix_models    # 8. 未知のプレフィックス (通常は空)
    )

    choices = []
    current_lower = current.lower()

    for model_id in prioritized_model_list:
        if current_lower in model_id.lower():
            choices.append(app_commands.Choice(name=model_id, value=model_id))
        if len(choices) >= 25: # Discordの表示上限
            break
            
    return choices

async def prompt_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = []
    if current.lower() in PROMPT_NAME_DEFAULT.lower():
        choices.append(app_commands.Choice(name=PROMPT_NAME_DEFAULT, value=PROMPT_NAME_DEFAULT))
    
    custom_choices = [
        app_commands.Choice(name=name, value=name)
        for name in sorted(available_prompts.keys()) if current.lower() in name.lower()
    ]
    choices.extend(custom_choices)
    return choices[:25] # こちらも25件上限

# --- コマンド本体 ---
@bot.tree.command(name="stop", description="現在このチャンネルで生成中のAIの応答を停止します。")
async def stop_generation(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    if channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True); return

    if channel_data[channel_id]["is_generating"]:
        if not channel_data[channel_id]["stop_generation_requested"]:
            channel_data[channel_id]["stop_generation_requested"] = True
            logger.info(f"チャンネル {channel_id}: ユーザー {interaction.user} (ID: {interaction.user.id}) により停止リクエスト。")
            await interaction.response.send_message("⏹️ 応答の停止を試みています...", ephemeral=True)
            try:
                if interaction.channel: await interaction.channel.send(f"⚠️ {interaction.user.mention} が応答生成の停止を試みています。")
            except discord.HTTPException as e: logger.warning(f"チャンネル {channel_id}: 停止試行の公開ログ送信失敗: {e}")
        else:
            await interaction.response.send_message("ℹ️ 既に停止リクエストが送信されています。", ephemeral=True)
    else:
        await interaction.response.send_message("ℹ️ 現在このチャンネルで生成中の応答はありません。", ephemeral=True)


@bot.tree.command(name="model", description="使用するAIモデルと、そのモデル用のシステムプロンプトを設定します。")
@app_commands.describe(
    model_identifier="モデルを選択 (例: ollama:llama3, gemini:gemini-1.5-pro-latest)",
    prompt_name=f"適用するシステムプロンプト ('{PROMPT_DIR_NAME}'内のファイル名、または'{PROMPT_NAME_DEFAULT}')"
)
@app_commands.autocomplete(model_identifier=model_autocomplete, prompt_name=prompt_autocomplete)
async def select_model(interaction: discord.Interaction, model_identifier: str, prompt_name: str = None):
    global active_model 
    channel_id = interaction.channel_id
    if channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True); return
    
    await interaction.response.defer(ephemeral=True, thinking=False)

    sel_model_type, sel_model_name = get_model_type_and_name(model_identifier)
    if sel_model_type == "gemini" and not genai:
        await interaction.followup.send(f"❌ エラー: モデル '{model_identifier}' はGeminiモデルですが、Gemini APIが設定されていません。", ephemeral=True)
        return

    if model_identifier not in available_bot_models:
        # キャッシュが古い可能性もあるので、再取得を試みる
        await fetch_and_update_available_models()
        if model_identifier not in available_bot_models: # 再取得してもない場合
            model_list_str = "\n- ".join(available_bot_models) if available_bot_models else "キャッシュにモデルがありません。管理者に連絡してください。"
            await interaction.followup.send(
                f"❌ エラー: モデル '{model_identifier}' は利用できません (キャッシュ更新後も)。\n利用可能なモデル (キャッシュ):\n- {model_list_str}",
                ephemeral=True
            ); return

    previous_model_identifier = active_model
    previous_prompt_content = system_prompts.get(previous_model_identifier)
    prev_model_type, prev_model_name = get_model_type_and_name(previous_model_identifier)
    previous_prompt_name_display = get_prompt_name_from_content(previous_prompt_content)

    active_model = model_identifier 
    model_changed = previous_model_identifier != active_model
    
    prompt_actually_changed = False
    ephemeral_message_lines = []
    current_model_type_disp, current_model_name_disp = get_model_type_and_name(active_model)


    if prompt_name: 
        new_prompt_content_for_selected_model: str | None = None
        valid_prompt_selection = False

        if prompt_name == PROMPT_NAME_DEFAULT:
            new_prompt_content_for_selected_model = None 
            valid_prompt_selection = True
        elif prompt_name in available_prompts:
            new_prompt_content_for_selected_model = available_prompts[prompt_name]
            valid_prompt_selection = True
        else:
            ephemeral_message_lines.append(f"⚠️ 不明なプロンプト名 '{prompt_name}'。プロンプトは変更されませんでした。")
            current_prompt_for_active_model = system_prompts.get(active_model)
            prompt_name_to_display = get_prompt_name_from_content(current_prompt_for_active_model)
            ephemeral_message_lines.append(f"📄 モデル **{current_model_name_disp} ({current_model_type_disp})** のプロンプトは **{prompt_name_to_display}** のままです。")


        if valid_prompt_selection:
            current_prompt_for_active_model = system_prompts.get(active_model)
            if new_prompt_content_for_selected_model != current_prompt_for_active_model:
                system_prompts[active_model] = new_prompt_content_for_selected_model
                prompt_actually_changed = True
                ephemeral_message_lines.append(f"📄 モデル **{current_model_name_disp} ({current_model_type_disp})** のシステムプロンプトを **{prompt_name}** に設定しました。")
                logger.info(f"チャンネル {channel_id}: モデル '{active_model}' のプロンプト設定 -> '{prompt_name}' by {interaction.user}")
            else:
                ephemeral_message_lines.append(f"ℹ️ モデル **{current_model_name_disp} ({current_model_type_disp})** のプロンプトは既に **{prompt_name}** です。")
    else: 
        if active_model not in system_prompts: 
            system_prompts[active_model] = None 
            prompt_actually_changed = True # 暗黙的にデフォルトに変更された (モデル変更時など)
            ephemeral_message_lines.append(f"📄 モデル **{current_model_name_disp} ({current_model_type_disp})** にプロンプト未設定だったため、**{PROMPT_NAME_DEFAULT}** を設定。")
        else: 
            maintained_prompt_content = system_prompts.get(active_model)
            maintained_prompt_name = get_prompt_name_from_content(maintained_prompt_content)
            ephemeral_message_lines.append(f"ℹ️ モデル **{current_model_name_disp} ({current_model_type_disp})** のプロンプト **{maintained_prompt_name}** を維持。")


    final_ephemeral_message = []
    if model_changed:
        final_ephemeral_message.append(f"✅ モデル変更: **{prev_model_name or 'N/A'} ({prev_model_type or 'N/A'})** → **{current_model_name_disp} ({current_model_type_disp})**。")
        logger.info(f"チャンネル {channel_id}: アクティブモデル変更 -> '{active_model}' by {interaction.user}")
    else:
        final_ephemeral_message.append(f"ℹ️ モデルは **{current_model_name_disp} ({current_model_type_disp})** のまま。")
    
    final_ephemeral_message.extend(ephemeral_message_lines)
    await interaction.followup.send("\n".join(final_ephemeral_message), ephemeral=True)

    if model_changed or prompt_actually_changed:
        log_parts = []
        final_active_prompt_name = get_prompt_name_from_content(system_prompts.get(active_model))
        
        if model_changed:
            log_parts.append(f"モデル: **{prev_model_name or 'N/A'} ({prev_model_type or 'N/A'})** → **{current_model_name_disp} ({current_model_type_disp})**")
        
        # プロンプトが実際に変わったか、あるいはモデルが変わってプロンプトが暗黙的に維持/設定された場合に表示
        if prompt_actually_changed:
            log_parts.append(f"プロンプト ({current_model_name_disp}): **{previous_prompt_name_display if model_changed else previous_prompt_name_display}** → **{final_active_prompt_name}**")
        elif model_changed: # モデルは変わったがプロンプトは明示的に変更されず、維持された場合
             log_parts.append(f"プロンプト ({current_model_name_disp}): **{final_active_prompt_name}** (維持)")


        if log_parts:
            public_log_message = f"🔧 {interaction.user.mention} が設定変更: " + ", ".join(log_parts)
            try:
                if interaction.channel: await interaction.channel.send(public_log_message)
            except discord.HTTPException as e: logger.error(f"チャンネル {channel_id}: モデル/プロンプト変更公開ログ送信失敗: {e}")


@bot.tree.command(name="set_prompt", description="現在アクティブなモデルのシステムプロンプトを設定します。")
@app_commands.describe(prompt_name=f"適用するシステムプロンプト ('{PROMPT_DIR_NAME}'内のファイル名、または'{PROMPT_NAME_DEFAULT}')")
@app_commands.autocomplete(prompt_name=prompt_autocomplete)
async def set_prompt(interaction: discord.Interaction, prompt_name: str):
    channel_id = interaction.channel_id
    if channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True); return
    if not active_model:
        await interaction.response.send_message("⚠️ モデル未選択。`/model` コマンドで選択してください。", ephemeral=True); return
    
    await interaction.response.defer(ephemeral=True, thinking=False)

    current_prompt_content = system_prompts.get(active_model)
    current_prompt_name_display = get_prompt_name_from_content(current_prompt_content)
    new_prompt_content: str | None = None
    valid_prompt = False

    if prompt_name == PROMPT_NAME_DEFAULT:
        new_prompt_content = None
        valid_prompt = True
    elif prompt_name in available_prompts:
        new_prompt_content = available_prompts[prompt_name]
        valid_prompt = True
    else:
        await interaction.followup.send(f"❌ エラー: 不明なプロンプト名 '{prompt_name}'。", ephemeral=True); return

    if valid_prompt:
        if new_prompt_content != current_prompt_content:
            system_prompts[active_model] = new_prompt_content
            model_type_disp, model_name_disp = get_model_type_and_name(active_model)
            logger.info(f"チャンネル {channel_id}: モデル '{active_model}' プロンプト変更 -> '{prompt_name}' by {interaction.user}")
            await interaction.followup.send(f"✅ モデル **{model_name_disp} ({model_type_disp})** のプロンプト設定 -> **{prompt_name}**。", ephemeral=True)
            
            public_log_message = f"🔧 {interaction.user.mention} がモデル **{model_name_disp} ({model_type_disp})** のプロンプト変更: **{current_prompt_name_display}** → **{prompt_name}**"
            try:
                if interaction.channel: await interaction.channel.send(public_log_message)
            except discord.HTTPException as e: logger.error(f"チャンネル {channel_id}: プロンプト変更公開ログ送信失敗: {e}")
        else:
            model_type_disp, model_name_disp = get_model_type_and_name(active_model)
            await interaction.followup.send(f"ℹ️ モデル **{model_name_disp} ({model_type_disp})** のプロンプトは既に **{prompt_name}**。", ephemeral=True)


@bot.tree.command(name="clear_history", description="このチャンネルの会話履歴と応答統計を消去します。")
async def clear_history_command(interaction: discord.Interaction): 
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True); return
    
    await interaction.response.defer(ephemeral=True, thinking=False)
    
    if target_channel_id in channel_data:
        channel_data[target_channel_id]["history"].clear()
        channel_data[target_channel_id]["stats"].clear()
        logger.info(f"チャンネルID {target_channel_id} 会話履歴/統計クリア完了 by {interaction.user}。")
        await interaction.followup.send("✅ このチャンネルの会話履歴と応答統計をクリアしました。", ephemeral=True)
    else: 
        await interaction.followup.send("ℹ️ クリア対象の会話履歴が見つかりませんでした。", ephemeral=True)


@bot.tree.command(name="show_history", description="このチャンネルの直近の会話履歴を表示します。")
@app_commands.describe(count=f"表示する履歴の件数 (デフォルト10, 最大 {HISTORY_LIMIT})")
async def show_history_command(interaction: discord.Interaction, count: app_commands.Range[int, 1, None] = 10): 
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True); return

    await interaction.response.defer(ephemeral=True, thinking=False)
    history = channel_data[target_channel_id]["history"]
    if not history:
        await interaction.followup.send("表示できる会話履歴がありません。", ephemeral=True); return

    actual_count = min(count, HISTORY_LIMIT, len(history))
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
        entry_text = f"`{start_index + i + 1}`. {author_str}:\n```\n{content_safe}\n```\n" 

        if len(history_text) + len(entry_text) > 3900: 
            history_text += "... (表示数上限のため省略)"
            break
        history_text += entry_text
    
    embed.description = history_text if history_text else "履歴内容が空です。"
    embed.set_footer(text=f"最大保持数: {HISTORY_LIMIT}件")
    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="list_models", description="利用可能な全てのAIモデルを一覧表示します。")
async def list_models_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True) 

    if not available_bot_models:
        await interaction.followup.send("現在利用可能なモデルはありません。", ephemeral=True)
        return

    embed = Embed(title="利用可能なAIモデル一覧", color=discord.Color.blue())
    
    # オートコンプリートと同じカテゴリ分けロジックを使用
    ollama_models_list = []
    gemini_2_5_models_list = []
    gemini_2_0_models_list = []
    gemini_1_5_models_list = []
    gemini_1_0_models_list = []
    gemini_gemma_models_list = []
    other_gemini_models_list = []
    unknown_prefix_models_list = []

    for model_id in available_bot_models: # available_bot_models はソート済み
        if model_id.startswith("ollama:"):
            ollama_models_list.append(model_id)
        elif model_id.startswith("gemini:"):
            name_after_prefix = model_id[len("gemini:"):]
            if name_after_prefix.startswith("gemini-2.5"):
                gemini_2_5_models_list.append(model_id)
            elif name_after_prefix.startswith("gemini-2.0"):
                gemini_2_0_models_list.append(model_id)
            elif name_after_prefix.startswith("gemini-1.5"):
                gemini_1_5_models_list.append(model_id)
            elif name_after_prefix.startswith("gemini-1.0"):
                gemini_1_0_models_list.append(model_id)
            elif name_after_prefix.startswith("gemma"):
                gemini_gemma_models_list.append(model_id)
            else:
                other_gemini_models_list.append(model_id)
        else:
            unknown_prefix_models_list.append(model_id)

    # Embedフィールドにモデルリストを追加するヘルパー関数 (変更なしで再利用可能)
    async def add_models_to_field(title: str, models_in_category: list[str], embed_obj: Embed):
        if not models_in_category:
            if len(embed_obj.fields) < 25:
                embed_obj.add_field(name=title, value="- (なし)", inline=False)
            return

        current_field_text = ""
        field_part_count = 1
        base_title = title

        for i, model_name in enumerate(sorted(models_in_category)): # カテゴリ内でソートして表示
            line_to_add = f"- `{model_name}`\n"
            
            if len(current_field_text) + len(line_to_add) > 1020: 
                if len(embed_obj.fields) < 25:
                    field_title_to_use = f"{base_title} ({field_part_count})" if field_part_count > 1 or (len(models_in_category) - i > 0) else base_title
                    embed_obj.add_field(name=field_title_to_use, value=current_field_text.strip() if current_field_text else " ", inline=False)
                    current_field_text = line_to_add 
                    field_part_count += 1
                else: 
                    if not embed_obj.footer: 
                        embed_obj.set_footer(text="モデル多数のため、リストの一部が省略されています。")
                    return 
            else:
                current_field_text += line_to_add
        
        if current_field_text and len(embed_obj.fields) < 25:
            field_title_to_use = f"{base_title} ({field_part_count})" if field_part_count > 1 else base_title
            embed_obj.add_field(name=field_title_to_use, value=current_field_text.strip(), inline=False)
        elif current_field_text and not embed_obj.footer: 
             embed_obj.set_footer(text="モデル多数のため、リストの一部が省略されています。")

    # 指定された優先順位でカテゴリごとにフィールドを追加
    await add_models_to_field("Ollama Models", ollama_models_list, embed)
    if len(embed.fields) < 25: await add_models_to_field("Gemini 2.5 Series Models", gemini_2_5_models_list, embed)
    if len(embed.fields) < 25: await add_models_to_field("Gemini 2.0 Series Models", gemini_2_0_models_list, embed)
    if len(embed.fields) < 25: await add_models_to_field("Gemini 1.5 Series Models", gemini_1_5_models_list, embed)
    if len(embed.fields) < 25: await add_models_to_field("Gemini 1.0 Series Models", gemini_1_0_models_list, embed)
    if len(embed.fields) < 25: await add_models_to_field("Gemini Gemma Models", gemini_gemma_models_list, embed)
    if len(embed.fields) < 25: await add_models_to_field("Other Gemini Models", other_gemini_models_list, embed)
    if unknown_prefix_models_list and len(embed.fields) < 25:
        await add_models_to_field("Unknown Prefix Models", unknown_prefix_models_list, embed)
    
    if not embed.fields and not available_bot_models: # 最初の available_bot_models チェックで捕捉されるが念のため
        embed.description = "表示可能なモデルがありません。"
    elif not embed.fields and available_bot_models : # カテゴリ分けされたが、何らかの理由でフィールドが一つも作られなかった場合
        embed.description = "モデルはありますが、カテゴリ表示に失敗しました。"


    try:
        await interaction.followup.send(embed=embed, ephemeral=True)
    except discord.HTTPException as e:
        logger.error(f"/list_models コマンド実行中にDiscord APIエラー: {e}")
        await interaction.followup.send("モデルリストの表示中にエラーが発生しました。リストが大きすぎるか、予期せぬ問題が発生した可能性があります。", ephemeral=True)


@bot.tree.command(name="set_param", description="LLMの生成パラメータ(temperature, top_k, top_p)を調整します。")
@app_commands.describe(parameter="調整するパラメータ名", value="設定する値 (例: 0.7, 50)。未設定に戻す場合は 'none' または 'default' と入力。")
@app_commands.choices(parameter=[
    app_commands.Choice(name="temperature", value="temperature"),
    app_commands.Choice(name="top_k", value="top_k"),
    app_commands.Choice(name="top_p", value="top_p"),
])
async def set_parameter(interaction: discord.Interaction, parameter: app_commands.Choice[str], value: str):
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True); return
    
    await interaction.response.defer(ephemeral=True, thinking=False)
    
    param_name = parameter.value
    current_params = channel_data[target_channel_id]["params"]
    response_message = ""
    original_value = current_params.get(param_name) 

    try:
        new_value_internal: float | int | None = None 
        
        if value.lower() in ['none', 'default', 'null', '']: 
            new_value_internal = None
        elif param_name == "temperature":
            try:
                float_val = float(value)
                if 0.0 <= float_val <= 2.0: new_value_internal = float_val
                else: raise ValueError("Temperature は 0.0 から 2.0 の範囲で指定してください。")
            except ValueError: raise ValueError("Temperature には数値を入力してください。")
        elif param_name == "top_k":
            try:
                int_val = int(value)
                if int_val >= 0: new_value_internal = int_val 
                else: raise ValueError("Top K は 0 以上の整数で指定してください。")
            except ValueError: raise ValueError("Top K には整数を入力してください。")
        elif param_name == "top_p":
            try:
                float_val = float(value)
                if 0.0 <= float_val <= 1.0: new_value_internal = float_val
                else: raise ValueError("Top P は 0.0 から 1.0 の範囲で指定してください。")
            except ValueError: raise ValueError("Top P には数値を入力してください。")

        is_changed = False
        if original_value is None and new_value_internal is not None: is_changed = True
        elif original_value is not None and new_value_internal is None: is_changed = True
        elif isinstance(original_value, (int, float)) and isinstance(new_value_internal, (int, float)):
            if not math.isclose(original_value, new_value_internal, rel_tol=1e-9): is_changed = True
        elif original_value != new_value_internal : # 上記以外 (通常は type が異なる場合など)
            is_changed = True


        if is_changed:
            current_params[param_name] = new_value_internal
            logger.info(f"チャンネル {target_channel_id}: パラメータ '{param_name}' 設定 -> '{new_value_internal}' by {interaction.user}")
            response_message = f"✅ パラメータ **{param_name}** 設定 -> **{new_value_internal if new_value_internal is not None else '未設定 (APIデフォルト)'}**。"
        else:
            response_message = f"ℹ️ パラメータ **{param_name}** は既に **{original_value if original_value is not None else '未設定 (APIデフォルト)'}**。"

    except ValueError as e:
        logger.warning(f"チャンネル {target_channel_id}: パラメータ設定エラー ({param_name}={value}): {e} by {interaction.user}")
        response_message = f"⚠️ 設定値エラー: {e}"
    except Exception as e:
        logger.error(f"チャンネル {target_channel_id}: パラメータ設定中エラー: {e}", exc_info=True)
        response_message = "❌ パラメータ設定中に予期せぬエラーが発生しました。"
        
    await interaction.followup.send(response_message, ephemeral=True)


@bot.tree.command(name="stats", description="現在の設定と直近の応答生成統計を表示します。")
async def show_stats_command(interaction: discord.Interaction): 
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True); return

    await interaction.response.defer(ephemeral=True, thinking=False)

    stats_deque = channel_data[target_channel_id]["stats"]
    total_stats_count = len(stats_deque)
    stats_max_len = channel_data[target_channel_id]["stats"].maxlen or 50

    embed = Embed(title="📊 BOTステータス & 応答統計", color=discord.Color.green())

    active_model_type, active_model_name = get_model_type_and_name(active_model)
    current_model_str = f"**{active_model_name or 'N/A'} ({active_model_type or 'N/A'})**" if active_model else "未設定"
    
    current_prompt_name_str = "N/A"
    if active_model:
        current_prompt_content = system_prompts.get(active_model)
        current_prompt_name_str = f"**{get_prompt_name_from_content(current_prompt_content)}**"
    
    current_params = channel_data[target_channel_id]["params"]
    params_str_parts = []
    if current_params.get("temperature") is not None: params_str_parts.append(f"Temp={current_params['temperature']}")
    if current_params.get("top_k") is not None: params_str_parts.append(f"TopK={current_params['top_k']}")
    if current_params.get("top_p") is not None: params_str_parts.append(f"TopP={current_params['top_p']}")
    params_str = ", ".join(params_str_parts) if params_str_parts else "APIデフォルト"

    embed.add_field(
        name="現在の設定",
        value=f"モデル: {current_model_str}\nプロンプト: {current_prompt_name_str}\nパラメータ: `{params_str}`",
        inline=False
    )

    if not stats_deque:
        embed.add_field(name=f"応答統計 (直近 0/{stats_max_len} 回)", value="記録なし。", inline=False)
    else:
        total_duration_sum, total_generated_tokens_sum, total_tps_sum, valid_tps_entries = 0.0, 0, 0.0, 0
        
        for stat_entry in stats_deque:
            duration = stat_entry.get("total_duration", 0.0)
            if 0.01 < duration < 600 : total_duration_sum += duration

            generated_tokens = stat_entry.get("total_tokens", stat_entry.get("generated_tokens", 0)) # Ollama or Gemini
            if generated_tokens > 0: total_generated_tokens_sum += generated_tokens
            
            tps = stat_entry.get("tokens_per_second", 0.0)
            if 0.01 < tps < 10000 : 
                total_tps_sum += tps
                valid_tps_entries += 1
        
        avg_duration = total_duration_sum / total_stats_count if total_stats_count > 0 else 0.0
        avg_generated_tokens = total_generated_tokens_sum / total_stats_count if total_stats_count > 0 else 0.0
        avg_tps = total_tps_sum / valid_tps_entries if valid_tps_entries > 0 else 0.0

        stats_summary = (
            f"平均応答時間: **{avg_duration:.2f} 秒**\n"
            f"平均生成トークン数: **{avg_generated_tokens:.1f} トークン**\n"
            f"平均TPS (有効なエントリのみ): **{avg_tps:.2f} tok/s**"
        )
        embed.add_field(name=f"応答統計 (直近 {total_stats_count}/{stats_max_len} 回)", value=stats_summary, inline=False)

    api_urls = [f"Ollama: {OLLAMA_API_URL}"]
    if genai: api_urls.append("Gemini: (Google Cloud)") 
    
    embed.set_footer(text=f"履歴保持数: {HISTORY_LIMIT} | APIエンドポイント: {', '.join(api_urls)}")
    await interaction.followup.send(embed=embed, ephemeral=True)


# --- BOT起動 ---
if __name__ == "__main__":
    if not TOKEN: logger.critical("環境変数 'DISCORD_TOKEN' が設定されていません。BOTを終了します。"); sys.exit(1)
    if CHAT_CHANNEL_ID is None: logger.critical("環境変数 'CHAT_CHANNEL_ID' が無効です。BOTを終了します。"); sys.exit(1) 
    if aiofiles is None: logger.warning("`aiofiles` がインストールされていません。一部のファイル関連機能が制限される可能性があります。")
    
    if genai is not None and not GEMINI_API_KEY:
        logger.warning("Geminiライブラリはありますが、環境変数 'GEMINI_API_KEY' が設定されていません。Gemini API機能は利用できません。")
    elif genai is None and GEMINI_API_KEY: 
         logger.warning("環境変数 'GEMINI_API_KEY' は設定されていますが、'google-generativeai'ライブラリのインポートに失敗しました。Gemini API機能は利用できません。")


    logger.info("--- LLM Discord BOT 起動プロセス開始 ---")
    logger.info(f"監視チャンネルID: {CHAT_CHANNEL_ID}")
    logger.info(f"デフォルトモデル: {DEFAULT_MODEL or '未設定 (起動時に自動選択試行)'}")
    logger.info(f"履歴保持数: {HISTORY_LIMIT}")
    logger.info(f"Ollama API URL: {OLLAMA_API_URL}")
    if GEMINI_API_KEY and genai:
        logger.info("Gemini APIキー: 設定済み (Gemini API 利用可能)")
    elif GEMINI_API_KEY and not genai:
        logger.info("Gemini APIキー: 設定済み (ただしライブラリインポート失敗のためGemini API 利用不可)")
    else:
        logger.info("Gemini APIキー: 未設定 (Gemini API 利用不可)")

    logger.info(f"カスタムプロンプトDir: {prompts_dir_path}")
    logger.info(f"プロンプトリロード間隔: {PROMPT_RELOAD_INTERVAL_MINUTES} 分")
    logger.info(f"モデルリスト更新間隔: {MODEL_UPDATE_INTERVAL_MINUTES} 分")
    logger.info("-------------------------------------------")

    try:
        bot.run(TOKEN, log_handler=None) 
    except discord.LoginFailure:
        logger.critical("Discordへのログインに失敗しました。DISCORD_TOKENが正しいか確認してください。")
    except discord.PrivilegedIntentsRequired:
        logger.critical("必要なPrivileged Intents (特にMessage Content Intent) がDiscord Developer Portalで有効になっていません。確認してください。")
    except ImportError as e: 
        logger.critical(f"必要なライブラリのインポートに失敗しました: {e}")
    except Exception as e:
        logger.critical(f"BOT起動中に致命的なエラーが発生しました: {e}", exc_info=True)

# --- END OF FILE bot.py ---