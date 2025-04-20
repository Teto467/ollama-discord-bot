import asyncio
import sys
import os
import json
import logging
import datetime
import time
from collections import defaultdict, deque
import math # NaNチェック用に追加

import discord
from discord import app_commands, Embed
from discord.ext import commands
from dotenv import load_dotenv
import aiohttp

# --- Windows用イベントループポリシーの設定 ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, # DEBUGレベルに変更すると、より詳細なログが見れます
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

# --- BOT設定 ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# --- グローバル変数 & 定数 ---
active_model = DEFAULT_MODEL
system_prompts = {} # モデルごとのカスタムシステムプロンプト {model_name: prompt_text}
# ★変更: チャンネルごとのデータを保持する構造
# history: 会話履歴 (deque)
# params: Ollama生成パラメータ (dict)
# stats: 応答統計 (deque of dict)
channel_data = defaultdict(lambda: {
    "history": deque(maxlen=HISTORY_LIMIT),
    "params": {"temperature": 0.7}, # デフォルトパラメータ
    "stats": deque(maxlen=50) # 直近50回の統計を保持
})

STREAM_UPDATE_INTERVAL = 1.5 # ストリーミング応答の更新間隔 (秒)
STREAM_UPDATE_CHARS = 75    # ストリーミング応答の更新文字数閾値
STATS_HISTORY_MAX = 50     # 保持する統計情報の最大数

# --- スクリプトファイルのディレクトリ基準のファイルパス設定 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
default_prompt_file_path = os.path.join(script_dir, "System prompt.txt")

# --- 非同期関数 ---

async def get_available_models():
    """Ollama APIから利用可能なモデルの一覧を取得する"""
    url = f"{OLLAMA_API_URL}/api/tags"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    return [model['name'] for model in models]
                else:
                    logger.warning(f"モデル一覧取得APIエラー - ステータス: {response.status}, URL: {url}")
                    return []
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Ollama APIへの接続に失敗しました: {e}. URL: {url}")
        return []
    except Exception as e:
        logger.error(f"モデル一覧の取得中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return []

async def fetch_channel_history(channel: discord.TextChannel, limit: int = 100):
    """指定されたチャンネルの過去メッセージを取得し、内部履歴に追加する"""
    if not isinstance(channel, discord.TextChannel):
        logger.warning(f"指定されたチャンネルが無効です: {channel}")
        return

    logger.info(f"チャンネル '{channel.name}' (ID: {channel.id}) の履歴取得を開始 (最大{limit}件)...")
    try:
        messages_to_add = []
        count = 0
        async for message in channel.history(limit=limit):
            # ボット自身のメッセージか、ユーザーのメッセージのみ取得
            if not message.author.bot or message.author.id == bot.user.id:
                 # コマンド呼び出しや空メッセージを除外することが望ましい場合がある
                if message.content: # 空でないメッセージのみ
                    messages_to_add.append({
                        "author_name": message.author.display_name,
                        "author_id": message.author.id,
                        "content": message.content,
                        "timestamp": message.created_at.isoformat(),
                        "is_bot": message.author.bot
                    })
                    count += 1

        added_count = 0
        history_deque = channel_data[channel.id]["history"]
        existing_timestamps_contents = { (msg["timestamp"], msg["content"]) for msg in history_deque }

        for msg in reversed(messages_to_add):
             # 重複チェックを強化 (タイムスタンプと内容で判断)
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
    if not history_deque:
        return []

    messages = []
    for msg in history_deque:
        role = "assistant" if msg["is_bot"] and msg["author_id"] == bot.user.id else "user"
        messages.append({"role": role, "content": msg["content"]})
    return messages

async def load_system_prompt_from_file(file_path: str = default_prompt_file_path) -> str | None:
    """指定されたファイルパスからシステムプロンプトを読み込む"""
    logger.info(f"システムプロンプトファイル '{file_path}' の読み込み試行...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                logger.warning(f"システムプロンプトファイル '{file_path}' は空です。")
                return None
            logger.info(f"システムプロンプトファイル '{file_path}' の読み込み成功。")
            return content
    except FileNotFoundError:
        logger.error(f"システムプロンプトファイル '{file_path}' が見つかりませんでした。")
        return None
    except PermissionError:
        logger.error(f"システムプロンプトファイル '{file_path}' の読み取り権限がありません。")
        return None
    except UnicodeDecodeError:
        logger.error(f"システムプロンプトファイル '{file_path}' の読み込み中にエンコーディングエラーが発生しました。ファイルがUTF-8で保存されているか確認してください。")
        return None
    except Exception as e:
        logger.error(f"システムプロンプトファイル '{file_path}' の読み込み中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return None

# --- ★新規: ストリーミング応答生成関数 ---
async def generate_response_stream(
    prompt: str,
    channel_id: int,
    message_to_edit: discord.Message, # 編集対象のメッセージ
    model: str = None,
    system_prompt_override: str = None
) -> tuple[str | None, dict | None, str | None]:
    """
    Ollama APIにリクエストを送信し、ストリーミングで応答を生成・表示する。
    戻り値: (最終的な応答テキスト, パフォーマンスメトリクス, エラーメッセージ)
    """
    if not model:
        model = active_model
        if not model:
            logger.error("応答生成ができません: デフォルトモデルもアクティブモデルも設定されていません。")
            return None, None, "エラー: 使用するモデルが設定されていません。"

    # チャンネル固有のパラメータを取得
    channel_params = channel_data[channel_id]["params"]

    # システムプロンプトを決定
    system_prompt = system_prompt_override
    using_custom_prompt = False
    if not system_prompt:
        default_prompt = "あなたはDiscordサーバーでユーザーを支援するAIアシスタントです。会話の流れを理解し、適切かつ役立つ応答を心がけてください。"
        custom_prompt = system_prompts.get(model)
        if custom_prompt:
            system_prompt = custom_prompt
            using_custom_prompt = True
            logger.debug(f"モデル '{model}' のカスタムシステムプロンプトを使用します。")
        else:
            system_prompt = default_prompt
            logger.debug(f"モデル '{model}' のデフォルトシステムプロンプトを使用します。")

    # 会話履歴を取得 (Ollama /v1/chat 形式に合わせる場合)
    # history_messages = build_chat_context(channel_id)
    # messages = history_messages + [{"role": "user", "content": prompt}]

    # Ollama APIへのリクエストデータ (/api/generate を使う場合)
    data = {
        "model": model,
        "prompt": prompt, # ユーザーの最新プロンプト
        # "context": [], # 必要なら過去のcontext IDを指定 (今回は履歴を毎回送信)
        "system": system_prompt,
        "stream": True, # ストリーミングを有効化
        "options": channel_params # チャンネル固有のパラメータを使用
    }
    # もし会話履歴全体をpromptとして含めたい場合（非推奨だが旧来の方法に近い）
    # context_str = build_chat_context_string(channel_id) # 別途定義が必要
    # full_prompt = f"{context_str}\nUser: {prompt}"
    # data["prompt"] = full_prompt


    logger.info(f"モデル '{model}' にストリーミングリクエスト送信中 (チャンネルID: {channel_id})...")
    logger.debug(f"送信データ (抜粋): { {k: v for k, v in data.items() if k != 'system'} }") # systemは長いので除外
    logger.debug(f"システムプロンプト (カスタム使用: {using_custom_prompt}): {system_prompt[:200]}...")

    full_response = ""
    last_update_time = time.monotonic()
    last_update_len = 0
    performance_metrics = None
    error_message = None
    start_time = time.monotonic()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_API_URL}/api/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=600) # タイムアウトを長めに設定
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama APIエラー (ステータス: {response.status}): {error_text}")
                    return None, None, f"Ollama APIとの通信中にエラーが発生しました (コード: {response.status})。"

                # ストリーミング処理
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            # print(f"DEBUG chunk: {chunk}") # デバッグ用

                            if "response" in chunk and not chunk.get("done", False):
                                full_response += chunk["response"]
                                current_time = time.monotonic()
                                # 一定時間経過 or 一定文字数追加でメッセージを編集
                                if (current_time - last_update_time > STREAM_UPDATE_INTERVAL or
                                        len(full_response) - last_update_len > STREAM_UPDATE_CHARS):
                                    # メッセージが長すぎる場合の切り捨て
                                    display_response = full_response
                                    if len(display_response) > 3900: # Embed Descriptionの上限近く
                                        display_response = display_response[:3900] + "..."

                                    embed = message_to_edit.embeds[0]
                                    embed.description = display_response + " ▌" # カーソル風
                                    try:
                                        await message_to_edit.edit(embed=embed)
                                        last_update_time = current_time
                                        last_update_len = len(full_response)
                                    except discord.HTTPException as e:
                                        logger.warning(f"ストリーミング中のメッセージ編集に失敗: {e}")
                                        # 失敗しても処理は続行するが、更新は止まる可能性がある
                                    await asyncio.sleep(0.1) # 短い待機

                            if chunk.get("done", False):
                                # 応答完了
                                end_time = time.monotonic()
                                total_duration = end_time - start_time
                                performance_metrics = {
                                    "total_duration": total_duration,
                                    "load_duration_sec": chunk.get('load_duration', 0) / 1e9,
                                    "prompt_eval_count": chunk.get('prompt_eval_count', 0),
                                    "prompt_eval_duration_sec": chunk.get('prompt_eval_duration', 0) / 1e9,
                                    "eval_count": chunk.get('eval_count', 0), # これが生成トークン数
                                    "eval_duration_sec": chunk.get('eval_duration', 0) / 1e9
                                }
                                eval_duration_sec = performance_metrics["eval_duration_sec"]
                                eval_count = performance_metrics["eval_count"]
                                if eval_duration_sec > 0 and eval_count > 0:
                                    performance_metrics["tokens_per_second"] = eval_count / eval_duration_sec
                                else:
                                    performance_metrics["tokens_per_second"] = 0
                                performance_metrics["total_tokens"] = eval_count

                                logger.info(f"ストリーミング生成完了 ({total_duration:.2f}秒). メトリクス: {performance_metrics.get('tokens_per_second', 0):.2f} tok/sec, {performance_metrics.get('total_tokens', 0)} tokens.")
                                break # 完了したのでループを抜ける

                        except json.JSONDecodeError as e:
                            logger.error(f"Ollama APIからのストリーミングJSON解析に失敗: {e}. Line: {line.decode('utf-8', errors='ignore')}")
                            # 不正な行があっても続行を試みる
                        except Exception as e:
                             logger.error(f"ストリーミング処理中に予期せぬエラー: {e}", exc_info=True)
                             error_message = "ストリーミング処理中にエラーが発生しました。"
                             # エラーが発生しても、それまでの応答は返す
                             break

    except asyncio.TimeoutError:
        logger.error("Ollama APIへのリクエストがタイムアウトしました。")
        error_message = "リクエストがタイムアウトしました。Ollamaサーバーの状態を確認するか、後でもう一度お試しください。"
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Ollama APIへの接続に失敗しました: {e}")
        error_message = f"Ollama API ({OLLAMA_API_URL}) に接続できませんでした。サーバーが起動しているか確認してください。"
    except Exception as e:
        logger.error(f"Ollama APIリクエスト中に予期せぬエラーが発生しました: {e}", exc_info=True)
        error_message = f"予期せぬエラーが発生しました: {str(e)}"

    # エラーが発生した場合でも、それまでに受信した応答とメトリクス（あれば）を返す
    return full_response.strip(), performance_metrics, error_message


# --- Discord イベントハンドラ ---

@bot.event
async def on_ready():
    """BOTがDiscordに接続し、準備が完了したときに呼び出される"""
    logger.info(f'{bot.user} (ID: {bot.user.id}) としてDiscordにログインしました')

    global active_model
    if not active_model:
        logger.warning("デフォルトモデルが設定されていません。利用可能なモデルから最初のモデルを選択します...")
        available_models = await get_available_models()
        if available_models:
            active_model = available_models[0]
            logger.info(f"アクティブモデルを '{active_model}' に設定しました。")
        else:
            logger.error("利用可能なモデルが見つかりませんでした。`/model` コマンドで手動設定が必要です。")

    chat_channel = bot.get_channel(CHAT_CHANNEL_ID)
    if chat_channel and isinstance(chat_channel, discord.TextChannel):
        logger.info(f"チャットチャンネル '{chat_channel.name}' (ID: {CHAT_CHANNEL_ID}) を認識しました。")
        await fetch_channel_history(chat_channel, limit=HISTORY_LIMIT * 2) # 履歴取得量を増やす
    else:
        logger.error(f"指定されたチャットチャンネルID ({CHAT_CHANNEL_ID}) が見つからないか、テキストチャンネルではありません。")

    try:
        synced = await bot.tree.sync()
        logger.info(f'{len(synced)}個のスラッシュコマンドを同期しました')
    except Exception as e:
        logger.error(f"スラッシュコマンドの同期中にエラーが発生しました: {e}")

@bot.event
async def on_message(message: discord.Message):
    """メッセージが送信されたときに呼び出される"""
    if message.author == bot.user:
        return
    if message.channel.id != CHAT_CHANNEL_ID: # 指定チャンネル以外は無視
        return
    # コマンドプリフィックスやスラッシュコマンド呼び出しは無視
    if message.content.startswith(bot.command_prefix) or message.content.startswith('/'):
        # ただし、標準のコマンド処理は行う
        await bot.process_commands(message)
        return

    # ユーザーメッセージを履歴に追加
    user_message_data = {
        "author_name": message.author.display_name,
        "author_id": message.author.id,
        "content": message.content,
        "timestamp": message.created_at.isoformat(),
        "is_bot": False
    }
    channel_data[message.channel.id]["history"].append(user_message_data)
    logger.debug(f"履歴追加 (User): {user_message_data['author_name']} - {user_message_data['content'][:50]}...")

    # ストリーミング応答開始
    async with message.channel.typing():
        # まず空のEmbedで応答メッセージを作成
        placeholder_embed = Embed(description="思考中... 🤔", color=discord.Color.light_gray())
        placeholder_embed.set_footer(text=f"Model: {active_model}")
        try:
            reply_message = await message.reply(embed=placeholder_embed, mention_author=False)
        except discord.HTTPException as e:
            logger.error(f"プレースホルダーメッセージの送信に失敗: {e}")
            return # 送信失敗時は処理中断

        # ストリーミング生成関数を呼び出し
        final_response_text, metrics, error_msg = await generate_response_stream(
            prompt=message.content,
            channel_id=message.channel.id,
            message_to_edit=reply_message, # 編集対象メッセージを渡す
            model=active_model
        )

    # 最終的なメッセージを編集
    final_embed = reply_message.embeds[0] # 既存のEmbedを取得して編集

    if error_msg:
        # エラーメッセージを最終結果として表示
        final_embed.description = f"⚠️ **エラーが発生しました** ⚠️\n\n{error_msg}"
        final_embed.color = discord.Color.red()
        logger.error(f"応答生成中にエラーが発生: {error_msg}")
    elif final_response_text:
        # 成功した場合、最終テキストとメトリクスを表示
        final_embed.description = final_response_text
        final_embed.color = discord.Color.blue()

        footer_text = f"Model: {active_model}"
        if metrics:
            # 統計情報をキューに追加
            channel_data[message.channel.id]["stats"].append(metrics)

            tok_sec = metrics.get("tokens_per_second", 0)
            total_tokens = metrics.get("total_tokens", 0)
            duration = metrics.get("total_duration", 0)
            # NaNチェックを追加
            if not math.isnan(tok_sec): footer_text += f" | {tok_sec:.2f} tok/s"
            if total_tokens > 0: footer_text += f" | {total_tokens} tokens"
            if not math.isnan(duration): footer_text += f" | {duration:.2f}s"
            # メトリクスの詳細をログに出力 (DEBUGレベル)
            logger.debug(f"応答メトリクス詳細: {metrics}")

        final_embed.set_footer(text=footer_text)

        # BOTの応答を履歴に追加
        bot_message_data = {
            "author_name": bot.user.display_name,
            "author_id": bot.user.id,
            "content": final_response_text,
            "timestamp": reply_message.created_at.isoformat(), # 応答完了時のタイムスタンプ
            "is_bot": True
        }
        channel_data[message.channel.id]["history"].append(bot_message_data)
        logger.debug(f"履歴追加 (Bot): {bot_message_data['author_name']} - {bot_message_data['content'][:50]}...")

    else:
        # テキストもエラーもない場合 (通常考えにくい)
        final_embed.description = "応答を生成できませんでした。"
        final_embed.color = discord.Color.orange()
        logger.warning("応答テキストもエラーメッセージも取得できませんでした。")

    try:
        await reply_message.edit(embed=final_embed)
    except discord.HTTPException as e:
        logger.error(f"最終メッセージの編集に失敗: {e}")
        # 編集失敗した場合、エラーメッセージを新規送信する試み
        try:
            await message.channel.send(f"エラー: 応答の最終表示に失敗しました。({e.status})")
        except discord.HTTPException as send_e:
             logger.error(f"最終メッセージ編集失敗後のエラー通知送信にも失敗: {send_e}")

# --- スラッシュコマンド ---

async def model_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    """/modelコマンドのmodel引数で、入力中にモデル名の候補を提示する"""
    models = await get_available_models()
    choices = [
        app_commands.Choice(name=model, value=model)
        for model in models if current.lower() in model.lower()
    ]
    return choices[:25]

@bot.tree.command(name="model", description="使用するAIモデルを変更します。")
@app_commands.describe(
    model="利用可能なモデル名を選択してください。",
    system_mode="Trueにすると、'System prompt.txt'ファイルからシステムプロンプトを読み込みます。"
)
@app_commands.autocomplete(model=model_autocomplete)
async def select_model(interaction: discord.Interaction, model: str, system_mode: bool = False):
    """スラッシュコマンド /model の実装"""
    global active_model
    if interaction.channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    available_models = await get_available_models()
    if model not in available_models:
        await interaction.response.send_message(
            f"エラー: モデル '{model}' は利用できません。\n`/models` コマンドで利用可能なモデルを確認してください。",
            ephemeral=True
        )
        return

    active_model = model
    response_message = f"✅ アクティブなAIモデルを **{model}** に変更しました。"

    if system_mode:
        loaded_prompt = await load_system_prompt_from_file()
        if loaded_prompt:
            system_prompts[model] = loaded_prompt
            response_message += f"\n📄 `System prompt.txt` からシステムプロンプトを設定しました。"
            logger.info(f"モデル '{model}' にファイルからシステムプロンプトを設定しました。")
        else:
            response_message += f"\n⚠️ `System prompt.txt` の読み込みに失敗しました。詳細はBOTのログを確認してください。"
            if model in system_prompts:
                del system_prompts[model]
                response_message += "\n🗑️ このモデルの既存のカスタムシステムプロンプトはクリアされました。"
                logger.info(f"モデル '{model}' のカスタムシステムプロンプトを削除しました (ファイル読み込み失敗のため)。")
    else:
        if model in system_prompts:
            del system_prompts[model]
            response_message += "\n🗑️ このモデルのカスタムシステムプロンプトをクリアしました。"
            logger.info(f"モデル '{model}' のカスタムシステムプロンプトをクリアしました。")

    await interaction.response.send_message(response_message, ephemeral=True)

@bot.tree.command(name="clear_prompt", description="現在アクティブなモデルのカスタムシステムプロンプトをクリアします。")
async def clear_prompt(interaction: discord.Interaction):
    """スラッシュコマンド /clear_prompt の実装"""
    if interaction.channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return
    if active_model in system_prompts:
        del system_prompts[active_model]
        logger.info(f"モデル '{active_model}' のカスタムシステムプロンプトをクリアしました。")
        await interaction.response.send_message(
            f"✅ モデル **{active_model}** のカスタムシステムプロンプトをクリアしました。デフォルトのプロンプトが使用されます。",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            f"ℹ️ モデル **{active_model}** にはカスタムシステムプロンプトが設定されていません。",
            ephemeral=True
        )

@bot.tree.command(name="clear_history", description="このチャンネルの会話履歴を全て消去します。")
async def clear_history(interaction: discord.Interaction):
    """スラッシュコマンド /clear_history の実装"""
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    if target_channel_id in channel_data:
        channel_data[target_channel_id]["history"].clear()
        logger.info(f"チャンネルID {target_channel_id} の会話履歴をクリアしました。")
        await interaction.response.send_message(
            "✅ このチャンネルの会話履歴をクリアしました。次回の会話は最初から始まります。",
            ephemeral=True
        )
    else:
        # 通常ここには到達しないはずだが念のため
        logger.warning(f"クリア対象のチャンネルID {target_channel_id} のデータが内部に存在しませんでした。")
        await interaction.response.send_message(
            "ℹ️ クリア対象の会話履歴が見つかりませんでした。",
            ephemeral=True
        )

@bot.tree.command(name="show_history", description="このチャンネルの直近の会話履歴を表示します。")
@app_commands.describe(count="表示する履歴の件数 (デフォルト10, 最大50)")
async def show_history(interaction: discord.Interaction, count: int = 10):
    """スラッシュコマンド /show_history の実装"""
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    history = channel_data[target_channel_id]["history"]

    if not history:
        await interaction.response.send_message("表示できる会話履歴がありません。", ephemeral=True)
        return

    count = max(1, min(count, HISTORY_LIMIT)) # 表示件数を制限
    history_list = list(history)
    start_index = max(0, len(history_list) - count)
    display_history = history_list[start_index:]

    embed = Embed(title=f"直近の会話履歴 ({len(display_history)}件)", color=discord.Color.light_gray())
    history_text = ""
    for i, msg in enumerate(display_history):
        author_str = f"🤖 **Assistant**" if msg["is_bot"] and msg["author_id"] == bot.user.id else f"👤 **{msg['author_name']}**"
        content_short = (msg['content'][:150] + '...') if len(msg['content']) > 150 else msg['content']
        history_text += f"`{start_index + i + 1}`. {author_str}: {content_short}\n"
        # Embed Description の文字数制限に配慮
        if len(history_text) > 3800:
             history_text += "... (以降省略)\n"
             break

    if not history_text: history_text = "履歴内容が空です。"
    embed.description = history_text
    embed.set_footer(text=f"全 {len(history_list)} 件中 {len(display_history)} 件を表示 | 最大保持数: {HISTORY_LIMIT}")
    await interaction.response.send_message(embed=embed, ephemeral=True)


# --- ★新規: パラメータ調整コマンド ---
@bot.tree.command(name="set_param", description="Ollamaの生成パラメータを調整します。")
@app_commands.describe(
    parameter="調整するパラメータ名 (temperature, top_k, top_p)",
    value="設定する値"
)
@app_commands.choices(parameter=[
    app_commands.Choice(name="temperature", value="temperature"),
    app_commands.Choice(name="top_k", value="top_k"),
    app_commands.Choice(name="top_p", value="top_p"),
])
async def set_parameter(interaction: discord.Interaction, parameter: app_commands.Choice[str], value: str):
    """スラッシュコマンド /set_param の実装"""
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    param_name = parameter.value
    current_params = channel_data[target_channel_id]["params"]

    try:
        if param_name == "temperature":
            float_value = float(value)
            if 0.0 <= float_value <= 2.0: # 一般的な範囲
                current_params[param_name] = float_value
            else:
                raise ValueError("Temperatureは0.0から2.0の間で設定してください。")
        elif param_name == "top_k":
            int_value = int(value)
            if int_value >= 0: # 0は無効化を意味する場合もある
                current_params[param_name] = int_value
            else:
                raise ValueError("Top Kは0以上の整数で設定してください。")
        elif param_name == "top_p":
            float_value = float(value)
            if 0.0 <= float_value <= 1.0:
                current_params[param_name] = float_value
            else:
                raise ValueError("Top Pは0.0から1.0の間で設定してください。")
        else:
             await interaction.response.send_message(f"エラー: 不明なパラメータ '{param_name}'", ephemeral=True)
             return

        logger.info(f"チャンネル {target_channel_id} のパラメータ '{param_name}' を '{value}' に設定しました。")
        await interaction.response.send_message(
            f"✅ パラメータ **{param_name}** を **{value}** に設定しました。",
            ephemeral=True
        )

    except ValueError as e:
        logger.warning(f"パラメータ設定エラー ({param_name}={value}): {e}")
        await interaction.response.send_message(f"⚠️ 設定値エラー: {e}", ephemeral=True)
    except Exception as e:
        logger.error(f"パラメータ設定中に予期せぬエラー: {e}", exc_info=True)
        await interaction.response.send_message("❌ パラメータ設定中にエラーが発生しました。", ephemeral=True)

# --- ★新規: 統計情報表示コマンド ---
@bot.tree.command(name="stats", description="直近の応答生成に関する統計情報を表示します。")
async def show_stats(interaction: discord.Interaction):
    """スラッシュコマンド /stats の実装"""
    target_channel_id = interaction.channel_id
    if target_channel_id != CHAT_CHANNEL_ID:
        await interaction.response.send_message("このコマンドは指定されたチャットチャンネルでのみ使用できます。", ephemeral=True)
        return

    stats_deque = channel_data[target_channel_id]["stats"]

    if not stats_deque:
        await interaction.response.send_message("統計情報がまだ記録されていません。", ephemeral=True)
        return

    total_count = len(stats_deque)
    total_duration = 0.0
    total_tokens_generated = 0
    total_tps = 0.0
    valid_tps_count = 0

    for stat in stats_deque:
        # NaN チェック
        if not math.isnan(stat.get("total_duration", float('nan'))):
             total_duration += stat["total_duration"]
        total_tokens_generated += stat.get("total_tokens", 0)
        tps = stat.get("tokens_per_second", float('nan'))
        if not math.isnan(tps) and tps > 0:
            total_tps += tps
            valid_tps_count += 1

    avg_duration = total_duration / total_count if total_count > 0 else 0
    avg_tokens = total_tokens_generated / total_count if total_count > 0 else 0
    avg_tps = total_tps / valid_tps_count if valid_tps_count > 0 else 0

    embed = Embed(title="📊 応答生成統計", color=discord.Color.green())
    embed.description = f"直近 **{total_count}** 回の応答生成に関する平均値です。"
    embed.add_field(name="平均応答時間", value=f"{avg_duration:.2f} 秒", inline=True)
    embed.add_field(name="平均生成トークン数", value=f"{avg_tokens:.1f} トークン", inline=True)
    embed.add_field(name="平均トークン/秒 (TPS)", value=f"{avg_tps:.2f} tok/s", inline=True)

    # 現在のパラメータ設定も表示
    current_params = channel_data[target_channel_id]["params"]
    params_str = ", ".join([f"{k}={v}" for k, v in current_params.items()])
    embed.add_field(name="現在のパラメータ設定", value=f"`{params_str}`", inline=False)

    embed.set_footer(text=f"統計は最大{STATS_HISTORY_MAX}件まで保持されます。")

    await interaction.response.send_message(embed=embed, ephemeral=True)


# --- BOT起動 ---
if __name__ == "__main__":
    if TOKEN is None:
        logger.critical("環境変数 'DISCORD_TOKEN' が設定されていません。BOTを起動できません。")
    elif CHAT_CHANNEL_ID is None:
         logger.critical("環境変数 'CHAT_CHANNEL_ID' が設定されていないか、不正です。BOTを起動できません。")
    else:
        logger.info("Ollama Discord BOTを起動します...")
        logger.info(f"監視対象チャンネルID: {CHAT_CHANNEL_ID}")
        logger.info(f"デフォルトモデル: {DEFAULT_MODEL if DEFAULT_MODEL else '未設定 (起動時に自動選択)'}")
        logger.info(f"会話履歴保持数: {HISTORY_LIMIT}")
        logger.info(f"Ollama API URL: {OLLAMA_API_URL}")
        logger.info(f"システムプロンプトファイルパス: {default_prompt_file_path}")
        try:
            bot.run(TOKEN, log_handler=None)
        except discord.LoginFailure:
            logger.critical("Discordへのログインに失敗しました。トークンが正しいか確認してください。")
        except discord.PrivilegedIntentsRequired:
             logger.critical("Message Content Intent が有効になっていません。Discord Developer PortalでBOTの設定を確認してください。")
        except Exception as e:
            logger.critical(f"BOTの起動中に致命的なエラーが発生しました: {e}", exc_info=True)