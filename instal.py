import subprocess
import sys
import os

def install_libraries():
    # インストールするライブラリのリスト
    libraries = [
        "discord.py",
        "python-dotenv",
        "aiohttp"
    ]
    
    print("ライブラリのインストールを開始します...")
    
    # 各ライブラリをインストール
    for lib in libraries:
        print(f"{lib} をインストール中...")
        try:
            # Windowsとそれ以外のOSでコマンドを分ける
            if os.name == 'nt':  # Windows
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", lib])
            else:  # Linux/macOS
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", lib])
            print(f"{lib} のインストールが完了しました。")
        except subprocess.CalledProcessError:
            print(f"{lib} のインストールに失敗しました。")
            return False
    
    # discord.pyの音声サポートが必要な場合（オプション）
    try:
        print("discord.pyの音声サポートをインストール中...")
        if os.name == 'nt':  # Windows
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "discord.py[voice]"])
        else:  # Linux/macOS
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "discord.py[voice]"])
        print("音声サポートのインストールが完了しました。")
    except subprocess.CalledProcessError:
        print("音声サポートのインストールに失敗しました。")
    
    # aiohttpの高速化オプション（推奨）
    try:
        print("aiohttpの高速化オプションをインストール中...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiodns"])
        print("高速化オプションのインストールが完了しました。")
    except subprocess.CalledProcessError:
        print("高速化オプションのインストールに失敗しました。")
    
    print("すべてのライブラリのインストールが完了しました。")
    return True

if __name__ == "__main__":
    install_libraries()
