"""
setup_zotero.py
================
Herkesin kendi Zotero hesabını ECE'ye bağlaması için interaktif kurulum.
Kişisel API anahtarınız bu repoya asla commit edilmez - sadece yerel .env dosyanıza yazılır.
"""

import os
from pathlib import Path

def prompt(msg, default=""):
    val = input(f"{msg} [{default}]: ").strip()
    return val or default

def main():
    print("=== ECE v3 - Zotero Bağlantı Kurulumu ===")
    print("Zotero API anahtarınızı almak için: https://www.zotero.org/settings/keys")
    print("")

    env_path = Path(".env")
    existing = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k,v = line.split("=",1)
                existing[k.strip()] = v.strip()

    lib_id = prompt("ZOTERO_LIBRARY_ID (sayısal ID)", existing.get("ZOTERO_LIBRARY_ID",""))
    api_key = prompt("ZOTERO_API_KEY", existing.get("ZOTERO_API_KEY",""))
    lib_type = prompt("ZOTERO_LIBRARY_TYPE (user/group)", existing.get("ZOTERO_LIBRARY_TYPE","user"))

    local = prompt("Yerel Zotero 7+ kullanmak ister misin? (true/false)", existing.get("ZOTERO_LOCAL","false"))

    lines = []
    # preserve other env vars
    for k in ["NEO4J_URI","NEO4J_USERNAME","NEO4J_PASSWORD","OLLAMA_BASE_URL","OLLAMA_EMBEDDING_MODEL","OLLAMA_MODEL"]:
        if k in existing:
            lines.append(f"{k}={existing[k]}")

    lines += [
        f"ZOTERO_LIBRARY_ID={lib_id}",
        f"ZOTERO_API_KEY={api_key}",
        f"ZOTERO_LIBRARY_TYPE={lib_type}",
        f"ZOTERO_LOCAL={local}",
    ]

    env_path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"\n✓ .env dosyası güncellendi. Bu dosya .gitignore'da olmalı!")
    print("Test: python zotero_connector.py list-collections")

if __name__ == "__main__":
    main()
