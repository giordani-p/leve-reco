"""
Helper para seleção de snapshots de usuários
Centraliza a funcionalidade de seleção interativa de usuários do users_faker.json
"""

import json
from pathlib import Path
from typing import Optional, Tuple


def select_profile_snapshot() -> Tuple[Optional[dict], str]:
    """
    Lista usuários do arquivo users_faker.json e permite seleção pelo nome_preferido.
    Opções:
      - Enter: pular (sem snapshot)
      - Número: carrega o usuário correspondente
      - 'm': colar JSON manualmente
    Retorna (snapshot_dict | None, label_str)
    """
    # Localiza o arquivo users_faker.json
    root_dir = Path(__file__).resolve().parents[1]
    users_file = root_dir / "files" / "snapshots" / "users_faker.json"

    print("\n=== Profile Snapshot (opcional) ===")

    users: list[dict] = []
    try:
        if users_file.exists():
            with open(users_file, 'r', encoding='utf-8') as f:
                users = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Erro ao carregar usuários: {e}")
        users = []

    if users:
        print("Usuários disponíveis:")
        for idx, user in enumerate(users, start=1):
            nome_preferido = user.get('dados_pessoais', {}).get('nome_preferido', f'Usuário #{idx}')
            idade = user.get('dados_pessoais', {}).get('idade', 'N/A')
            localizacao = user.get('dados_pessoais', {}).get('localizacao', 'N/A')
            print(f"[{idx}] {nome_preferido} ({idade}, {localizacao})")
        print("[m] Colar JSON manualmente")
        print("[Enter] Pular (sem snapshot)")
    else:
        print("Nenhum usuário encontrado em users_faker.json.")
        print("[m] Colar JSON manualmente")
        print("[Enter] Pular (sem snapshot)")

    while True:
        choice = input("Selecione uma opção (número, 'm' ou Enter): ").strip().lower()

        if choice == "":
            return None, "nenhum"

        if choice == "m":
            profile_raw = input("Cole o JSON do snapshot e pressione Enter:\n")
            if not profile_raw.strip():
                print("Entrada vazia. Voltando ao menu de snapshot.")
                continue
            try:
                return json.loads(profile_raw), "[manual]"
            except json.JSONDecodeError as e:
                print(f"JSON inválido: {e}. Tente novamente.")
                continue

        if choice.isdigit() and users:
            idx = int(choice)
            if 1 <= idx <= len(users):
                selected_user = users[idx - 1]
                nome_preferido = selected_user.get('dados_pessoais', {}).get('nome_preferido', f'Usuário #{idx}')
                return selected_user, nome_preferido
            else:
                print("Número fora do intervalo. Tente novamente.")
                continue

        print("Opção inválida. Tente novamente.")


def load_snapshot_from_file(snapshot_path: str) -> Optional[dict]:
    """Carrega snapshot de um arquivo JSON."""
    try:
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[ERRO] Falha ao carregar snapshot '{snapshot_path}': {e}")
        return None
