# cli/main_reco.py
"""
CLI do Sistema de Recomendação - Leve Agents

Interface de linha de comando para o sistema de recomendação de trilhas educacionais.
Suporta diferentes modos de busca e fontes de dados para máxima flexibilidade.

Funcionalidades:
- Busca híbrida (BM25 + semântica) com blending configurável
- Suporte a múltiplas fontes: API da Leve ou arquivos locais
- Personalização baseada em perfil do usuário (snapshot)
- Output em formato JSON ou texto legível
- Configuração flexível de parâmetros de busca

Fontes de dados:
- --source api: Endpoint /api/trails do backend da Leve
- --source files: Arquivo local files/trails/trails_examples.json

Exemplos de uso:
- python -m cli.main_reco -q "Quero aprender programação do zero"
- python -m cli.main_reco -q "trilhas para iniciantes" --json
- python -m cli.main_reco -q "Como organizar meus estudos?" --source api
- python -m cli.main_reco -q "Quais áreas combinam com meu perfil?" --snapshot-path files/snapshots/ana_001.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace, is_dataclass, asdict
from typing import Optional, Tuple

from reco.config import RecoConfig
from reco.pipeline import run as run_pipeline, sync_trails_to_database
from reco.recommendation_logger import log_recommendation
from reco.data_loader import load_trails as load_trails_file
from schemas.trail_input import TrailInput
from helpers.snapshot_selector import select_profile_snapshot, load_snapshot_from_file
import time
import json
from pathlib import Path


# --------- Argumentos --------- #
def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="reco-cli",
        description="Sistema de Recomendação (CLI) — Leve (P1 híbrida).",
    )

    # Entrada do usuário
    parser.add_argument(
        "-q", "--user-question",
        required=True,
        help="Pergunta/dúvida do jovem (texto livre).",
    )
    parser.add_argument(
        "--snapshot-path",
        default=None,
        help="Caminho do JSON de snapshot (opcional). Se não fornecido, permite seleção interativa.",
    )
    parser.add_argument(
        "--trails-path",
        default="files/trails/trails_sanitized.json",
        help="Caminho do JSON de trilhas (default: files/trails/trails_sanitized.json). Ignorado quando --source=api.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Número máximo de trilhas sugeridas (1..3, default: 3).",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="UUID do usuário (opcional, para personalização futura).",
    )
    parser.add_argument(
        "--contexto-extra",
        default=None,
        help="Contexto adicional livre (ex.: 'iniciante; JavaScript').",
    )

    # Fontes de dados
    parser.add_argument(
        "--source",
        choices=["api", "files"],
        default="api",
        help="Fonte do catálogo de trilhas: 'api' ou 'files' (default: api).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override da base URL do backend (ex.: http://localhost:3030). Útil com --source=api.",
    )

    # Controles da P1 (retrieval/ranking)
    parser.add_argument(
        "--collection",
        choices=["trilhas", "vagas"],
        default="trilhas",
        help="Coleção alvo da recomendação (default: trilhas).",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "bm25", "dense"],
        default="hybrid",
        help="Modo de indexação/ranking: híbrido, apenas BM25, ou apenas denso (default: hybrid).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.60,
        help="Peso do componente denso no blending (0..1). Ex.: 0.60 = 60%% denso, 40%% BM25 (default: 0.60).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Ativa normalização dos escores antes do blending (recomendado para hybrid).",
    )
    parser.add_argument(
        "--threshold-trails",
        type=float,
        default=0.55,
        help="Threshold (limiar) de aceitação para TRILHAS (default: 0.55).",
    )
    parser.add_argument(
        "--threshold-jobs",
        type=float,
        default=0.65,
        help="Threshold (limiar) de aceitação para VAGAS (default: 0.65).",
    )

    # Saída
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime o objeto TrailOutput completo em JSON.",
    )
    parser.add_argument(
        "--sync-embeddings",
        action="store_true",
        help="Sincroniza embeddings com o banco de dados antes da consulta.",
    )

    return parser.parse_args(argv)




# --------- Saída "amigável" --------- #
def _print_pretty(output) -> None:
    from textwrap import indent

    if getattr(output, "status", None) == "ok" and getattr(output, "suggested_trails", None):
        print("\n✨ Sugestões para você")
        for i, s in enumerate(output.suggested_trails, start=1):
            title = getattr(s, "title", None) or getattr(s, "nome", None) or getattr(s, "slug", None) or f"Trilha #{i}"
            public_id = getattr(s, "publicId", None)
            description = getattr(s, "description", None)
            why = getattr(s, "why_match", "") or getattr(s, "motivo", "") or ""
            match_score = getattr(s, "match_score", None)
            
            print(f"\n#{i} — {title}")
            
            # Exibe publicId se disponível
            if public_id:
                print(indent(f"ID: {public_id}", "  "))
            
            # Exibe description se disponível
            if description and description.strip():
                # Trunca description se for muito longa
                desc_text = description.strip()
                if len(desc_text) > 200:
                    desc_text = desc_text[:197] + "..."
                print(indent(f"Descrição: {desc_text}", "  "))
            
            if why:
                print(indent(f"por que indicar: {why}", "  "))
            if isinstance(match_score, (int, float)):
                print(indent(f"match_score: {match_score:.2f}", "  "))
        short = getattr(output, "short_answer", "") or ""
        cta = getattr(output, "cta", "") or ""
        if short:
            print("\n" + short)
        if cta:
            print(f"👉 {cta}\n")
    else:
        # fora_do_escopo ou erro controlado
        msg = getattr(output, "mensagem_padrao", None) or "Não foi possível recomendar trilhas agora."
        short = getattr(output, "short_answer", None) or "Tente reformular sua pergunta com uma palavra-chave."
        cta = getattr(output, "cta", "") or ""
        print("\n" + msg)
        print("\n" + short)
        if cta:
            print(f"👉 {cta}\n")


# --------- Verificação de conexão --------- #
def _check_database_connection(cfg: RecoConfig) -> Tuple[bool, str]:
    """Verifica se a conexão com PostgreSQL está funcionando."""
    try:
        from reco.database.connection import get_connection
        db = get_connection(cfg)
        stats = db.get_embedding_stats()
        total = stats.get('total_embeddings', 0)
        return True, f"✅ Conexão com PostgreSQL OK. Embeddings: {total}"
    except Exception as e:
        return False, f"❌ Erro de conexão com PostgreSQL: {e}"


def _check_embeddings_status(cfg: RecoConfig) -> Tuple[bool, str]:
    """Verifica se há embeddings no banco."""
    try:
        from reco.database.connection import get_connection
        db = get_connection(cfg)
        stats = db.get_embedding_stats()
        
        total = stats.get('total_embeddings', 0)
        if total == 0:
            return False, "Nenhum embedding encontrado no banco"
        
        return True, f"Encontrados {total} embeddings no banco"
    except Exception as e:
        return False, f"Erro ao verificar banco: {e}"


# --------- Auxiliares de configuração --------- #
def _safe_replace_cfg(cfg: RecoConfig, **kwargs) -> RecoConfig:
    """
    Tenta aplicar campos adicionais em RecoConfig sem quebrar quando o campo não existe.
    - Se for dataclass, usa replace() quando possível; senão, faz setattr com guarda.
    - Isso permite evoluir a config aos poucos sem acoplar a CLI.
    """
    if is_dataclass(cfg):
        # Tenta apenas para chaves existentes
        cfg_fields = set(asdict(cfg).keys())
        replace_kwargs = {k: v for k, v in kwargs.items() if k in cfg_fields}
        if replace_kwargs:
            try:
                cfg = replace(cfg, **replace_kwargs)
            except TypeError:
                # Fallback: setattr
                for k, v in replace_kwargs.items():
                    try:
                        setattr(cfg, k, v)
                    except Exception:
                        pass
        # Para chaves inexistentes, ignora silenciosamente
    else:
        # Não é dataclass? Faz setattr best-effort
        for k, v in kwargs.items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


# --------- Main --------- #
def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Carrega snapshot se fornecido, senão permite seleção interativa
    snapshot_path = args.snapshot_path
    if not snapshot_path:
        try:
            # Se não foi fornecido snapshot via argumento, permite seleção interativa
            snapshot_data, snapshot_label = select_profile_snapshot()
            if snapshot_data:
                # Cria um arquivo temporário com os dados selecionados
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
                    snapshot_path = f.name
                print(f"\nSnapshot selecionado: {snapshot_label}")
            else:
                print(f"\nSnapshot selecionado: {snapshot_label}")
        except (EOFError, KeyboardInterrupt):
            # Se não há input disponível (ambiente não-interativo), pula snapshot
            print("\nSnapshot selecionado: nenhum (ambiente não-interativo)")
            snapshot_path = None

    # Monta a entrada validada
    try:
        user_input = TrailInput(
            user_question=args.user_question,
            user_id=args.user_id,
            contexto_extra=args.contexto_extra,
            max_results=args.max_results,
        )
    except Exception as e:
        print(f"[ERRO] Entrada inválida: {e}", file=sys.stderr)
        return 2

    # Configuração base
    cfg = RecoConfig(SOURCE=args.source)
    if args.api_base:
        cfg = _safe_replace_cfg(cfg, TRAILS_API_BASE=args.api_base)

    # Parâmetros P1 — aplicados de forma resiliente (apenas se existirem na RecoConfig)
    # Campos sugeridos na config:
    #   MODE (str: 'hybrid'|'bm25'|'dense')
    #   ALPHA (float: peso do denso no blending)
    #   NORMALIZE (bool)
    #   THRESHOLDS (dict: {'trilhas': float, 'vagas': float})
    #   COLLECTION (str: 'trilhas'|'vagas')
    cfg = _safe_replace_cfg(
        cfg,
        MODE=args.mode,
        ALPHA=args.alpha,
        NORMALIZE=args.normalize,
        THRESHOLDS={"trilhas": args.threshold_trails, "vagas": args.threshold_jobs},
        COLLECTION=args.collection,
    )

    # Verificar conexão com PostgreSQL se usando índice persistente
    if cfg.USE_PERSISTENT_INDEX:
        print("Verificando conexão com PostgreSQL...")
        conn_ok, conn_msg = _check_database_connection(cfg)
        print(conn_msg)
        
        if not conn_ok:
            print("Dica: Verifique se o PostgreSQL está rodando e as variáveis de ambiente estão corretas.")
            return 1
        
        # Verificar se há embeddings no banco
        emb_ok, emb_msg = _check_embeddings_status(cfg)
        if not emb_ok:
            print(f"{emb_msg}")
            if not args.sync_embeddings:
                print("Execute com --sync-embeddings para sincronizar os dados primeiro.")
                return 1

    # Sincronização de embeddings se solicitada
    if args.sync_embeddings:
        print("Sincronizando embeddings com o banco...")
        try:
            # Carregar trilhas para sincronização
            if cfg.SOURCE == "api":
                from reco.data_loader_api import fetch_trails as fetch_trails_api
                raw_trails = fetch_trails_api(cfg)
            else:
                raw_trails = load_trails_file(args.trails_path)
            
            # Normalizar trilhas
            from reco.normalizer import to_candidates, dedupe_by_public_id, filter_by_status
            candidates_all = to_candidates(raw_trails)
            candidates_all = dedupe_by_public_id(candidates_all)
            candidates = filter_by_status(candidates_all, cfg.ALLOWED_STATUS)
            
            # Sincronizar
            sync_stats = sync_trails_to_database(candidates, cfg)
            print(f"✅ Sincronização concluída: {sync_stats}")
            
        except Exception as e:
            print(f"❌ Erro na sincronização: {e}", file=sys.stderr)
            return 1

    # Execução com logging
    start_time = time.time()
    success = True
    error_message = None
    
    try:
        output = run_pipeline(
            user_input=user_input,
            snapshot_path=snapshot_path,
            trails_path=args.trails_path,  # ignorado quando SOURCE='api'
            cfg=cfg,
        )
    except ConnectionError as e:
        success = False
        error_message = str(e)
        print(f"❌ Erro de conexão com PostgreSQL: {e}", file=sys.stderr)
        print("Dica: Verifique se o banco está rodando e execute --sync-embeddings primeiro.")
        return 1
    except Exception as e:
        success = False
        error_message = str(e)
        print(f"[ERRO] Falha ao executar o recomendador: {e}", file=sys.stderr)
        print(f"[DEBUG] Tipo do erro: {type(e).__name__}", file=sys.stderr)
        import traceback
        print(f"[DEBUG] Traceback completo:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    
    execution_time_ms = (time.time() - start_time) * 1000
    
    # Log da recomendação
    try:
        # Carrega perfil do usuário se disponível
        user_profile = {}
        if args.snapshot_path and Path(args.snapshot_path).exists():
            with open(args.snapshot_path, 'r', encoding='utf-8') as f:
                user_profile = json.load(f)
        
        # Converte output para formato de logging
        results = []
        if hasattr(output, 'suggested_trails') and output.suggested_trails:
            for trail in output.suggested_trails:
                result = {
                    'publicId': str(getattr(trail, 'publicId', '')),
                    'title': getattr(trail, 'title', ''),
                    'score': getattr(trail, 'match_score', 0.0),
                    'explanation': getattr(trail, 'why_match', ''),
                    'metadata': {
                        'difficulty': getattr(trail, 'difficulty', ''),
                        'tags': getattr(trail, 'tags', []),
                        'status': getattr(trail, 'status', '')
                    }
                }
                results.append(result)
        
        # Registra no logger
        session_id = log_recommendation(
            user_profile=user_profile,
            query_text=args.user_question,
            results=results,
            execution_time_ms=execution_time_ms,
            model_name=cfg.EMBEDDING_MODEL,
            num_recommendations=args.max_results,
            filters={
                'collection': getattr(cfg, 'COLLECTION', 'trilhas'),
                'mode': getattr(cfg, 'MODE', 'hybrid'),
                'alpha': getattr(cfg, 'ALPHA', 0.5)
            },
            success=success,
            error_message=error_message
        )
        
        if not getattr(args, "json", False):
            print(f"\nSessão registrada: {session_id}")
            print(f"Tempo de execução: {execution_time_ms:.2f}ms")
    
    except Exception as log_error:
        print(f"[AVISO] Erro ao registrar log: {log_error}", file=sys.stderr)

    # Impressão
    if getattr(args, "json", False):
        # Pydantic v2
        if hasattr(output, "model_dump_json"):
            print(output.model_dump_json(indent=2))
        # Pydantic v1 (compat)
        elif hasattr(output, "json"):
            print(output.json(indent=2, ensure_ascii=False))
        else:
            # Melhor esforço
            try:
                print(json.dumps(output, ensure_ascii=False, indent=2))
            except TypeError:
                print(str(output))
    else:
        _print_pretty(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
