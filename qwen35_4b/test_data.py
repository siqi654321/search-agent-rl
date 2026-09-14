"""CPU regressions for the complete data pipeline; no GPU/model downloads."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0,str(HERE))
import data as driver
import pipeline as app
from data_code import generate as generator, semantic as cleaner, batches as batch
from data_code.chains import run as build_chain_bundles
from data_code.causal import CandidateRejected, _exact_action_spans, _materialize_candidate, build
from data_code.traces import build_rows


def _retrieval_outputs(module, monkeypatch):
    """Exercise retained handlers with an in-memory corpus; never load an index/model."""
    docs = {str(i): {'docid': str(i), 'title': f'Document {i}',
                    'text': f'Alpha Project joined Meridian Network. Evidence number {i}.',
                    'contents': f'Document {i}\nAlpha Project joined Meridian Network. Evidence number {i}.',
                    'url': f'https://source{i}.example/article'} for i in range(1,16)}
    corpus = SimpleNamespace(get_doc=docs.get, get_docs=lambda ids: [docs[x] for x in ids if x in docs])
    def search(query_list, num=None, return_score=False):
        items = list(docs.values())[:num]
        return [items for _ in query_list], [[float(len(items)-i) for i in range(len(items))] for _ in query_list]
    settings = dict(retriever=SimpleNamespace(corpus=corpus, batch_search=search),
                    config=SimpleNamespace(retrieval_topk=50), allowed_docids=None,
                    search_global_enable=True, search_global_rank_cutoff=50, search_global_output_topk=10,
                    search_global_rerank_mode='rrf_lexical', search_global_rich_topk=5,
                    search_global_rich_budget=2100, search_global_rich_mode='chunks',
                    search_global_compact_mode='query', search_global_compact_snippet_chars=240,
                    search_global_max_output_chars=15000, search_card_chunk_chars=2100,
                    search_card_overlap_chars=240, search_result_link_label='OpenURL',
                    search_result_open_tool_name='open', browser_max_num_lines=64,
                    dual_retrieval_enable=True, dual_rrf_k=60, dual_primary_weight=1.0,
                    dual_secondary_weight=2.0, dual_primary_rerank_mode='rrf')
    for key,value in settings.items():
        monkeypatch.setattr(module,key,value)
    def secondary(queries, topk):
        assert queries == ['Reasoning: Identify the organization.\nQuery: Alpha project']
        assert topk == 50
        return list(reversed(docs))
    monkeypatch.setattr(module,'_secondary_ranked_docids',secondary)
    def query(excluded=()):
        request = module.LiteSearchRequest(queries=['Alpha project'], topk=50,
            reasoning='Identify the organization.', frontier_enable=True, frontier_exclude_docids=list(excluded))
        return module.search_endpoint(request)
    full = query()
    monkeypatch.setattr(module,'allowed_docids',set(list(docs)[:13]))
    first = query()
    second = query(first['frontier']['selected_docids'])
    exhausted = query(list(docs)[:13])
    results = [full,first,second,exhausted,
        module.rank_search_endpoint(module.RankSearchRequest(queries=['Alpha project'],topk=50)),
        module.open_document_endpoint(module.OpenDocumentRequest(cursor='doc://1')),
        module.find_document_endpoint(module.FindDocumentRequest(cursor='doc://1',pattern='Meridian')),
        module.open_document_endpoint(module.OpenDocumentRequest(cursor='doc://14')),
        module.find_document_endpoint(module.FindDocumentRequest(cursor='doc://14',pattern='Meridian'))]
    def unavailable(**kwargs):
        raise RuntimeError('offline fixture')
    monkeypatch.setattr(module,'_secondary_ranked_docids',unavailable)
    results.append(query())
    return [{k:v for k,v in row.items() if k not in {'elapsed_ms','timing_ms'}} for row in results]


def test_atomic_generation_id_and_prompt_golden():
    specs,_ = generator.build_specs([_chain_bundle()], variants_per_bundle=32,
        constraint_pattern=[5], seed=2026, max_questions=-1, answer_mode='terminal_atomic_fact')
    payload = json.dumps([(s.candidate_id,generator.build_prompt(s)) for s in specs],ensure_ascii=False)
    # Captured before removing the unused answer-mode branches.
    assert hashlib.sha256(payload.encode()).hexdigest() == '8640dee0d27d424624e0ac666ce39d7d8f26975e599da7fe3daa7767b0eb3998'
    with pytest.raises(ValueError,match='terminal_atomic_fact'):
        generator.build_specs([_chain_bundle()], variants_per_bundle=1, constraint_pattern=[5],
                              seed=2026,max_questions=-1,answer_mode='anchor')


def test_retrieval_outputs_match_pre_simplification_golden(monkeypatch):
    import retrieval
    outputs = _retrieval_outputs(retrieval,monkeypatch)
    payload = json.dumps(outputs,sort_keys=True,ensure_ascii=False)
    assert hashlib.sha256(payload.encode()).hexdigest() == 'c8e6e2cc23ca6804f9409b635ea8579eec5a88a7ec80f38ee7eafab3e0259f90'
    assert outputs[0]['frontier']['selected_docids'][:2] == ['15','14']
    assert not {'14','15'} & set(outputs[1]['frontier']['selected_docids'])
    assert outputs[2]['frontier']['selected_docids'] == ['3','2','1']
    assert outputs[3]['frontier']['exhausted'] and outputs[3]['frontier']['selected_count'] == 0
    assert outputs[5]['success'] and outputs[6]['success']
    assert not outputs[7]['success'] and not outputs[8]['success']
    assert outputs[-1]['fusion']['applied'] is False and outputs[-1]['fusion']['fallback_error']


def test_retrieval_retains_only_recipe_endpoints():
    tree = ast.parse((HERE/'retrieval.py').read_text())
    paths = [decorator.args[0].value for node in tree.body if isinstance(node,ast.FunctionDef)
             for decorator in node.decorator_list if isinstance(decorator,ast.Call)
             and isinstance(decorator.func,ast.Attribute) and isinstance(decorator.func.value,ast.Name)
             and decorator.func.value.id == 'app']
    assert set(paths) == {'/health','/rank','/search','/open','/find'}
    assert len(paths) == 5
    assert not any(isinstance(node,ast.ClassDef) and node.name == 'BM25Retriever' for node in tree.body)


def test_all_data_commands_are_local_and_dry_runs_write_nothing(tmp_path):
    env = {**os.environ,'DRY_RUN':'1','RUN_ROOT':str(tmp_path/'runs'),
           'DATA_ROOT':str(tmp_path/'data'),'PYTHONDONTWRITEBYTECODE':'1'}
    logs = {}
    for stage, command in [('sft','sft-data'),('qa','qa'),('rollout','rollout'),('select','select'),('eval','eval-data')]:
        result = subprocess.run(['bash',str(HERE/'run.sh'),command],env=env,capture_output=True,text=True)
        assert result.returncode == 0, result.stderr
        logs[stage] = result.stdout
        assert '4B_scaffold_sft_rl/overlay' not in result.stdout
    assert not (tmp_path/'data').exists() and not (tmp_path/'runs').exists()
    assert '--exclude-input' in logs['sft'] and '--rows 2000' in logs['sft']
    assert '--schema-version v2' in logs['sft'] and '--min-tool-questions 250' in logs['sft']
    assert 'VERL_TOOL_AGENT_DISTILL_TRACE="1"' in logs['sft']
    assert logs['qa'].count('--tensor-parallel-size 2') == 4
    assert '--answer-mode terminal_atomic_fact' in logs['qa']
    assert '--variants-per-bundle 20' in logs['qa'] and '--variants-per-bundle 32' in logs['qa']
    for line in logs['qa'].splitlines():
        if '/generate.py ' in line or '/semantic.py ' in line:
            assert '--benchmark-jsonl' not in line and '--qrel-' not in line
    assert logs['rollout'].count('[launch]') == 5
    assert '[sample] 1076 questions' in logs['rollout']


def test_rollout_and_eval_sampling_are_intentionally_different():
    assert driver.batch_sizes(5076,1000) == [1000,1000,1000,1000,1076]
    assert driver.batch_sizes(66,1000) == [66]
    a = driver.rollout_config(Path('/model'),Path('/data'),'qa',1000)
    b = driver.rollout_config(Path('/model'),Path('/data'),'sft',2000)
    e = app.agent_config(Path('/model'),Path('/data'),'eval')
    prefix = 'actor_rollout_ref.rollout.'
    for config,n,t,p in [(a,8,1.0,1.0),(b,4,0.6,0.95),(e,1,0.6,0.95)]:
        assert config[prefix+'n'] == config[prefix+'val_kwargs.n'] == n
        assert config[prefix+'temperature'] == config[prefix+'val_kwargs.temperature'] == t
        assert config[prefix+'top_p'] == config[prefix+'val_kwargs.top_p'] == p
    assert a[prefix+'multi_turn.max_tool_response_length'] == 4096
    assert a[prefix+'max_model_len'] == 122880
    assert b[prefix+'max_model_len'] == 127000
    assert b[prefix+'multi_turn.tool_config_path'].endswith('/wiki_tools.yaml')
    assert a[prefix+'multi_turn.tool_config_path'].endswith('/tools.yaml')
    assert e[prefix+'multi_turn.max_tool_response_length'] == 16384
    assert a['data.val_batch_size'] >= a[prefix+'agent.num_workers']


def test_stage_resume_checks_inputs_outputs_code_and_never_overwrites_foreign_files(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_ROOT',str(tmp_path))
    monkeypatch.setenv('DRY_RUN','0')
    source, output = tmp_path/'source',tmp_path/'output'
    source.write_text('one')
    calls = []
    def action():
        output.write_text('result'); calls.append(1)
    driver.cached('fixture',[source],[output],{'a':1},action)
    driver.cached('fixture',[source],[output],{'a':1},action)
    assert len(calls) == 1
    source.write_text('two')
    with pytest.raises(ValueError,match='changed'):
        driver.cached('fixture',[source],[output],{'a':1},action)
    assert output.read_text() == 'result'
    with pytest.raises(ValueError,match='without a build receipt'):
        driver.cached('foreign',[source],[output],{},action)


def test_fresh_pool_accepts_actual_count_only_with_matching_provenance(tmp_path,monkeypatch):
    source, report = tmp_path/'pool.parquet',tmp_path/'report.json'
    rows = [{'prompt':[{'role':'system','content':'Tools'},{'role':'user','content':f'Q{i}'}],
             'reward_model':{'ground_truth':{'target':'answer'}},'data_source':'safe'} for i in range(200)]
    pq.write_table(pa.Table.from_pylist(rows),source)
    monkeypatch.setenv('RL_DATA',str(source)); monkeypatch.setenv('RL_REPORT',str(report))
    monkeypatch.setenv('FROZEN_DATA','0')
    report.write_text(json.dumps({'qwen35_4b_provenance':{'output_sha256':app.digest(source)}}))
    assert app.data_check('rl')['RL_DATA']['rows'] == 200
    report.write_text('{}')
    with pytest.raises(ValueError,match='provenance'):
        app.data_check('rl')


def test_prepare_native_format_preserves_ids_targets_and_fixed_date(tmp_path):
    raw, out = tmp_path/'raw.parquet',tmp_path/'prepared.parquet'
    pq.write_table(pa.Table.from_pylist([{'query_id':'q1','question':'Which network did this project join?',
                                        'answer':'Meridian Network'}]),raw)
    subprocess.run([sys.executable,str(HERE/'prepare.py'),'--input',str(raw),'--output',str(out),
                    '--tool-prompt-mode','native','--native-tool-profile','search_open_find',
                    '--date','2026-09-04'],check=True)
    row = pq.read_table(out).to_pylist()[0]
    assert row['prompt'][0]['content'].endswith('2026-09-04')
    assert 'Meridian Network' not in str(row['prompt'])
    assert row['extra_info']['query_id'] == 'q1'
    assert row['reward_model']['ground_truth'] == 'Meridian Network'


def test_installed_loop_has_opt_in_trace_and_eval_has_exact_telemetry():
    loop_text = (HERE/'files/verl/experimental/agent_loop/tool_agent_loop.py').read_text()
    assert 'def _record_distill_intervention(' in loop_text
    assert 'VERL_TOOL_AGENT_DISTILL_TRACE' in loop_text
    text = (HERE/'evaluate.py').read_text()
    assert '"response_token_count"' in text and '"num_turns"' in text
    assert '"distill_trace_events"' in text and '"distill_success_turns"' in text


def test_qa_generation_to_semantic_clean_to_native_parquet_with_mock_http(tmp_path,monkeypatch):
    """Exercise real prompt, HTTP payload, caches, validator, cleaner and formatter.

    Only the model HTTP response is mocked; the production stages are executed.
    A test-only zip adapter permits these CPU tests on Python 3.9; GPU runs need 3.11.
    """
    import io
    from data_code import qa_common
    if sys.version_info < (3,10):
        def strict_zip(*args, strict=False):
            values = [list(x) for x in args]
            if strict and len({len(x) for x in values}) > 1:
                raise ValueError('unequal zip lengths')
            return zip(*values)
        for module in [generator,cleaner,qa_common]:
            monkeypatch.setattr(module,'zip',strict_zip,raising=False)
    bundle = _chain_bundle()
    bundle['verification_evidence'][0]['quote'] = (
        'Alpha Project moved its headquarters to Lisbon in 2015 after joining '
        'the Meridian Network under a regional cooperation agreement.')
    question = ('An American engineer born in Portland in 1972 later directed a computing '
        'laboratory and founded a Seattle technology initiative in 2001. A Canadian '
        'designer born in Vancouver in 1968 chaired a design council and presented '
        'that same initiative with the Northern Design Prize in 2011. The initiative '
        'subsequently relocated its headquarters to Lisbon under a regional cooperation '
        'agreement, as recorded in accounts of the relocation. What proper-named network had the initiative joined?')
    response = {'question':question,'answer':'Meridian Network','answer_type':'organization',
        'answer_is_named_entity':True,'single_answer':True,
        'answer_evidence_quote':bundle['verification_evidence'][0]['quote'].replace('Alpha Project','[SHARED ENTITY]'),
        'used_chain_ids':['C1','C2'],'used_verification_ids':['V1'],
        'chain_clues':[{'chain_id':'C1','bridge_description':'American engineer born in Portland',
                       'edge_relation':'founded the initiative in Seattle','grounded_terms':['Portland','Seattle','2001']},
                      {'chain_id':'C2','bridge_description':'Canadian designer born in Vancouver',
                       'edge_relation':'presented the Northern Design Prize','grounded_terms':['Vancouver','Northern Design Prize','2011']}],
        'retrieval_plan':['resolve the engineer','follow founding relation','cross-check designer','retrieve named network']}
    seen = []
    def http(request,timeout):
        payload = json.loads(request.data)
        assert payload['chat_template_kwargs']['enable_thinking'] is False
        assert payload['model'] == 'fixture-teacher'
        seen.append(payload)
        content = json.dumps(response if len(seen)==1 else _audit())
        return io.BytesIO(json.dumps({'choices':[{'message':{'content':content},'finish_reason':'stop'}]}).encode())
    monkeypatch.setattr(qa_common,'urlopen',http)
    source, generated, clean = tmp_path/'bundle.jsonl',tmp_path/'generated',tmp_path/'clean'
    _write_jsonl(source,[bundle])
    generator.main(['--bundles',str(source),'--output-dir',str(generated),
                    '--base-url','http://fixture.invalid/v1','--model','fixture-teacher',
                    '--answer-mode','terminal_atomic_fact','--variants-per-bundle','1','--constraint-pattern','5',
                    '--max-questions','1','--output-question-limit','1','--semantic-retry-passes','0',
                    '--min-generated-questions','1','--prepare-script',str(HERE/'prepare.py'),'--date','2026-09-04'])
    before = (generated/'questions.jsonl').read_bytes()
    cleaner.main(['--input',str(generated/'questions.jsonl'),'--output-dir',str(clean),
                  '--base-url','http://fixture.invalid/v1','--model','fixture-teacher',
                  '--minimum-clean-questions','1','--prepare-script',str(HERE/'prepare.py')])
    assert len(seen)==2
    assert (generated/'questions.jsonl').read_bytes()==before
    row = pq.read_table(clean/'questions.parquet').to_pylist()[0]
    gt = row['reward_model']['ground_truth']
    assert gt['target']=='Meridian Network' and gt['answer_docid']=='e3'
    assert gt['chain_bundle_id']=='chain-1'
    assert 'Meridian Network' not in str(row['prompt'])
    assert row['extra_info']['query_id']==row['query_id']
    policy=json.loads((clean/'semantic_clean.report.json').read_text())['policy']
    assert policy['audit_scope']=='final_request_answer_contract_only'
    assert policy['upstream_chain_factuality_judged'] is False


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

def _anchor_bundle() -> dict:
    return {
        "bundle_id": "flat-1",
        "answer": "Alpha Project",
        "benchmark_answer_overlap": False,
        "qrel_positive_overlap": False,
        "evidence": [
            {
                "docid": "e1",
                "domain": "edge-one.example",
                "title": "Regional technology history",
                "quote": "Alpha Project was founded by Dana Reed in Seattle in 2001.",
                "score": 12,
                "signatures": {"other_entities": ["Dana Reed"], "years": ["2001"]},
            },
            {
                "docid": "e2",
                "domain": "edge-two.example",
                "title": "Northern Design Prize archive",
                "quote": "Mira Chen presented the Northern Design Prize to Alpha Project in 2011.",
                "score": 11,
                "signatures": {
                    "other_entities": ["Mira Chen", "Northern Design Prize"],
                    "years": ["2011"],
                },
            },
            {
                "docid": "e3",
                "domain": "edge-three.example",
                "title": "Lisbon organizations",
                "quote": "Alpha Project moved its headquarters to Lisbon in 2015.",
                "score": 10,
                "signatures": {"other_entities": ["Lisbon"], "years": ["2015"]},
            },
        ],
    }

def test_builds_two_independent_bridge_chains(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    safe = tmp_path / "safe.txt"
    bundles = tmp_path / "anchor-bundles.jsonl"
    output = tmp_path / "chains"
    _write_jsonl(
        corpus,
        [
            {
                "docid": "e1",
                "title": "Regional technology history",
                "url": "https://edge-one.example/a",
                "text": "Alpha Project was founded by Dana Reed in Seattle in 2001.",
            },
            {
                "docid": "e2",
                "title": "Northern Design Prize archive",
                "url": "https://edge-two.example/b",
                "text": "Mira Chen presented the Northern Design Prize to Alpha Project in 2011.",
            },
            {
                "docid": "e3",
                "title": "Lisbon organizations",
                "url": "https://edge-three.example/c",
                "text": "Alpha Project moved its headquarters to Lisbon in 2015.",
            },
            {
                "docid": "b1",
                "title": "Dana Reed",
                "url": "https://profiles.example/dana",
                "text": (
                    "Dana Reed is an American engineer born in Portland in 1972. "
                    "She studied electrical engineering and later led a public computing laboratory."
                ),
            },
            {
                "docid": "b2",
                "title": "Mira Chen",
                "url": "https://people.example/mira",
                "text": (
                    "Mira Chen is a Canadian designer born in Vancouver in 1968. "
                    "She chaired the Northern Design Council and curated its annual exhibition."
                ),
            },
        ],
    )
    safe.write_text("e1\ne2\ne3\nb1\nb2\n", encoding="utf-8")
    _write_jsonl(bundles, [_anchor_bundle()])
    report = build_chain_bundles(
        argparse.Namespace(
            corpus=corpus,
            corpus_split="train",
            safe_docids=safe,
            anchor_bundles=bundles,
            output_dir=output,
            min_chains_per_bundle=2,
            max_chains_per_bundle=3,
            max_distractors_per_chain=2,
            max_bridge_chars=2400,
            min_bridge_words=10,
            max_target_bundles=-1,
            required_output_bundles=1,
            parquet_batch_size=32,
            progress_every=0,
            max_docs=-1,
            seed=2026,
            fail_on_gate=True,
        )
    )
    assert report["gate"]["passed"] is True
    row = json.loads(
        (output / "grounded_chain_bundles.jsonl").read_text(encoding="utf-8").strip()
    )
    assert row["chain_count"] == 2
    assert {chain["bridge_entity"] for chain in row["chains"]} == {"Dana Reed", "Mira Chen"}
    assert {chain["bridge_doc"]["docid"] for chain in row["chains"]} == {"b1", "b2"}
    assert row["verification_evidence"][0]["docid"] == "e3"

def _chain_bundle() -> dict:
    return {
        "chain_bundle_id": "chain-1",
        "source_bundle_id": "flat-1",
        "answer": "Alpha Project",
        "chains": [
            {
                "chain_id": "C1",
                "bridge_entity": "Dana Reed",
                "bridge_doc": {
                    "docid": "b1",
                    "title": "Dana Reed",
                    "domain": "profiles.example",
                    "excerpt": "Dana Reed was born in Portland in 1972 and led a computing laboratory.",
                },
                "edge_doc": {
                    "docid": "e1",
                    "title": "Regional technology history",
                    "domain": "edge-one.example",
                    "quote": "Alpha Project was founded by Dana Reed in Seattle in 2001.",
                },
                "partial_match_distractors": [],
            },
            {
                "chain_id": "C2",
                "bridge_entity": "Mira Chen",
                "bridge_doc": {
                    "docid": "b2",
                    "title": "Mira Chen",
                    "domain": "people.example",
                    "excerpt": "Mira Chen was born in Vancouver in 1968 and chaired a design council.",
                },
                "edge_doc": {
                    "docid": "e2",
                    "title": "Northern Design Prize archive",
                    "domain": "edge-two.example",
                    "quote": "Mira Chen presented the Northern Design Prize to Alpha Project in 2011.",
                },
                "partial_match_distractors": [],
            },
        ],
        "verification_evidence": [
            {
                "docid": "e3",
                "title": "Lisbon organizations",
                "domain": "edge-three.example",
                "quote": "Alpha Project moved its headquarters to Lisbon in 2015.",
            }
        ],
    }

def test_terminal_atomic_fact_mode_rejects_fragments_meta_language_and_verbatim_clues() -> None:
    bundle = _chain_bundle()
    bundle["verification_evidence"][0]["quote"] = (
        "Alpha Project moved its headquarters to Lisbon in 2015 after joining "
        "the Meridian Network under a regional cooperation agreement."
    )
    specs, counts = generator.build_specs(
        [bundle],
        variants_per_bundle=1,
        constraint_pattern=[5],
        seed=2026,
        max_questions=-1,
        answer_mode="terminal_atomic_fact",
    )
    assert counts["candidate_specs"] == 1
    prompt = generator.build_prompt(specs[0])
    assert "Alpha Project" not in prompt
    assert "page title is intentionally withheld" in prompt

    question = (
        "An American engineer born in Portland in 1972 later directed a computing "
        "laboratory and founded a Seattle technology initiative in 2001. A Canadian "
        "designer born in Vancouver in 1968 chaired a design council and presented "
        "that same initiative with the Northern Design Prize in 2011. The initiative "
        "subsequently relocated its headquarters to Lisbon under a regional cooperation "
        "agreement. What proper-named network had the initiative joined?"
    )
    response = {
        "question": question,
        "answer": "Meridian Network",
        "answer_type": "organization",
        "answer_is_named_entity": True,
        "answer_evidence_quote": (
            "[SHARED ENTITY] moved its headquarters to Lisbon in 2015 after joining "
            "the Meridian Network under a regional cooperation agreement."
        ),
        "single_answer": True,
        "used_chain_ids": ["C1", "C2"],
        "used_verification_ids": ["V1"],
        "chain_clues": [
            {
                "chain_id": "C1",
                "bridge_description": "American engineer born in Portland",
                "edge_relation": "founded the initiative in Seattle",
                "grounded_terms": ["Portland", "Seattle", "2001"],
            },
            {
                "chain_id": "C2",
                "bridge_description": "Canadian designer born in Vancouver",
                "edge_relation": "presented the Northern Design Prize",
                "grounded_terms": ["Vancouver", "Northern Design Prize", "2011"],
            },
        ],
        "retrieval_plan": [
            "resolve the engineer",
            "follow the founding relation",
            "cross-check the designer",
            "retrieve the named network",
        ],
    }
    row, reason = generator.validate_response(
        specs[0], json.dumps(response), min_question_words=50
    )
    assert reason == ""
    assert row is not None
    assert row["schema"] == "browsecomp_safe_atomic_fact_chain_qa_v4"
    assert row["source"] == "browsecomp_safe_atomic_fact_chain_synth_v4"
    assert row["answer"] == "Meridian Network"
    assert row["max_verbatim_source_run"] <= 7

    fragment = dict(response)
    fragment["answer"] = "in both French and English"
    fragment["answer_type"] = "other"
    fragment["answer_evidence_quote"] = "in both French and English"
    row, reason = generator.validate_response(
        specs[0], json.dumps(fragment), min_question_words=50
    )
    assert row is None
    assert reason == "atomic_answer_type"

    meta = dict(response)
    meta["question"] = question.replace(
        "The initiative subsequently", "The terminal passage says the initiative"
    )
    row, reason = generator.validate_response(
        specs[0], json.dumps(meta), min_question_words=50
    )
    assert row is None
    assert reason == "synthetic_meta_language"

    copied = dict(response)
    copied["question"] = question.replace(
        "subsequently relocated its headquarters to Lisbon under",
        "moved its headquarters to Lisbon in 2015 after joining under",
    )
    row, reason = generator.validate_response(
        specs[0], json.dumps(copied), min_question_words=50
    )
    assert row is None
    assert reason == "verbatim_evidence_span"

def test_full_scale_variants_diversify_without_changing_variant_zero_prompt() -> None:
    bundle = _chain_bundle()
    specs, _ = generator.build_specs(
        [bundle],
        variants_per_bundle=2,
        constraint_pattern=[5],
        seed=2026,
        max_questions=-1,
        answer_mode="terminal_atomic_fact",
    )
    prompt_zero = generator.build_prompt(specs[0])
    prompt_one = generator.build_prompt(specs[1])
    assert "Diversity slot" not in prompt_zero
    assert "Diversity slot 1" in prompt_one
    assert "internal generation preference" in prompt_one

class CharacterTokenizer:
    """Tiny fast-tokenizer stand-in: one source character is one token."""

    is_fast = True

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text)))

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

def _tool_row(index: int) -> dict:
    return {
        "sample_id": f"tool-{index}",
        "query_id": f"tool-q-{index}",
        "question": f"tool question {index}",
        "context_text": f"context before tool {index}",
        "chosen_text": (
            "<think>ordinary reasoning is masked</think>\n"
            f"<tool_call><function=search><parameter=query>q{index}</parameter>"
            "</function></tool_call><|im_end|>"
        ),
        "rejected_text": "",
        "intervention_reason": "successful_tool_call",
        "assistant_turn": index + 1,
        "trajectory_is_deep_search": True,
    }

def _answer_row(index: int) -> dict:
    return {
        "sample_id": f"answer-{index}",
        "query_id": f"answer-q-{index}",
        "question": f"answer question {index}",
        "context_text": f"evidence context {index}",
        "chosen_text": (
            "<think>ordinary reasoning is masked</think>\n"
            f"<answer>answer-{index}</answer><|im_end|>"
        ),
        "rejected_text": "",
        "intervention_reason": "successful_final_answer",
        "assistant_turn": 8,
        "trajectory_is_deep_search": True,
    }

def _correction_row(index: int, reason: str) -> dict:
    rejected = f"bad-output-{index}"
    context = f"evidence before correction {index}"
    if reason == "duplicate_loop":
        rejected = (
            "<tool_call><function=search><parameter=query>same query</parameter>"
            "</function></tool_call><|im_end|>"
        )
        context = (
            f"evidence before duplicate {index}<|im_start|>assistant\n"
            + rejected
            + "<|im_start|>tool\nDUPLICATE_CALL_BLOCKED<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return {
        "sample_id": f"correction-{reason}-{index}",
        "query_id": f"correction-q-{reason}-{index}",
        "question": f"correction question {reason} {index}",
        "context_text": context,
        "chosen_text": f"<answer>answer-{index}</answer><|im_end|>",
        "rejected_text": rejected,
        "intervention_reason": reason,
        "assistant_turn": 10,
        "trajectory_is_deep_search": False,
    }

def test_exact_action_spans_exclude_reasoning_and_include_turn_end() -> None:
    completion = (
        "<think>do not train this</think>\n"
        "<tool_call><function=open></function></tool_call><|im_end|>"
    )
    spans = _exact_action_spans(completion, action_kind="tool")

    selected = [completion[start:end] for start, end in spans]
    assert selected[0].startswith("<tool_call>")
    assert selected[1] == "<|im_end|>"
    assert all("do not train this" not in piece for piece in selected)

def test_duplicate_correction_rewinds_before_rejected_action() -> None:
    source = _correction_row(0, "duplicate_loop")
    sample = _materialize_candidate(source)

    assert sample["duplicate_context_rewound"] is True
    assert sample["context_rewind_method"] == "exact_failed_output"
    assert source["rejected_text"] not in sample["context_text"]
    assert "DUPLICATE_CALL_BLOCKED" not in sample["context_text"]
    assert sample["context_text"].endswith("<|im_start|>assistant\n")

def test_duplicate_correction_fails_closed_without_exact_rewind() -> None:
    source = _correction_row(0, "duplicate_loop")
    source["context_text"] = "context without the failed output"

    try:
        _materialize_candidate(source)
    except CandidateRejected as exc:
        assert str(exc) == "duplicate_rewind_exact_match_missing"
    else:
        raise AssertionError("unsafe duplicate correction was accepted")

def test_build_balances_actual_loss_tokens_and_passes_causal_gate(
    tmp_path: Path,
) -> None:
    rows: list[dict] = []
    rows.extend(_tool_row(index) for index in range(24))
    rows.extend(_answer_row(index) for index in range(12))
    for reason in ("answer_pseudo_tool", "missing_answer", "duplicate_loop"):
        rows.extend(_correction_row(index, reason) for index in range(4))
    input_path = tmp_path / "candidates.parquet"
    pd.DataFrame(rows).to_parquet(input_path, index=False)
    output_path = tmp_path / "causal.parquet"
    args = argparse.Namespace(
        input=input_path,
        output=output_path,
        report=tmp_path / "causal.report.json",
        tokenizer="unused-test-tokenizer",
        max_length=127000,
        seed=2026,
        trust_remote_code=False,
        tool_loss_ratio=0.50,
        answer_loss_ratio=0.25,
        correction_loss_ratio=0.25,
        loss_share_tolerance=0.05,
        max_tool_actions_per_question=4,
        min_deep_tool_fraction=1.0,
        min_tool_questions=1,
        min_answer_questions=1,
        min_correction_questions=1,
        min_pseudo_answer_corrections=1,
        min_answer_rescue_corrections=1,
        min_duplicate_loop_corrections=1,
        min_duplicate_rewind_rate=1.0,
    )

    frame, report = build(args, tokenizer=CharacterTokenizer())

    assert report["final_gate"]["passed"] is True
    assert set(frame["causal_category"]) == {"tool", "answer", "correction"}
    assert report["duplicate_rewind"]["materialized_exact_rewinds"] == 4
    for category, requested in report["requested_loss_shares"].items():
        assert abs(report["realized_loss_shares"][category] - requested) <= 0.05
    for _, row in frame.iterrows():
        spans = json.loads(row["loss_spans_json"])
        selected_text = "".join(row["chosen_text"][start:end] for start, end in spans)
        assert "ordinary reasoning is masked" not in selected_text
        assert selected_text.endswith("<|im_end|>")

def _audit(**overrides):
    value = {
        "verdict": "pass",
        "requested_answer_type": "organization",
        "answer_semantic_type": "organization",
        "answer_is_usable_target": True,
        "answer_is_self_contained_for_requested_form": True,
        "answer_fits_final_request": True,
        "request_only_repairable": False,
        "repair_preserves_original_constraints": True,
        "replacement_request": "",
        "reason": "consistent",
    }
    value.update(overrides)
    parsed, reason = cleaner.parse_audit_response(json.dumps(value))
    assert reason == ""
    assert parsed is not None
    return parsed

def _row() -> dict:
    return {
        "query_id": "bcp-safe-atomic-fact-chain-test",
        "question": (
            "A regional engineer established a technical initiative after leading a public "
            "laboratory, while a Canadian designer later connected the same initiative to "
            "an annual design prize. Separate historical evidence records its move to Lisbon "
            "under a cooperation agreement, and another account describes the named network "
            "that accepted the initiative during that transition. Which organization was the "
            "initiative recorded as joining under that agreement?"
        ),
        "question_words": 70,
        "answer": "Meridian Network",
        "answer_type": "organization",
        "answer_evidence_quote": "The initiative joined the Meridian Network under the agreement.",
        "bridge_entities": ["Dana Reed", "Mira Chen"],
    }

def test_type_mismatch_can_repair_only_final_wh_request() -> None:
    row = _row()
    original = row["question"]
    audit = _audit(
        verdict="repair",
        requested_answer_type="organization",
        answer_semantic_type="work",
        answer_fits_final_request=False,
        request_only_repairable=True,
        replacement_request="Which film title is described by that final comparison?",
        reason="the answer is a film, not an organization",
    )
    row["answer"] = "Movie 43"
    repaired, reason = cleaner.apply_request_repair(
        row, audit, min_words=50, max_words=140
    )
    assert reason == ""
    assert repaired is not None
    old_split = cleaner.split_final_request(original)
    new_split = cleaner.split_final_request(repaired["question"])
    assert old_split is not None and new_split is not None
    assert new_split[0] == old_split[0]
    assert repaired["question"].endswith(
        "Which film title is described by that final comparison?"
    )
    assert "Movie 43" not in repaired["question"]
    assert repaired["answer"] == "Movie 43"
    assert repaired["answer_evidence_quote"] == row["answer_evidence_quote"]
    assert repaired["bridge_entities"] == row["bridge_entities"]
    assert repaired["answer_type"] == "work"

def test_explicit_name_fragment_is_not_rejected_as_non_entity() -> None:
    row = _row()
    row["answer"] = "Busby"
    row["answer_evidence_quote"] = "The author surname shown in the entry is Busby."
    audit = _audit(
        requested_answer_type="name_fragment",
        answer_semantic_type="name_fragment",
    )
    assert cleaner.audit_action(row, audit) == (
        "pass",
        "semantic_contract_passed",
    )

def test_terminal_quote_requires_only_literal_answer_presence() -> None:
    row = _row()
    # The short quote intentionally does not restate the long upstream chain.
    row["answer_evidence_quote"] = "The listed organization was Meridian Network."
    assert cleaner.audit_action(row, _audit())[0] == "pass"
    row["answer_evidence_quote"] = "The listed organization was another network."
    assert cleaner.audit_action(row, _audit()) == (
        "reject",
        "deterministic_answer_missing_from_terminal_quote",
    )

def test_repair_verification_rejects_dropped_request_constraints() -> None:
    row = _row()
    row["semantic_clean_status"] = "repaired"
    audit = _audit(repair_preserves_original_constraints=False)
    action, _reason = cleaner.audit_action(row, audit)
    assert action == "reject"

def test_next_batches_exclude_all_previous_samples_and_preserve_rows(tmp_path):
    source = tmp_path / "source.parquet"
    table = pa.table({"query_id": [f"q{i}" for i in range(40)],
                      "prompt": [[{"role": "user", "content": str(i)}] for i in range(40)],
                      "reward_model": [{"ground_truth": {"target": str(i)}} for i in range(40)]})
    pq.write_table(table, source)
    original = source.read_bytes()
    first = batch.prepare_sample(source, tmp_path / "first", 12, 2026)
    first_path = tmp_path / "first/sample.report.json"
    old_files = {p: p.read_bytes() for p in (first_path, tmp_path / "first/questions.parquet")}
    second = batch.prepare_sample(source, tmp_path / "second", 12, 2026, [first_path])
    third = batch.prepare_sample(source, tmp_path / "third", 12, 2026,
                                 [tmp_path / "second/sample.report.json", first_path, first_path])
    baseline = batch.prepare_sample(source, tmp_path / "baseline", 36, 2026)
    assert first["query_ids"] + second["query_ids"] + third["query_ids"] == baseline["query_ids"]
    assert not set(first["query_ids"]) & set(second["query_ids"])
    assert second["excluded_questions"] == 12
    assert second["available_rows_after_exclusion"] == 28
    assert second["remaining_rows_after_sample"] == 16
    assert second["sample_excluded_overlap"] == 0
    assert third["excluded_questions"] == 24 and third["remaining_rows_after_sample"] == 4
    assert source.read_bytes() == original
    for path, content in old_files.items():
        assert path.read_bytes() == content
    kept = pq.read_table(tmp_path / "second/questions.parquet")
    assert kept.schema == table.schema
    assert kept.to_pylist() == table.take(second["source_row_indices"]).to_pylist()

def test_core_does_not_reintroduce_ratio_gap_gates():
    row = dict(rollouts=8, exact_telemetry=True, informative=True, max_base_f1=1,
               best_evidence_supported=True, best_length_matched=True,
               best_search_behavior_matched=False, success_failure_gap_matched=False)
    assert batch.is_length_evidence_core(row)
    for changes in ({"rollouts": 4}, {"exact_telemetry": False}, {"informative": False},
                    {"max_base_f1": 0.8}, {"best_evidence_supported": False}, {"best_length_matched": False}):
        assert not batch.is_length_evidence_core({**row, **changes})

def _completed_batch(tmp_path):
    source = tmp_path / "source.parquet"
    stage = tmp_path / "stage"
    pq.write_table(pa.table({"query_id": ["q1"], "prompt": ["original prompt"]}), source)
    batch.prepare_sample(source, stage, 1, 2026)
    batch.write_json(stage / "wall.report.json", {"rollout_wall_seconds": 100})
    rows = []
    for i in range(8):
        output = (
            '<tool_call>{"name":"open","arguments":{"url":"doc://e5"}}</tool_call>'
            '<tool_response>Alpha Project was founded here.</tool_response>'
            f'<answer>{"Alpha Project" if i == 0 else "Completely Wrong"}</answer>'
        )
        rows.append({"query_id": "q1", "question": "Which project is described by the evidence?",
                     "gts": {"target": "Alpha Project", "answer_docid": "e5", "chain_bundle_id": "b1"},
                     "score": 1.0 if i == 0 else 0.0, "output": output,
                     "num_turns": 70 if i == 0 else 102, "response_token_count": 45000 if i == 0 else 67000,
                     "search_calls": 1, "open_calls": 1, "find_calls": 0, "opened_docids": ["e5"],
                     "search_frontier_events": [{"dual_retrieval_enabled": True, "dual_retrieval_applied": True}]})
    dump = stage / "rollouts-n8/0.jsonl"
    dump.parent.mkdir()
    dump.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return stage, rows

def test_complete_analysis_materializes_core_not_full_profile(tmp_path):
    stage, _ = _completed_batch(tmp_path)
    report = batch.analyze_batch(stage, HERE / "reward.py")
    assert report["counts"]["length_evidence_core"] == 1
    assert report["counts"]["full_success_profile"] == 0
    assert report["mean_strict_reward"] == 0.125
    assert report["trajectories_per_hour"] == 288
    assert report["selected_unique_known_chain_bundles"] == 1
    kept = pq.read_table(Path(report["outputs"]["parquet"]))
    assert kept.column("prompt").to_pylist() == ["original prompt"]
    selected = json.loads(Path(report["outputs"]["selected_jsonl"]).read_text())
    assert selected["selection_mode"] == "length_evidence_core"
    assert not selected["full_success_profile_matched"]

@pytest.mark.parametrize("change,match", [
    ({"score": 0.6}, "reward mismatch"),
    ({"gts": {"target": "Alpha Project"}}, "answer_docid"),
    ({"query_id": "different"}, "runtime audit failed"),
    ({"search_frontier_events": [{"dual_retrieval_enabled": True, "dual_retrieval_applied": False}]}, "runtime audit failed"),
])
def test_bad_runtime_or_reward_never_materializes_training_rows(tmp_path, change, match):
    stage, rows = _completed_batch(tmp_path)
    rows[0].update(change)
    (stage / "rollouts-n8/0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=match):
        batch.analyze_batch(stage, HERE / "reward.py")
    assert not (stage / "analysis/length-evidence-core.parquet").exists()

def test_build_rows_suppresses_ordinary_turns_after_runtime_rescue(tmp_path: Path) -> None:
    source = tmp_path / "rollout.jsonl"
    source.write_text(
        json.dumps(
            {
                "query_id": "rescued",
                "question": "Who?",
                "score": 1.0,
                "finalization_reason": "turn_limit",
                "final_answer_attempts": 1,
                "distill_trace_events": [
                    {
                        "reason": "turn_limit",
                        "context_text": "prompt before the framework instruction",
                        "failed_assistant_output": "<tool_call>repeat search</tool_call>",
                        "corrected_assistant_output": "<answer>Alpha</answer>",
                        "correction_succeeded": True,
                        "assistant_turn": 20,
                    }
                ],
                "distill_success_turns": [
                    {
                        "kind": "tool_call",
                        "context_text": "late looping context",
                        "chosen_assistant_output": "<tool_call>repeat search</tool_call>",
                        "assistant_turn": 20,
                        "tool_execution_succeeded": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows, counts = build_rows([source], min_score=1.0, allow_truncated_context=False)

    assert len(rows) == 1
    assert rows[0]["intervention_reason"] == "turn_limit"
    assert counts["selected_events"] == 1
    assert counts["runtime_assisted_success_rows"] == 1
    assert counts["suppressed_runtime_assisted_success_turns"] == 1
    assert counts["selected_success_turns"] == 0

def test_build_rows_adds_deep_search_and_turn_position_metadata(tmp_path: Path) -> None:
    source = tmp_path / "deep.jsonl"
    turns = []
    for assistant_turn in range(1, 13):
        tool = "search" if assistant_turn < 5 else ("open" if assistant_turn < 9 else "find")
        turns.append(
            {
                "kind": "tool_call",
                "context_text": f"context-{assistant_turn}",
                "chosen_assistant_output": (
                    f'<tool_call><function={tool}><parameter=query>x</parameter></function></tool_call>'
                ),
                "assistant_turn": assistant_turn,
                "tool_execution_succeeded": True,
            }
        )
    turns.append(
        {
            "kind": "final_answer",
            "context_text": "answer-context",
            "chosen_assistant_output": "<answer>Alpha</answer>",
            "assistant_turn": 13,
        }
    )
    source.write_text(
        json.dumps(
            {
                "query_id": "deep",
                "question": "Who?",
                "score": 1.0,
                "search_calls": 4,
                "open_calls": 4,
                "find_calls": 4,
                "distill_success_turns": turns,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows, counts = build_rows(
        [source],
        min_score=1.0,
        allow_truncated_context=False,
        deep_search_min_tool_turns=12,
        schema_version="v2",
    )

    assert len(rows) == 13
    assert counts["natural_deep_search_success_rows"] == 1
    assert {row["trajectory_is_deep_search"] for row in rows} == {True}
    assert {row["trajectory_tool_turns"] for row in rows} == {12}
    assert rows[0]["turn_phase"] == "early"
    assert rows[-1]["turn_phase"] == "late"
    assert rows[0]["tool_name"] == "search"
    assert rows[5]["tool_name"] == "open"
    assert rows[9]["tool_name"] == "find"
    assert rows[-1]["assistant_output_kind"] == "final_answer"
