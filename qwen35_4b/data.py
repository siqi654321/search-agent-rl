"""Complete data path: protected SFT traces, safe QA synthesis, strict n8 and core selection."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from urllib.request import Request, urlopen

import pipeline as app
from pipeline import start_service, write_json

HERE, ROOT = app.HERE, app.ROOT


def p(key):
    return app.path(key)


def fingerprint(path):
    """Hash files; directory identity uses a manifest, not a second full corpus read."""
    path = Path(path)
    app.need(path)
    if path.is_file():
        return {'path': str(path.resolve()), 'sha256': app.digest(path)}
    files = [(str(f.relative_to(path)), f.stat().st_size, f.stat().st_mtime_ns)
             for f in sorted(path.rglob('*')) if f.is_file()]
    return {'path': str(path.resolve()), 'directory_manifest_sha256':
            hashlib.sha256(json.dumps(files).encode()).hexdigest()}


def code_hashes():
    files = [HERE/'data.py', HERE/'pipeline.py', HERE/'prepare.py', HERE/'evaluate.py',
             HERE/'reward.py', HERE/'run.sh', HERE/'tools.yaml',
             HERE/'wiki_tools.yaml', HERE/'wiki_retrieval.py', HERE/'retrieval.py',
             *[app.FILES/name for name in sorted(app.INSTALLED_FILES)],
             *sorted((HERE/'data_code').glob('*.py'))]
    return {str(f.relative_to(HERE)): app.digest(f) for f in files}


def cached(label, inputs, outputs, recipe, action):
    """Resume completed stages only after validating inputs, code and output hashes.

    A matching started receipt permits retrying an interrupted *data build*;
    rollout attempts are kept separately, never appended or mixed across runs.
    """
    if app.dry():
        print('[data]', label, '(dry run)', flush=True)
        return action()
    state = p('DATA_ROOT')/'state'
    started, done = state/(label+'.started.json'), state/(label+'.done.json')
    signature = {'inputs': [fingerprint(f) for f in inputs], 'recipe': recipe, 'code': code_hashes()}
    if done.exists():
        receipt = json.loads(done.read_text())
        if receipt['signature'] != signature or receipt['outputs'] != [fingerprint(f) for f in outputs]:
            raise ValueError(f'{label}: cached inputs/code/output changed. Use a NEW DATA_ROOT; nothing overwritten')
        print('[data] verified cache:', label, flush=True)
        return
    if started.exists():
        if json.loads(started.read_text()) != signature:
            raise ValueError(f'{label}: interrupted stage has different inputs/config/code; use a new DATA_ROOT')
    else:
        if any(f.exists() for f in outputs):
            raise ValueError(f'{label}: output exists without a build receipt; use a new DATA_ROOT or skip this stage')
        write_json(started, signature)
    action()
    write_json(done, {'signature': signature, 'outputs': [fingerprint(f) for f in outputs]})


def command(argv, label):
    argv = [str(x) for x in argv]
    print('[data command]', shlex.join(argv), flush=True)
    if app.dry():
        return
    log = p('RUN_ROOT')/'data-logs'/(label+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[data log]', log, flush=True)
    with log.open('a') as handle:
        subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def cli(name, *args):
    return [sys.executable, str(HERE/'data_code'/(name+'.py')), *map(str, args)]


def stamp(report, output, inputs):
    value = json.loads(report.read_text())
    value['qwen35_4b_provenance'] = {'output_sha256': app.digest(output),
        'inputs': [fingerprint(f) for f in inputs], 'data_code': code_hashes()}
    write_json(report, value)


@contextmanager
def env_overlay(values):
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def stopped(state):
    """Only stop recorded processes whose live argv matches exactly."""
    if not state.exists():
        return
    saved = json.loads(state.read_text())
    if app.stop_owned(state):
        for _ in range(60):
            proc = Path('/proc')/str(saved['pid'])/'cmdline'
            if not proc.exists() or not proc.read_bytes():
                return
            time.sleep(1)
        raise ValueError(f'Owned service PID {saved["pid"]} did not stop; inspect it before continuing')
    proc = Path('/proc')/str(saved.get('pid', 0))/'cmdline'
    if proc.exists() and proc.read_bytes():
        raise ValueError(f'Stale/foreign PID record: {state}; no process stopped')


def stop_retrievers():
    if app.dry():
        return
    for port in [os.environ['PRIMARY_PORT'], os.environ['SECONDARY_PORT']]:
        stopped(p('RUN_ROOT')/'services'/(port+'.json'))


def wiki_service():
    port = int(os.environ['PRIMARY_PORT'])
    expected = {'corpus_source_path': str(p('WIKI_CORPUS')), 'index_path': str(p('WIKI_INDEX')),
                'query_encoder_model': str(p('WIKI_ENCODER')), 'server_source_sha256': app.digest(HERE/'wiki_retrieval.py'),
                'service_kind': 'wiki_e5'}
    def ready():
        h = app.service_health(port)
        return bool(h and h.get('success') and all(h.get('runtime', {}).get(k) == v for k,v in expected.items()))
    if not app.dry():
        for key in ['WIKI_CORPUS','WIKI_INDEX','WIKI_ENCODER']:
            app.need(p(key))
        stop_retrievers()
        if app.service_health(port):
            raise ValueError(f'Port {port} is occupied by a foreign service; stop it manually or change PRIMARY_PORT')
    state = p('RUN_ROOT')/'services'/f'{port}.json'
    start_service([sys.executable, HERE/'wiki_retrieval.py', '--index_path', p('WIKI_INDEX'),
                   '--corpus_path', p('WIKI_CORPUS'), '--retriever_name','e5', '--retriever_model',p('WIKI_ENCODER'),
                   '--faiss_gpu','--host','127.0.0.1','--port',str(port),'--topk','50',
                   '--lite_search_snippet_chars','800'], os.environ['RETRIEVAL_GPUS'], state,
                  p('RUN_ROOT')/'services/wiki.log', ready)
    if not app.dry():
        req = Request(f'http://127.0.0.1:{port}/search', data=json.dumps({'queries':['Albert Einstein physicist born'],
                      'topk':10,'frontier_enable':True}).encode(), headers={'Content-Type':'application/json'})
        with urlopen(req, timeout=600) as handle:
            result = json.load(handle)
        if not result.get('success') or 'doc://' not in str(result.get('result')):
            raise ValueError('Wiki search smoke failed; inspect services/wiki.log')


@contextmanager
def teachers():
    devices = os.environ['TEACHER_GPUS'].split(',')
    if app.gpu_count(os.environ['TEACHER_GPUS']) != 8:
        raise ValueError('QA generation uses four TP=2 replicas on eight GPUs')
    ports = [int(os.environ['TEACHER_BASE_PORT'])+i for i in range(4)]
    if not all(1024 <= x <= 65535 for x in ports) or {int(os.environ['PRIMARY_PORT']),int(os.environ['SECONDARY_PORT'])} & set(ports):
        raise ValueError('Teacher ports must be unprivileged and distinct from retrieval ports')
    states = []
    if not app.dry():
        app.hf(p('TEACHER_MODEL'))
        stop_retrievers()  # Free retrieval GPUs before loading four 35B replicas.
    try:
        # Start sequentially to avoid concurrent host-memory spikes during weight loading.
        for i, port in enumerate(ports):
            state = p('RUN_ROOT')/'services'/f'teacher-{port}.json'
            if not app.dry():
                stopped(state)
                if app.service_health(port):
                    raise ValueError(f'Teacher port {port} is occupied; no foreign service reused')
            states.append(state)
            def ready(port=port):
                with urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as f:
                    return str(p('TEACHER_MODEL')) in {m['id'] for m in json.load(f)['data']}
            start_service([sys.executable,'-m','sglang.launch_server','--model-path',p('TEACHER_MODEL'),
                           '--tensor-parallel-size','2','--mem-fraction-static','0.80','--trust-remote-code',
                           '--host','127.0.0.1','--port',str(port)], ','.join(devices[i*2:i*2+2]), state,
                          p('RUN_ROOT')/'services'/f'teacher-{port}.log', ready)
        yield [f'http://127.0.0.1:{port}/v1' for port in ports]
    finally:
        if not app.dry():
            for state in reversed(states):
                stopped(state)


def rollout_config(model, data, kind, rows):
    cfg = app.agent_config(model, data, 'eval')
    prefix = 'actor_rollout_ref.rollout.'
    protected = kind == 'sft'
    n, temperature, top_p = (4,0.6,0.95) if protected else (8,1.0,1.0)
    workers, batch = (min(128,rows), min(512,rows)) if protected else (min(64,rows), min(128,rows))
    cfg.update({'data.val_batch_size':batch,'data.validation_shuffle':False,'data.shuffle':False,
                'data.val_max_samples':-1,'data.dataloader_num_workers':8,
                'actor_rollout_ref.model.trust_remote_code':True,
                prefix+'max_model_len':127000 if protected else 122880,
                prefix+'gpu_memory_utilization':0.40, prefix+'max_num_seqs':workers,
                prefix+'max_num_batched_tokens':32768, prefix+'free_cache_engine':False,
                prefix+'enable_chunked_prefill':True,prefix+'enable_prefix_caching':True,
                prefix+'calculate_log_probs':False,prefix+'agent.num_workers':workers,
                prefix+'multi_turn.max_tool_response_length':4096,
                prefix+'multi_turn.tool_config_path':str(HERE/('wiki_tools.yaml' if protected else 'tools.yaml')),
                prefix+'nnodes':1,prefix+'n_gpus_per_node':8})
    for key, value in {'n':n,'temperature':temperature,'top_p':top_p,'top_k':-1}.items():
        cfg[prefix+key] = value
        cfg[prefix+'val_kwargs.'+key] = value
    cfg[prefix+'val_kwargs.do_sample'] = True
    return cfg


def validate_dump(file, ids, n, protected=False):
    counts = Counter()
    with file.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[str(row.get('query_id'))] += 1
            if protected and row.get('distill_trace_enabled') is not True:
                raise ValueError('SFT rollout is missing opt-in intervention traces')
    if set(counts) != set(map(str,ids)) or any(x != n for x in counts.values()):
        raise ValueError(f'Rollout must contain exactly {len(ids)} questions x {n} trajectories; got {dict(Counter(counts.values()))}')


def rollout(data, stage, model, kind, rows, ids):
    cfg = rollout_config(model, data, kind, rows)
    n = 4 if kind == 'sft' else 8
    out = stage/f'rollouts-n{n}'
    controls = {'BROWSECOMP_DUAL_RETRIEVAL_ENABLE': '0' if kind == 'sft' else '1'}
    if kind == 'sft':
        controls.update({'VERL_TOOL_AGENT_FIRST_TOOL_CALL_BOUNDARY':'1','VERL_TOOL_AGENT_FIRST_TOOL_CALL_MAX_TOKENS':'8192',
            'VERL_TOOL_AGENT_FINAL_ANSWER_BOUNDARY':'1','VERL_TOOL_AGENT_GENERIC_ANSWER_RESCUE':'1',
            'VERL_TOOL_AGENT_DUPLICATE_STREAK_LIMIT':'2','TOOL_AGENT_DUPLICATE_STREAK_LIMIT':'2',
            'VERL_TOOL_AGENT_DISTILL_TRACE':'1','VERL_TOOL_AGENT_DISTILL_TRACE_MAX_EVENTS':'8',
            'VERL_TOOL_AGENT_DISTILL_TRACE_MAX_CHARS':'400000'})
    with env_overlay(controls):
        cfg.update(app.ray_env())
        def action():
            if app.gpu_count(os.environ['TRAIN_GPUS']) != 8:
                raise ValueError('Data rollout recipe uses eight GPUs')
            if not app.dry():
                app.check_installed()
                app.hf(model)
                if out.exists():
                    # A failed attempt is recoverable and not mixed with the next attempt.
                    count = 1
                    while (stage/f'failed-rollouts-n{n}-{count}').exists():
                        count += 1
                    archive = stage/f'failed-rollouts-n{n}-{count}'
                    out.rename(archive)
                    print('[data] retained incomplete attempt:', archive, flush=True)
            if kind == 'sft':
                wiki_service()
            else:
                app.retrieval('safe')
            cfg_run = {**cfg, 'trainer.validation_data_dir':str(out),'trainer.experiment_name':f'data_{kind}'}
            started = time.monotonic()
            try:
                app.launch([sys.executable,str(HERE/'evaluate.py'),'--config-path',str(ROOT/'verl/trainer/config'),
                            '--config-name','ppo_trainer'], cfg_run, os.environ['TRAIN_GPUS'], out,
                           {} if app.dry() else {'source':fingerprint(data)})
                if not app.dry():
                    validate_dump(out/'0.jsonl',ids,n,protected=kind=='sft')
                    write_json(stage/'wall.report.json', {'rollout_wall_seconds':time.monotonic()-started})
            finally:
                if kind == 'sft' and not app.dry():
                    stop_retrievers()
        retrieval_keys = ['WIKI_CORPUS','WIKI_INDEX','WIKI_ENCODER'] if kind == 'sft' else [
            'CORPUS','SAFE_DOCIDS','SEARCH_INDEX','SEARCH_ENCODER','REASONING_INDEX','REASONING_ENCODER']
        cached('rollout-'+stage.name, [data,model,*[p(key) for key in retrieval_keys]],
               [out/'0.jsonl',stage/'wall.report.json'], cfg, action)


def sft_data():
    root = p('DATA_ROOT')/'sft-source'
    cohort, sample = root/'excluded500.parquet', root/'questions.parquet'
    for label, output, rows, seed, exclude in [('sft-exclusion',cohort,500,2026,None),
                                             ('sft-sample',sample,int(os.environ['SFT_ROWS']),4041,cohort)]:
        argv = cli('sample','--input',p('RAW_SFT_DATA'),'--output',output,'--rows',rows,'--seed',seed,
                   '--date',os.environ['SFT_DATE'])
        if exclude is not None:
            argv += ['--exclude-input',str(exclude)]
        cached(label,[p('RAW_SFT_DATA')]+([exclude] if exclude else []),[output],argv,
               lambda argv=argv,label=label: command(argv,label))
    if app.dry():
        ids = []
    else:
        import pyarrow.parquet as pq
        ids = [x['query_id'] for x in pq.read_table(sample)['extra_info'].to_pylist()]
    rollout(sample, root, p('BASE_MODEL'),'sft',int(os.environ['SFT_ROWS']),ids)
    dump, candidates = root/'rollouts-n4/0.jsonl', root/'candidates.parquet'
    argv = cli('traces','--input',dump,'--output',candidates,'--report',root/'candidates.report.json',
               '--schema-version','v2','--min-score','1','--deep-search-min-tool-turns','12',
               '--source-label','qwen35_4b_sft','--final-turn-end-text','<|im_end|>')
    cached('sft-candidates',[dump],[candidates],argv,lambda: command(argv,'sft-candidates'))
    argv = cli('causal','--input',candidates,'--output',p('SFT_DATA'),'--report',p('SFT_REPORT'),
               '--tokenizer',p('BASE_MODEL'),'--max-length','127000','--seed','2026',
               '--min-tool-questions','250','--min-answer-questions','750')
    def build():
        command(argv,'sft-causal')
        if not app.dry():
            stamp(p('SFT_REPORT'),p('SFT_DATA'),[candidates])
    cached('sft-causal',[candidates,p('BASE_MODEL')/'tokenizer.json'],[p('SFT_DATA'),p('SFT_REPORT')],argv,build)


def qa_data():
    root = p('DATA_ROOT')/'qa'
    capacity, anchors, chains = root/'capacity', root/'anchors', root/'chains'
    exclusion = ['--benchmark-jsonl',p('BENCHMARK_JSONL'),'--qrel-golds',p('QREL_GOLDS'),
                 '--qrel-evidence',p('QREL_EVIDENCE')]
    argv = cli('corpus','--corpus',p('CORPUS'),*exclusion,'--output-dir',capacity,
               '--body-target-policy','report','--fail-on-gate')
    def audit():
        command(argv,'qa-firewall')
        if not app.dry():
            p('SAFE_DOCIDS').parent.mkdir(parents=True, exist_ok=True)
            p('SAFE_DOCIDS').write_bytes((capacity/'safe_docids.txt').read_bytes())
    cached('qa-firewall',[p(k) for k in ['CORPUS','BENCHMARK_JSONL','QREL_GOLDS','QREL_EVIDENCE']],
           [p('SAFE_DOCIDS'),capacity/'anchor_candidates.jsonl',capacity/'capacity.report.json'],argv,audit)
    argv = cli('anchors','--corpus',p('CORPUS'),*exclusion,'--safe-docids',p('SAFE_DOCIDS'),
               '--anchors',capacity/'anchor_candidates.jsonl','--output-dir',anchors,'--anchor-sample-size','4500')
    cached('qa-anchors',[p('CORPUS'),p('SAFE_DOCIDS'),capacity/'anchor_candidates.jsonl',p('BENCHMARK_JSONL'),
                        p('QREL_GOLDS'),p('QREL_EVIDENCE')],[anchors/'grounded_anchor_bundles.jsonl'],argv,
           lambda: command(argv,'qa-anchors'))
    argv = cli('chains','--corpus',p('CORPUS'),'--safe-docids',p('SAFE_DOCIDS'),
               '--anchor-bundles',anchors/'grounded_anchor_bundles.jsonl','--output-dir',chains,
               '--min-chains-per-bundle','2','--max-chains-per-bundle','3','--max-distractors-per-chain','2',
               '--max-bridge-chars','2400','--required-output-bundles','64','--parquet-batch-size','256','--seed','2026')
    bundle = chains/'grounded_chain_bundles.jsonl'
    cached('qa-chains',[p('CORPUS'),p('SAFE_DOCIDS'),anchors/'grounded_anchor_bundles.jsonl'],[bundle],argv,
           lambda: command(argv,'qa-chains'))
    generated, clean = root/'generated', p('QA_DATA').parent
    if p('QA_DATA').name != 'questions.parquet' or clean == generated:
        raise ValueError('QA_DATA must be a questions.parquet path distinct from qa/generated')
    def generate_and_clean():
        with teachers() as urls:
            url_args = [v for url in urls for v in ('--base-url',url)]
            for phase,variants,cap,retries in [('initial',20,3,3),('expanded',32,5,1)]:
                command(cli('generate','--bundles',bundle,'--output-dir',generated,*url_args,
                    '--model',p('TEACHER_MODEL'),'--answer-mode','terminal_atomic_fact','--unique-answers',
                    '--variants-per-bundle',variants,'--max-accepted-per-bundle',cap,'--constraint-pattern','5',
                    '--max-questions','-1','--output-question-limit','8000','--seed','2026',
                    '--concurrency','32','--batch-size','64','--temperature','0.35','--max-tokens','1792',
                    '--semantic-retry-passes',retries,'--min-question-words','70','--max-question-words','140',
                    '--max-verbatim-source-run','7','--min-generated-questions','1','--target-behavior-matched-groups','400',
                    '--assumed-behavior-match-rate','0.08','--assumed-clean-question-rate','0.06','--rollout-plan','full_n8',
                    '--date',os.environ['QA_DATE'],'--prepare-script',HERE/'prepare.py'), 'qa-'+phase)
            command(cli('semantic','--input',generated/'questions.jsonl','--output-dir',clean,*url_args,
                        '--model',p('TEACHER_MODEL'),'--minimum-clean-questions','1','--temperature','0.05',
                        '--date',os.environ['QA_DATE'],'--prepare-script',HERE/'prepare.py'),'qa-clean')
        if not app.dry():
            import pyarrow.parquet as pq
            count = pq.read_metadata(p('QA_DATA')).num_rows
            if count == 0:
                raise ValueError('No clean questions generated; inspect the semantic report')
            write_json(clean/'capacity.report.json', {'questions':count,'target_rl_groups':400,
                       'assumed_behavior_match_rate':0.08,'projected_groups':count*0.08,
                       'capacity_passed':count*0.08>=400,'policy':'report_only; does not block rollout'})
    cached('qa-generate-clean',[bundle,p('TEACHER_MODEL')],
           [p('QA_DATA'),clean/'semantic_clean.report.json'],
           {'teacher':str(p('TEACHER_MODEL')),'phases':[[20,3,3],[32,5,1]],'date':os.environ['QA_DATE']},generate_and_clean)


def batch_sizes(total, size):
    if total <= 0 or size <= 0:
        raise ValueError('Question count and batch size must be positive')
    # Preserve the original five-part plan: 5076 -> 1000*4 + 1076,
    # not an extra tiny 76-question sixth batch.
    count = max(1,total//size)
    return [size]*(count-1) + [total-size*(count-1)]


def rollout_batches(batch_number=None):
    from data_code import batches
    size = int(os.environ['ROLLOUT_BATCH_ROWS'])
    if size <= 0:
        raise ValueError('ROLLOUT_BATCH_ROWS must be positive')
    if app.dry():
        print('[data] dry plan uses historical 5076 rows; real run reads actual QA_DATA length')
        total = 5076
    else:
        import pyarrow.parquet as pq
        total = pq.read_metadata(p('QA_DATA')).num_rows
    sizes = batch_sizes(total,size)
    count = len(sizes)
    if batch_number is not None and not 1 <= batch_number <= count:
        raise ValueError(f'Batch must be 1..{count}')
    previous = []
    for i in range(1,count+1):
        stage = p('DATA_ROOT')/'batches'/f'batch-{i:03}'
        rows = sizes[i-1]
        report = stage/'sample.report.json'
        if batch_number is None or i == batch_number:
            if app.dry():
                print(f'[sample] {rows} questions without replacement; exclude {len(previous)} complete previous batches')
                sample = {'query_ids':[]}
            else:
                sample = batches.prepare_sample(p('QA_DATA'),stage,rows,2026,previous)
            rollout(stage/'questions.parquet',stage,p('SFT_MODEL'),'qa',rows,sample['query_ids'])
        previous.append(report)


def select_data():
    if app.dry():
        print('[select] audit n8/dual/exact telemetry; valid-F1 variance + exact success + best-tier evidence/length; merge original rows')
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_code import batches
    reports = sorted((p('DATA_ROOT')/'batches').glob('batch-*/sample.report.json'))
    if not reports:
        raise ValueError('No sampled rollout batches; run bash run.sh rollout first')
    inputs = [p('QA_DATA')]
    for report in reports:
        inputs.extend([report,report.parent/'questions.parquet',report.parent/'rollouts-n8/0.jsonl',report.parent/'wall.report.json'])
    def build():
        all_ids, selected_ids, tables, summaries = set(),set(),[],[]
        for report in reports:
            sample = json.loads(report.read_text())
            ids = set(sample['query_ids'])
            if all_ids & ids or sample['source_sha256'] != app.digest(p('QA_DATA')):
                raise ValueError('Batch source changed or question overlap found; no merged pool written')
            all_ids |= ids
            summary = batches.analyze_batch(report.parent,HERE/'reward.py')
            table = pq.read_table(summary['outputs']['parquet'])
            selected_ids |= set(table['query_id'].to_pylist())
            tables.append(table); summaries.append(summary)
        merged = pa.concat_tables(tables)
        if len(selected_ids) != len(merged):
            raise ValueError('Duplicate selected query IDs')
        p('RL_DATA').parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(merged,p('RL_DATA'),compression='zstd')
        write_json(p('RL_REPORT'), {'schema':'qwen35_4b_rl_core_pool_v1','rows':len(merged),
                   'source_questions':pq.read_metadata(p('QA_DATA')).num_rows,'rolled_questions':len(all_ids),
                   'full_source_covered':len(all_ids)==pq.read_metadata(p('QA_DATA')).num_rows,
                   'batch_counts':[s['counts']['length_evidence_core'] for s in summaries],
                   'policy':'preserve source rows; no reward shaping or additional behavior/bundle gate',
                   'can_fill_training_batch':len(merged)>=128,'target_400_is_a_gate':False})
        stamp(p('RL_REPORT'),p('RL_DATA'),inputs)
        print('[selected]',len(merged),'questions ->',p('RL_DATA'),flush=True)
    cached('rl-selection',inputs,[p('RL_DATA'),p('RL_REPORT')],{'selection':'length_evidence_core'},build)


def eval_data():
    argv = [sys.executable,str(HERE/'prepare.py'),'--input',str(p('RAW_EVAL_DATA')),'--output',str(p('EVAL_DATA')),
            '--tool-prompt-mode','native','--native-tool-profile','search_open_find','--date',os.environ['EVAL_DATE']]
    def prepare():
        command(argv,'eval-prepare')
        if not app.dry():
            app.data_check('eval')
    cached('eval-prepare',[p('RAW_EVAL_DATA')],[p('EVAL_DATA')],argv,prepare)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['sft','qa','rollout','select','eval'])
    parser.add_argument('batch',nargs='?',type=int,help='Optional rollout batch number; default processes the entire pool')
    args = parser.parse_args()
    if args.batch is not None and args.stage != 'rollout':
        parser.error('batch number is only valid for rollout')
    action = {'sft':sft_data,'qa':qa_data,'rollout':lambda:rollout_batches(args.batch),
              'select':select_data,'eval':eval_data}[args.stage]
    if app.dry():
        return action()
    with app.run_lock():
        return action()


if __name__ == '__main__':
    try:
        main()
    except (ValueError,KeyError,FileNotFoundError,subprocess.CalledProcessError) as error:
        raise SystemExit(f'ERROR: {error}')
