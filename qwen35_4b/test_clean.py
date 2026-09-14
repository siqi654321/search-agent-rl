"""CPU checks. All installation/mutation fixtures live in temporary clones."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline as app
import reward
ROOT = app.ROOT


class PackageTests(unittest.TestCase):
    def test_syntax(self):
        for source in HERE.rglob('*.py'):
            # Complete VERL sources include existing Python 3.10 match syntax;
            # test_real_baseline_install_idempotence_and_hashes checks those
            # with the explicit baseline fallback on this CPU-only Python 3.9.
            if (HERE/'files') in source.parents:
                continue
            ast.parse(source.read_text(), filename=source.name)
        for source in HERE.glob('*.sh'):
            subprocess.run(['bash', '-n', str(source)], check=True)

    def test_no_private_paths(self):
        private = re.compile(r'/(?:Users|home|mnt)/[\w-]+|/opt/[\w-]+/|Trial [0-9]{6,}|trial-[0-9]{6,}')
        for source in HERE.rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and '.pytest_cache' not in source.parts:
                self.assertFalse(private.search(source.read_text()), source.name)

    def test_copy_scope_and_digests(self):
        self.assertEqual(app.payload_hashes(), app.INSTALLED_FILES)
        targets = list((HERE/'files').rglob('*.py'))
        self.assertEqual({str(p.relative_to(HERE/'files')) for p in targets}, set(app.INSTALLED_FILES))
        self.assertEqual(len(targets), 18)
        self.assertFalse(any('/trainer/ppo/' in name or '/checkpoint/' in name for name in app.INSTALLED_FILES))
        self.assertFalse(list(HERE.rglob('*.patch')))
        for source in targets:
            self.assertEqual(source.name, Path(str(source.relative_to(HERE/'files'))).name)

    def test_rl_recipe(self):
        c = app.agent_config(Path('/model'), Path('/data'), 'rl')
        expected = {'data.train_batch_size':128, '++data.gen_batch_size':128,
                    'actor_rollout_ref.rollout.n':8, 'actor_rollout_ref.actor.ppo_mini_batch_size':32,
                    'actor_rollout_ref.actor.optim.lr':5e-7, 'actor_rollout_ref.actor.kl_loss_coef':5e-4,
                    'actor_rollout_ref.rollout.temperature':1.0, 'actor_rollout_ref.rollout.top_p':1.0,
                    'actor_rollout_ref.rollout.top_k':-1, 'actor_rollout_ref.rollout.multi_turn.max_tool_response_length':4096,
                    'trainer.total_training_steps':20, 'trainer.save_freq':5, 'trainer.max_actor_ckpt_to_keep':4,
                    'trainer.test_freq':-1, 'trainer.val_before_train':False, '++algorithm.filter_groups.enable':False}
        for key, value in expected.items():
            self.assertEqual(c[key], value, key)
        self.assertFalse(any('pin' in k or 'online_mix' in k for k in c))
        self.assertEqual(c['reward.custom_reward_function.path'], str(HERE/'reward.py'))

    def test_eval_recipe(self):
        c = app.agent_config(Path('/model'), Path('/data'), 'eval')
        for key, value in {'temperature':0.6,'top_p':0.95,'n':1,'top_k':-1}.items():
            self.assertEqual(c['actor_rollout_ref.rollout.'+key], value)
            self.assertEqual(c['actor_rollout_ref.rollout.val_kwargs.'+key], value)
        self.assertEqual(c['actor_rollout_ref.rollout.multi_turn.max_tool_response_length'],16384)

    def test_sft_causal_recipe(self):
        with patch.dict(os.environ, {'BASE_MODEL':'/model','SFT_DATA':'/data'}):
            c = app.sft_config()
        self.assertEqual(c['+data.loss_spans_key'], 'loss_spans_json')
        self.assertEqual(c['data.train_batch_size'],16)
        self.assertEqual(c['trainer.total_training_steps'],112)
        self.assertEqual(c['trainer.save_freq'],28)
        self.assertEqual(c['engine.ulysses_sequence_parallel_size'],4)
        self.assertEqual(c['optim.lr'],2e-7)

    def test_all_dry_runs_write_nothing_and_load_four_distinct_models(self):
        with tempfile.TemporaryDirectory(prefix='qwen35-4b-dry-') as d:
            out = Path(d)/'outputs'
            env = {**os.environ,'DRY_RUN':'1','RUN_ROOT':str(out),'PYTHONDONTWRITEBYTECODE':'1'}
            logs = []
            for script, args in [('sft',[]),('rl',[]),('eval',['20','15','10','5']),
                                 ('eval',['sft']),('eval',['scaffold']),('retrieval',['safe']),('retrieval',['full'])]:
                result = subprocess.run(['bash',str(HERE/'run.sh'),script,*args], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                logs.append(result.stdout)
            self.assertFalse(out.exists())
            for step in [20,15,10,5]:
                self.assertIn(f'/rl/global_step_{step}/actor/huggingface',logs[2])
            self.assertEqual(logs[2].count('[launch]'),4)

    def test_single_run_lock(self):
        with tempfile.TemporaryDirectory(prefix='qwen35-4b-lock-') as d, patch.dict(os.environ,{'RUN_ROOT':d}):
            with app.run_lock():
                with self.assertRaisesRegex(ValueError,'Another stage'):
                    with app.run_lock():
                        self.fail('two owners')
            with app.run_lock():
                pass

    def test_unified_entry_rejects_typos_without_starting_work(self):
        self.assertEqual({p.name for p in HERE.glob('*.sh')},{'install.sh','run.sh'})
        for args in [['rl','ignored'],['all','ignored'],['unknown'],['retrieval','safe','full']]:
            result = subprocess.run(['bash',str(HERE/'run.sh'),*args],capture_output=True,text=True,
                                    env={**os.environ,'DRY_RUN':'1','PYTHONDONTWRITEBYTECODE':'1'})
            self.assertNotEqual(result.returncode,0,result.stdout)
            self.assertNotIn('[launch]',result.stdout)
            self.assertNotIn('[copy]',result.stdout)

    def test_shared_service_start_records_exact_owner_and_waits_for_health(self):
        with tempfile.TemporaryDirectory(prefix='qwen35-4b-service-') as d, patch.dict(os.environ,{'DRY_RUN':'0'}):
            state, log = Path(d)/'service.json',Path(d)/'service.log'
            process = Mock(pid=1234)
            process.poll.return_value = None
            ready = Mock(side_effect=[ValueError('still loading'),False,True])
            with patch.object(app.subprocess,'Popen',return_value=process) as start, patch.object(app.time,'sleep'):
                app.start_service(['python3',HERE/'retrieval.py'],'0,1',state,log,ready,timeout=3,metadata={'mode':'safe'})
            record = json.loads(state.read_text())
            self.assertEqual(record['command'],['python3',str(HERE/'retrieval.py')])
            self.assertEqual(record['pid'],1234)
            self.assertEqual(record['owner'],'qwen35_4b')
            self.assertEqual(record['mode'],'safe')
            self.assertEqual(start.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'0,1')
            process.terminate.assert_not_called()

    def test_shared_service_timeout_retains_pid_record_for_inspection(self):
        with tempfile.TemporaryDirectory(prefix='qwen35-4b-service-') as d, patch.dict(os.environ,{'DRY_RUN':'0'}):
            state, log = Path(d)/'service.json',Path(d)/'service.log'
            process = Mock(pid=1234)
            process.poll.return_value = None
            with patch.object(app.subprocess,'Popen',return_value=process), patch.object(app.time,'sleep'):
                with self.assertRaisesRegex(ValueError,'retained for inspection'):
                    app.start_service(['python3',HERE/'retrieval.py'],'0,1',state,log,lambda: False,timeout=1)
            self.assertTrue(state.exists())
            process.terminate.assert_not_called()

    def test_gpu_layout_rejects_duplicate_or_unsafe_ids(self):
        self.assertEqual(app.gpu_count('0,1,2,3,4,5,6,7'),8)
        for value in ['0,0','0,1;echo','',',1']:
            with self.assertRaises(ValueError):
                app.gpu_count(value)

    def test_service_identity_and_safe_full_separation(self):
        with tempfile.TemporaryDirectory(prefix='qwen35-4b-health-') as d:
            allow = (Path(d)/'safe.txt').resolve(); allow.write_text('1\n2\n')
            env = {'SAFE_DOCIDS':str(allow),'CORPUS':'/corpus','SEARCH_INDEX':'/index',
                   'SEARCH_ENCODER':'/encoder','SECONDARY_PORT':'1250'}
            runtime = {'corpus_source_path':'/corpus','corpus_rows':100195,
                       'query_encoder_model':'/encoder','index_path':'/index',
                       'allowed_docids_path':str(allow),'allowed_docids_count':2,
                       'server_source_sha256':app.digest(HERE/'retrieval.py'),
                       'search_global_enable':True,'dual_retrieval_enable':True,
                       'dual_secondary_rank_url':'http://127.0.0.1:1250/rank'}
            with patch.dict(os.environ,env):
                payload = {'success':True,'runtime':runtime}
                self.assertTrue(app.service_matches(payload,'safe','primary'))
                self.assertFalse(app.service_matches(payload,'full','primary'))
                for key, value in [('allowed_docids_count',1),('index_path','/other'),
                                   ('dual_retrieval_enable',False),('server_source_sha256','wrong')]:
                    wrong = {'success':True,'runtime':{**runtime,key:value}}
                    self.assertFalse(app.service_matches(wrong,'safe','primary'),key)

    def test_never_kill_unowned_or_system_pid(self):
        with tempfile.TemporaryDirectory(prefix='qwen35-4b-pid-') as d, patch.object(app.os,'kill') as kill:
            state = Path(d)/'pid.json'
            self.assertFalse(app.stop_owned(state))
            for pid in [0,1,999999999]:
                state.write_text(json.dumps({'pid':pid,'command':['python','/other.py']}))
                self.assertFalse(app.stop_owned(state))
            kill.assert_not_called()

    def test_frozen_data_schema(self):
        if importlib.util.find_spec('pyarrow') is None:
            self.skipTest('optional pyarrow unavailable')
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(prefix='qwen35-4b-data-') as d:
            source = Path(d)/'data.parquet'
            rows = [{'prompt':[{'role':'system','content':'Native tools'},{'role':'user','content':f'Q{i}'}],
                     'reward_model':{'ground_truth':{'target':'answer'}}, 'data_source':'safe'} for i in range(338)]
            pq.write_table(pa.Table.from_pylist(rows),source)
            with patch.dict(os.environ,{'RL_DATA':str(source), 'FROZEN_DATA':'1'}):
                self.assertEqual(app.data_check('rl')['RL_DATA']['rows'],338)
                pq.write_table(pa.Table.from_pylist(rows[:-1]),source)
                with self.assertRaisesRegex(ValueError,'338'):
                    app.data_check('rl')
                rows[0]['extra_info']={'benchmark_overfit':{'benchmark_contaminated':True}}
                pq.write_table(pa.Table.from_pylist(rows),source)
                with self.assertRaisesRegex(ValueError,'overfit'):
                    app.data_check('rl')


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='qwen35-4b-install-')
        self.addCleanup(self.tmp.cleanup)
        self.workspace = (Path(self.tmp.name)/'workspace').resolve()
        self.repo = self.workspace/'verl'
        self.workspace.mkdir()
        subprocess.run(['git','clone','--no-hardlinks','--quiet',str(ROOT),str(self.repo)],check=True)
        self.bundle = self.workspace/HERE.name
        shutil.copytree(HERE,self.bundle,ignore=shutil.ignore_patterns('__pycache__'))

    def install(self, dry=False):
        return subprocess.run(['bash',str(self.bundle/'install.sh')],capture_output=True,text=True,
                              env={**os.environ,'DRY_RUN':'1' if dry else '0','PYTHONDONTWRITEBYTECODE':'1'})

    def test_full_pipeline_dry_run_on_clean_baseline(self):
        output, data = self.repo/'dry-output',self.repo/'dry-data'
        result = subprocess.run(['bash',str(self.bundle/'run.sh')],capture_output=True,text=True,
                                env={**os.environ,'DRY_RUN':'1','RUN_ROOT':str(output),'DATA_ROOT':str(data),
                                     'PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.count('[launch]'),12)  # SFT n4 + SFT train + 5*n8 + RL + 4 evals.
        self.assertIn('--answer-mode terminal_atomic_fact',result.stdout)
        self.assertIn('[select] audit n8',result.stdout)
        self.assertFalse(output.exists())
        self.assertFalse(data.exists())

    def test_real_baseline_install_idempotence_and_hashes(self):
        untouched = ['verl/trainer/ppo/ray_trainer.py','verl/trainer/config/algorithm.py',
                     'verl/utils/checkpoint/checkpoint_manager.py','verl/utils/checkpoint/fsdp_checkpoint_manager.py']
        before = {p:app.digest(self.repo/p) for p in untouched}
        self.assertEqual(self.install(dry=True).returncode,0)
        self.assertFalse((self.repo/'verl/tools/corpus_browser_tool.py').exists())
        result = self.install()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.count('[copy] cp '),18)
        for name, expected in app.INSTALLED_FILES.items():
            self.assertEqual(app.digest(self.repo/name),expected,name)
            # Some unchanged baseline code uses Python 3.10 match syntax.
            # Deployment requires 3.11; allow CPU installer checks on 3.9.
            source = (self.repo/name).read_text()
            try:
                ast.parse(source,filename=name)
            except SyntaxError as error:
                if sys.version_info >= (3,10) or not re.match(r'\s*match\s+.+:',error.text or ''):
                    raise
                baseline = subprocess.check_output(['git','-C',str(self.repo),'show','HEAD:'+name],text=True)
                self.assertIn(error.text.strip(),baseline)
        result = self.install()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('already present',result.stdout)
        self.assertEqual(before,{p:app.digest(self.repo/p) for p in untouched})

    def test_dirty_destination_preserved(self):
        target = self.repo/'verl/trainer/ppo/ray_trainer.py'
        target.write_text(target.read_text()+'\n# user edit\n')
        before = app.digest(target)
        self.assertNotEqual(self.install().returncode,0)
        self.assertEqual(app.digest(target),before)

    def test_untracked_target_collision_preserved(self):
        target = self.repo/'verl/utils/dataset/prompt_completion_sft_dataset.py'
        target.write_text('# user file\n')
        tracked = self.repo/'verl/experimental/agent_loop/agent_loop.py'
        before = app.digest(tracked)
        self.assertNotEqual(self.install().returncode,0)
        self.assertEqual(target.read_text(),'# user file\n')
        self.assertEqual(app.digest(tracked),before)  # All targets checked before the first cp.

    def test_payload_tamper_refused(self):
        target = self.bundle/'files/verl/tools/search_tool.py'
        target.write_text(target.read_text()+'\n')
        self.assertNotEqual(self.install().returncode,0)

    def test_partial_manual_copy_then_install(self):
        for name in list(app.INSTALLED_FILES)[:4]:
            target = self.repo/name
            target.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['cp',str(self.bundle/'files'/name),str(target)],check=True)
        result = self.install()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.count('[copy] cp '),14)
        for name, expected in app.INSTALLED_FILES.items():
            self.assertEqual(app.digest(self.repo/name),expected)

    def test_full_manual_copy_can_be_verified(self):
        subprocess.run(['cp','-R',str(self.bundle/'files/verl')+'/.',str(self.repo/'verl')+'/'],check=True)
        result = subprocess.run([sys.executable,str(self.bundle/'pipeline.py'),'verify'],
                                capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('all 18 installed Python files match',result.stdout)
        self.assertIn('already present',self.install().stdout)

    def test_source_symlink_refused(self):
        target = self.bundle/'files/verl/tools/visit_tool.py'
        outside = Path(self.tmp.name)/'source.py'
        target.rename(outside)
        target.symlink_to(outside)
        self.assertNotEqual(self.install().returncode,0)
        self.assertFalse((self.repo/'verl/tools/corpus_browser_tool.py').exists())

    def test_hard_linked_target_refused(self):
        target = self.repo/'verl/experimental/agent_loop/agent_loop.py'
        outside = Path(self.tmp.name)/'linked.py'
        os.link(target,outside)
        before = app.digest(outside)
        self.assertNotEqual(self.install().returncode,0)
        self.assertEqual(app.digest(outside),before)

    def test_staged_changes_preserved(self):
        target = self.repo/'verl/experimental/agent_loop/agent_loop.py'
        subprocess.run(['cp',str(self.bundle/'files'/target.relative_to(self.repo)),str(target)],check=True)
        subprocess.run(['git','-C',str(self.repo),'add',str(target.relative_to(self.repo))],check=True)
        index_before = subprocess.check_output(['git','-C',str(self.repo),'diff','--cached'])
        self.assertNotEqual(self.install().returncode,0)
        self.assertEqual(index_before,subprocess.check_output(['git','-C',str(self.repo),'diff','--cached']))

    def test_modified_install_refused(self):
        self.assertEqual(self.install().returncode,0)
        target = self.repo/'verl/tools/search_tool.py'
        target.write_text(target.read_text()+'\n# later edit\n')
        result = self.install()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('modified',result.stderr)

    def test_symlink_refused_without_touching_external_target(self):
        outside = Path(self.tmp.name)/'outside.py'; outside.write_text('# preserve\n')
        (self.repo/'verl/tools/corpus_browser_tool.py').symlink_to(outside)
        self.assertNotEqual(self.install().returncode,0)
        self.assertEqual(outside.read_text(),'# preserve\n')


class RewardTests(unittest.TestCase):
    def score(self,text,target='Eiffel Tower'):
        return reward.compute_score(data_source='any_source',solution_str=text,ground_truth={'target':target},extra_info={})

    def test_exact_partial_and_alias(self):
        self.assertEqual(self.score('<answer>The Eiffel Tower</answer>'),1)
        self.assertAlmostEqual(self.score('<answer>Eiffel</answer>'),2/3)
        self.assertEqual(self.score('<answer>Eiffel Tower</answer>',['Tour Eiffel','Eiffel Tower']),1)

    def test_invalid_protocol_fails_closed(self):
        for text in ['no answer','<answer></answer>','<answer>wrong</answer><answer>Eiffel Tower</answer>',
                     '<answer>Eiffel Tower</answer> trailing',
                     '<tool_call>{"name":"search"}</tool_call><answer>Eiffel Tower</answer>']:
            self.assertEqual(self.score(text),0,text)

    def test_user_prompt_answer_example_is_not_a_second_answer(self):
        text = ('<tool_call>{"name":"search"}</tool_call>user\n<tool_response>Eiffel Tower</tool_response>\n'
                'assistant\n<think>done</think>user\nThe search phase is closed by the runtime. '
                'Return <answer></answer>.\nassistant\n<answer>Eiffel Tower</answer>')
        self.assertEqual(self.score(text),1)

    def test_no_reward_mode_or_shaping_leaks_from_environment(self):
        with patch.dict(os.environ,{'SEARCH_REWARD_MODE':'legacy','SEARCH_STRICT_EVIDENCE_COEF_ENABLED':'1'}):
            self.assertEqual(self.score('<answer>Eiffel Tower</answer>'),1)
            self.assertEqual(self.score('<answer>wrong</answer><answer>Eiffel Tower</answer>'),0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
