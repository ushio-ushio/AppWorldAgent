"""Create a redacted, per-Executor-step audit for the frozen Train-15 runs."""
from __future__ import annotations
import json,re
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'.v1_2_b_stage'/'src'))
from agentlab.evaluation.report import parse_report_counts

RUNS=ROOT/'runs'
FIRST=RUNS/'agentlab_v2_lite_train15_siwc_20261123_184018'
RERUN=RUNS/'agentlab_v2_lite_train15_siwc_rerun3_20261123'
OUT=ROOT/'.v1_2_b_stage'/'reports'
IDS=json.loads((OUT/'v2_lite_train15_task_ids.json').read_text(encoding='utf-8'))
API_RE=re.compile(r'apis\.([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\s*\(')

def load(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def code_brief(code):
    if not code:return ''
    calls=[a+'.'+b for a,b in API_RE.findall(code)]
    # Do not emit code or literals: they may contain task credentials or private records.
    if calls:return 'Python: '+', '.join(dict.fromkeys(calls))+f'（{len(code.splitlines())} 行）'
    if 'complete_task' in code:return '提交任务（complete_task）'
    return f'Python 本地处理/辅助代码（{len(code.splitlines())} 行；未识别到直接 API 调用）'

def outcome(tool):
    result=tool.get('tool_result') or {}
    if result.get('validation')=='blocked':
        msg=str(result.get('error') or '')
        if 'Completion check failed' in msg:return '提交被 Completion Guard 拦截；向模型返回缺口提示'
        return '本地安全/协议校验拦截；未执行'
    if result.get('ok') is False:
        msg=str(result.get('result') or result.get('error') or '')
        m=re.search(r'(TypeError|ValueError|KeyError|AttributeError|PermissionError|HTTPError|RequestException|Execution failed)',msg)
        return '执行失败'+('（'+m.group(1)+'）' if m else '')
    if result.get('ok') is True:
        n=len(str(result.get('result') or ''))
        return f'工具返回成功，结果 {n} 字符（内容不展开）'
    return '无明确执行结果'

def make_task(task,run,run_label):
    d=run/task; trace=d/'trace.jsonl'
    if not trace.exists():return [],None
    rows=load(trace); by_step={};
    for row in rows:
        if isinstance(row.get('step'),int):by_step.setdefault(row['step'],[]).append(row)
    summary=json.loads((d/'summary.json').read_text(encoding='utf-8')) if (d/'summary.json').exists() else {}
    steps=[];seen=Counter()
    for number,events in sorted(by_step.items()):
        calls=[x for x in events if x.get('event')=='tool_call']; details=[]; repeat=[]
        obs_refs=[x for x in events if x.get('event')=='observation_ref']
        for x in calls:
            name=x.get('tool_name','?'); raw=x.get('arguments','');
            try: args=json.loads(raw) if isinstance(raw,str) else raw
            except Exception: args={}
            if name=='execute_python':
                executed=x.get('executed_requests') or []
                apis=list(dict.fromkeys(str(q.get('api_name')) for q in executed if q.get('api_name')))
                if not apis:
                    apis=list(dict.fromkeys(api for ref in obs_refs if ref.get('source_tool')=='execute_python'
                                            for api in ref.get('api_names',[])))
                brief=('Python 调用 AppWorld API：'+', '.join(apis)+f'（{len(executed)} 个请求记录）'
                       if apis else code_brief(str(args.get('code') or '')))
            elif name=='search_api': brief='API 搜索：'+str(args.get('query',''))[:130]
            elif name=='get_api_schema': brief='读取 API Schema：'+str(args.get('api_name',''))
            elif name in {'search_history','read_observation'}: brief=f'历史信息检索：{name}'
            else: brief=name+'（参数已省略）'
            key=(name,json.dumps(args,sort_keys=True,ensure_ascii=False))
            seen[key]+=1
            if seen[key]>1:repeat.append(name+' 与此前完全相同')
            result_note=outcome(x)
            if name=='execute_python' and x.get('executed_requests') and (x.get('tool_result') or {}).get('ok') is True:
                result_note=result_note.replace('工具返回成功','Python Shell 正常返回；内部业务 API 的逐项结果未在该字段完整记录')
            details.append(f"{name}: {brief}；{result_note}")
        step_event=next((x for x in events if x.get('event')=='step'),{})
        ctx=next((x for x in events if x.get('event')=='context_view'),{})
        guard=next((x for x in events if x.get('event')=='completion_guard'),None)
        protocol=next((x for x in events if x.get('event')=='protocol_error'),None)
        if guard:details.append(f"Completion Guard: {guard.get('status')} / allowed={guard.get('allowed')}")
        if protocol:details.append('工具协议无效：要求返回有效注册工具调用')
        if not details:details=['未记录工具调用或动作（核对 Trace 事件）']
        usage=step_event.get('usage') or {}
        steps.append({'task_id':task,'run':run_label,'step':number,'actions':details,
                      'tool_calls':len(calls),'input_tokens':usage.get('input_tokens'),
                      'output_tokens':usage.get('output_tokens'),'context_bytes':ctx.get('visible_message_bytes'),
                      'repeat_exact_action':repeat,'protocol_error':bool(protocol),
                      'guard_status':guard.get('status') if guard else None,
                      'stop_reason':step_event.get('stop_reason')})
    return steps,summary

all_steps=[];summaries={}
for task in IDS:
    s,summary=make_task(task,FIRST,'初轮')
    all_steps.extend(s);summaries[task]={'initial':summary}
for task in ('b7a9ee9_3','ce359b5_3','afc0fce_2'):
    s,summary=make_task(task,RERUN,'故障题复测')
    all_steps.extend(s);summaries.setdefault(task,{})['rerun']=summary
(OUT/'v2_lite_train15_step_audit.json').write_text(json.dumps(all_steps,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# V2-Lite 固定 15 题逐回合工作审计','','本报告从冻结任务的 JSONL Trace 重建每个 Executor 回合实际调用的工具、API、返回状态、输入 Token 和显式上下文字节。不会展开代码文本、账户凭据、模型私有推理文本或业务记录；Python 动作仅列 API 名称和代码行数。Planner 请求单独列在每题摘要中，不计入 Executor 步数。三道 SIWC 故障题的复测作为独立回合序列列出，不与首轮合并。','']
for task in IDS:
    info=summaries.get(task,{}); initial=info.get('initial') or {}; rerun=info.get('rerun')
    eval_path=FIRST/task/'evaluation_report.md'; report=eval_path.read_text(encoding='utf-8') if eval_path.exists() else ''
    passed, total = parse_report_counts(report)
    score = f"{passed}/{total}" if passed is not None and total is not None else "未得到完整评分"
    lines += [f"## {task}", '', f"首轮结果：{score}；Executor 步数 {initial.get('steps','未知')}；Planner 请求 {initial.get('planner_calls','未知')}；停止原因 `{initial.get('stop_reason','未知')}`。"]
    steps=[x for x in all_steps if x['task_id']==task and x['run']=='初轮']
    if steps:
        lines+=['','| 回合 | 动作与工具结果 | 输入 Token | 上下文字节 | 重复动作 |','|---:|---|---:|---:|---|']
        for x in steps:
            desc='<br>'.join(x['actions']).replace('|','/');rep='；'.join(x['repeat_exact_action']) or '—'
            lines.append(f"| {x['step']} | {desc} | {x['input_tokens'] if x['input_tokens'] is not None else '—'} | {x['context_bytes'] if x['context_bytes'] is not None else '—'} | {rep} |")
    if rerun:
        lines += ['',f"复测：停止 `{rerun.get('stop_reason','未知')}`；回合 {rerun.get('steps','未知')}；独立 Trace 位于 `{RERUN/task}`。",'']
    else:lines.append('')
lines += ['## 汇总观察','','逐回合机器可读明细：`reports/v2_lite_train15_step_audit.json`。','']
proto=Counter(x['task_id'] for x in all_steps if x['run']=='初轮' and x['protocol_error'])
guards=Counter(x['task_id'] for x in all_steps if x['run']=='初轮' and x['guard_status']=='MISSING')
repeat=Counter(x['task_id'] for x in all_steps if x['run']=='初轮' and x['repeat_exact_action'])
lines += [f"- 初轮有工具协议错误的任务：{dict(proto)}。",f"- 初轮被 Guard 标记 MISSING 的任务/回合数：{dict(guards)}。",f"- Trace 中出现完全相同工具名和参数再次提交的任务：{dict(repeat)}。这些是程序精确匹配候选，不自动等于无效，因为状态核验或恢复可能合理。","- 步数超过 100：本批没有；`6ea6792_2` 达到 100 回合停止。若看到更高的外部步数，应核对是否把 Planner、Graph 重试或复测步数累加进了 Executor 回合。","- 下一步压缩决策应先看每题逐步 API 序列：把 API Search / Schema 查找、只读业务查询、业务写入/确认、错误恢复、提交守卫拦截分开统计；不应仅按 `execute_python` 次数判断冗余。",'']
(OUT/'v2_lite_train15_step_audit.md').write_text('\n'.join(lines),encoding='utf-8')
print('steps=',len(all_steps),'tasks=',len(IDS),'report=',OUT/'v2_lite_train15_step_audit.md')
