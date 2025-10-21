import pickle, os, json

def summarize_pickle(path: str):
    print(f"\n=== {path} ===")
    if not os.path.exists(path):
        print("missing")
        return
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        summary = {
            'final_phase': data.get('final_phase'),
            'phase_index': data.get('phase_index'),
            'agents': list(data.get('agent_performance', {}).keys()),
            'episode_counts': {k: len(v) for k,v in data.get('agent_performance', {}).items()},
            'phases_defined': [ph.get('name') for ph in data.get('phases', [])],
        }
        perf = {}
        for k, v in data.get('agent_performance', {}).items():
            rec = v[-10:]
            avg = sum(d.get('reward', 0.0) for d in rec)/max(1, len(rec)) if rec else 0.0
            perf[k] = round(avg, 2)
        summary['recent_avg_reward'] = perf
        print(json.dumps(summary, indent=2))
    except Exception as e:
        print('error:', e)

if __name__ == '__main__':
    paths = [
        './examples/multi_agent_racing_results/curriculum_progress.pkl',
        './multi_agent_racing_results/curriculum_progress.pkl'
    ]
    for p in paths:
        summarize_pickle(p)
