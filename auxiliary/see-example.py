import pickle
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Inspect specific debate round from pickle data.")
    parser.add_argument('pkl_file', type=str, help='Path to .pkl file')
    
    # Optional args for non-interactive mode
    parser.add_argument('--t', type=int, help='Topology index')
    parser.add_argument('--q', type=int, help='Question index')
    parser.add_argument('--k', type=int, help='Round index')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_file):
        print(f"File not found: {args.pkl_file}")
        sys.exit(1)

    print(f"Loading {args.pkl_file}...")
    try:
        with open(args.pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)
        
    print(f"Loaded data with {len(data)} topologies.")
    
    # Get inputs if not provided
    if args.t is None:
        try:
            t = int(input(f"Enter topology index t (0-{len(data)-1}): "))
        except ValueError:
            print("Invalid integer.")
            sys.exit(1)
    else:
        t = args.t

    if t < 0 or t >= len(data):
        print(f"Error: Topology index {t} out of range (0-{len(data)-1})")
        sys.exit(1)

    topo_data = data[t]
    topo_name = topo_data.get('topology_name', 'Unknown')
    results = topo_data.get('results', [])
    print(f"Selected Topology: {topo_name}")
    print(f"Number of questions: {len(results)}")

    if args.q is None:
        try:
            q = int(input(f"Enter question index q (0-{len(results)-1}): "))
        except ValueError:
            print("Invalid integer.")
            sys.exit(1)
    else:
        q = args.q
        
    if q < 0 or q >= len(results):
        print(f"Error: Question index {q} out of range (0-{len(results)-1})")
        sys.exit(1)
        
    question_data = results[q]
    if question_data is None:
        print(f"Question {q} data is None (failed or skipped).")
        sys.exit(0)
        
    debate_rounds = question_data.get('debate_rounds', [])
    print(f"Question: {question_data.get('question', 'Unknown')}")
    print(f"Correct Answer: {question_data.get('correct_answer', 'Unknown')}")
    print(f"Final Answer: {question_data.get('final_answer', 'Unknown')}")
    print(f"Number of rounds: {len(debate_rounds)}")
    
    if len(debate_rounds) == 0:
        print("No rounds available for this question.")
        sys.exit(0)

    if args.k is None:
        try:
            k = int(input(f"Enter round index k (0-{len(debate_rounds)-1}): "))
        except ValueError:
            print("Invalid integer.")
            sys.exit(1)
    else:
        k = args.k
        
    if k < 0 or k >= len(debate_rounds):
         print(f"Error: Round index {k} out of range (0-{len(debate_rounds)-1})")
         sys.exit(1)
         
    round_data = debate_rounds[k]
    
    print(f"\n--- Output for T={t} ({topo_name}), Q={q}, Round={k} ---")
    
    # Start printing contents
    # round_data is usually a list of agent responses
    if isinstance(round_data, list):
        for agent_resp in round_data:
            print("-" * 50)
            if isinstance(agent_resp, dict):
                agent_id = agent_resp.get('agent_id', 'Unknown')
                is_mal = agent_resp.get('is_malicious', False)
                ans = agent_resp.get('answer', 'N/A')
                
                print(f"Agent ID: {agent_id} {'[MALICIOUS]' if is_mal else ''}")
                print(f"Answer: {ans}")
                
                if 'reason' in agent_resp:
                    print(f"Reason: {agent_resp['reason']}")
                elif 'st_embedding' in agent_resp:
                    print(f"Reason: [Replaced by Embeddings]")
                    st = agent_resp['st_embedding']
                    st_len = len(st) if hasattr(st, '__len__') else 'Unknown'
                    print(f"   ST Embedding shape: {st_len}")
                    
                    tk = agent_resp['tk_embedding']
                    tk_count = len(tk) if hasattr(tk, '__len__') else 'Unknown'
                    tk_dim = len(tk[0]) if tk_count != 'Unknown' and tk_count > 0 else 'Unknown'
                    print(f"   Token Embeddings: {tk_count} tokens x {tk_dim} dim")
                else:
                    print("Reason field missing and no embeddings found.")
                
                # Debug print other keys if needed
                # print(f"Keys: {list(agent_resp.keys())}")
            else:
                print(f"Raw response: {agent_resp}")
    else:
        print(f"Round data structure is not a list: {type(round_data)}")
        print(round_data)

if __name__ == "__main__":
    main()
