import argparse
import subprocess
import sys
import time

def run_training(timeframe, timesteps):
    """
    Run train_agent.py for a specific timeframe.
    """
    cmd = [
        sys.executable, 
        "src/train_agent.py",
        "--timeframe", timeframe,
        "--timesteps", str(timesteps)
    ]
    
    print(f"[{timeframe}] Starting training for {timesteps} steps...")
    try:
        # Run process and wait for completion
        result = subprocess.run(cmd, check=True)
        print(f"[{timeframe}] Training completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{timeframe}] Training failed with error code {e.returncode}.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Orchestrate Multi-Agent Training")
    parser.add_argument("--timeframe", type=str, choices=['5m', '15m', '1h', '4h'], help="Specific timeframe to train. If None, trains all.")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps per agent")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel (Experimental - High Memory Usage)")
    
    args = parser.parse_args()
    
    timeframes = ['5m', '15m', '1h', '4h']
    
    if args.timeframe:
        # Train single agent
        run_training(args.timeframe, args.timesteps)
    else:
        # Train all agents
        if args.parallel:
            print("Starting PARALLEL training for all agents...")
            processes = []
            for tf in timeframes:
                cmd = [sys.executable, "src/train_agent.py", "--timeframe", tf, "--timesteps", str(args.timesteps)]
                p = subprocess.Popen(cmd)
                processes.append((tf, p))
                print(f"[{tf}] Process started (PID: {p.pid})")
            
            # Wait for all
            for tf, p in processes:
                p.wait()
                if p.returncode == 0:
                     print(f"[{tf}] Finished successfully.")
                else:
                     print(f"[{tf}] Failed with code {p.returncode}.")
        else:
            print("Starting SEQUENTIAL training for all agents...")
            for tf in timeframes:
                success = run_training(tf, args.timesteps)
                if not success:
                    print(f"Aborting sequence due to failure in {tf}.")
                    break

if __name__ == "__main__":
    main()
