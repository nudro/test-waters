"""
Main Chat Interface for CV Agent
Terminal-based chat interface to interact with the vessel tracking agent.
"""

from agent import get_agent, run_query
import sys
import threading
import queue


def print_welcome():
    """Print welcome message."""
    print("\n" + "="*70)
    print("🤖 CV Agent - Vessel Tracking Assistant")
    print("="*70)
    print("I can help you with:")
    print("  • Processing videos for vessel detection and tracking")
    print("  • Calculating motion metrics (velocity, speed, acceleration)")
    print("  • Querying vessel data by track ID")
    print("  • Assigning track IDs via GUI")
    print("\nType 'quit', 'exit', or 'q' to exit")
    print("Type 'help' for more information")
    print("Press Ctrl+C during agent reasoning to cancel and start over")
    print("="*70 + "\n")


def print_help():
    """Print help message."""
    print("\n" + "-"*70)
    print("Available Commands:")
    print("-"*70)
    print("  • Process video: 'Process video /path/to/video.mp4'")
    print("  • Get vessel motion: 'Get motion data for track-id 2'")
    print("  • List vessels: 'List all tracked vessels'")
    print("  • Calculate velocity: 'Calculate velocity of track-id 1'")
    print("  • Get trajectory: 'Show trajectory of track-id 0'")
    print("  • Assign IDs: 'Open GUI to assign track IDs for video.mp4'")
    print("\nExamples:")
    print("  'Process video /media/nudro/USB DISK/all_shipping.mp4'")
    print("  'What is the velocity of track-id 2?'")
    print("  'List all vessels in vessel_tracking_results/video_tracking_results.json'")
    print("-"*70 + "\n")


def format_response(response: str):
    """Format agent response for better readability."""
    # Simple formatting - can be enhanced later
    print("\n" + "─"*70)
    print("Agent:")
    print("─"*70)
    print(response)
    print("─"*70 + "\n")


def chat_loop():
    """Main chat loop."""
    print_welcome()
    
    # Initialize agent
    print("Initializing agent...")
    try:
        agent = get_agent()
        print("✓ Agent ready!\n")
    except Exception as e:
        print(f"✗ Error initializing agent: {e}")
        print("Please check:")
        print("  1. API key file exists at /home/nudro/Documents/keys/maritime-oai")
        print("  2. OpenAI API key is valid")
        print("  3. Required packages are installed (see requirements_agent.txt)")
        sys.exit(1)
    
    # Conversation history (optional - can be enhanced)
    conversation_history = []
    
    # Main loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋\n")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            # Run query through agent with interrupt capability
            print("\n[Agent thinking...] (Press 'q' + Enter or Ctrl+C to cancel)")
            
            # Use threading to allow interruption
            response_queue = queue.Queue()
            interrupt_flag = threading.Event()
            cancel_queue = queue.Queue()
            
            def run_agent_query():
                """Run agent query in separate thread."""
                try:
                    result = run_query(agent, user_input)
                    if not interrupt_flag.is_set():
                        response_queue.put(("success", result))
                except Exception as e:
                    if not interrupt_flag.is_set():
                        response_queue.put(("error", str(e)))
            
            # Start agent query in background thread
            agent_thread = threading.Thread(target=run_agent_query, daemon=True)
            agent_thread.start()
            
            # Monitor for 'q' + Enter in background (non-blocking)
            def monitor_cancel():
                """Monitor for 'q' + Enter to cancel."""
                try:
                    # This will be consumed by the next input() call, but we check it first
                    import select
                    if sys.stdin.isatty():
                        # Only try non-blocking read if we're in a TTY
                        if select.select([sys.stdin], [], [], 0)[0]:
                            line = sys.stdin.readline()
                            if line.strip().lower() == 'q':
                                interrupt_flag.set()
                                cancel_queue.put("q")
                except:
                    # Fallback: if select doesn't work, just continue
                    pass
            
            # Wait for agent with periodic checks
            # User can press Ctrl+C (KeyboardInterrupt) or 'q' + Enter to cancel
            interrupted = False
            try:
                # Wait for agent with periodic checks
                while agent_thread.is_alive():
                    agent_thread.join(timeout=0.5)  # Check every 0.5 seconds
                    if not response_queue.empty():
                        break
                    # Check for 'q' cancel
                    try:
                        if not cancel_queue.empty():
                            cancel_queue.get_nowait()
                            interrupted = True
                            print("\n\n⚠️  Interrupted by user ('q'). Cancelling agent query...")
                            break
                    except queue.Empty:
                        pass
                    # Print a dot to show it's still thinking
                    print(".", end="", flush=True)
                    # Try to monitor for cancel (non-blocking)
                    monitor_cancel()
                
                print()  # New line after dots
            except KeyboardInterrupt:
                interrupt_flag.set()
                interrupted = True
                print("\n\n⚠️  Interrupted by user (Ctrl+C). Cancelling agent query...")
            
            if interrupted:
                print("  Query cancelled. You can enter a new prompt.\n")
                continue
            
            # Get result from queue
            try:
                result_type, result_value = response_queue.get(timeout=0.1)
                if result_type == "error":
                    print(f"\n✗ Error: {result_value}\n")
                    continue
                else:
                    response = result_value
            except queue.Empty:
                # Agent still running, wait for it
                agent_thread.join(timeout=10.0)
                try:
                    result_type, result_value = response_queue.get(timeout=0.1)
                    if result_type == "error":
                        print(f"\n✗ Error: {result_value}\n")
                        continue
                    else:
                        response = result_value
                except queue.Empty:
                    print("\n⚠️  Agent is taking longer than expected.")
                    print("  You can press Ctrl+C to force cancel, or wait for completion.\n")
                    # Wait for completion
                    agent_thread.join()
                    try:
                        result_type, result_value = response_queue.get(timeout=1.0)
                        if result_type == "error":
                            print(f"\n✗ Error: {result_value}\n")
                            continue
                        else:
                            response = result_value
                    except:
                        print("\n⚠️  No response from agent. Please try again.\n")
                        continue
            
            # Format and display response
            format_response(response)
            
            # Store in history (optional)
            conversation_history.append({
                "user": user_input,
                "agent": response
            })
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")
            print("Please try again or type 'quit' to exit.\n")


if __name__ == "__main__":
    chat_loop()
