import requests
import gradio as gr

# Use HF internal URL (IMPORTANT)
BASE_URL = "http://127.0.0.1:7860"

def analyze_email(email):
    try:
        # Call your existing API
        res = requests.get(f"{BASE_URL}/tasks")
        tasks = res.json()

        return {
            "input_email": email,
            "message": "Backend connected ✅",
            "tasks_available": len(tasks)
        }

    except Exception as e:
        return {"error": str(e)}

demo = gr.Interface(
    fn=analyze_email,
    inputs=gr.Textbox(lines=10, placeholder="Paste email here..."),
    outputs="json",
    title="📧 Email Triage AI",
    description="Analyze emails using your deployed backend"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)