import gradio as gr
from env import EmailTriagerEnvironment

# ✅ Direct backend (NO API)
env = EmailTriagerEnvironment()


def smart_action(email):
    email_lower = email.lower()

    if "refund" in email_lower or "charged" in email_lower:
        return {
            "response_text": "We apologize for the inconvenience. We will reverse the duplicate charge shortly.",
            "category": "billing",
            "priority": 4
        }

    elif "password" in email_lower or "reset" in email_lower:
        return {
            "response_text": "Please use the latest reset link and check spam folder.",
            "category": "technical",
            "priority": 3
        }

    elif "blocked" in email_lower or "can't access" in email_lower:
        return {
            "response_text": "Try resetting password and verifying email.",
            "category": "technical",
            "priority": 5
        }

    elif "unsubscribe" in email_lower:
        return {
            "response_text": "You can unsubscribe using the link provided.",
            "category": "general",
            "priority": 2
        }

    return {
        "response_text": "We will review your issue and get back to you.",
        "category": "general",
        "priority": 2
    }


def analyze_email(email):
    try:
        env.reset()
        action = smart_action(email)
        result = env.step(action)

        return (
            action["category"],
            action["priority"],
            action["response_text"],
            result.get("reward", "N/A"),
        )

    except Exception as e:
        return ("Error", "-", str(e), "-")


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📧 Email Triage AI")
    gr.Markdown("Classify emails, assign priority, and generate responses instantly.")

    email_input = gr.Textbox(lines=8, placeholder="Paste customer email here...")
    btn = gr.Button("Analyze Email")

    category = gr.Textbox(label="Category")
    priority = gr.Textbox(label="Priority")
    response = gr.Textbox(label="Generated Response")
    score = gr.Textbox(label="Score")

    btn.click(
        fn=analyze_email,
        inputs=email_input,
        outputs=[category, priority, response, score]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)