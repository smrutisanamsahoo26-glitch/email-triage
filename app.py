import requests
import gradio as gr

# ✅ Your HF Space URL (same space calling backend)
BASE_URL = "https://smrutisanam-email-triage-env.hf.space"


# ---------------------------
# Smart Action Generator
# ---------------------------
def smart_action(email):
    email_lower = email.lower()

    if "refund" in email_lower or "charged" in email_lower:
        return {
            "response_text": "We apologize for the inconvenience. We will reverse the duplicate charge and confirm your refund shortly.",
            "category": "billing",
            "priority": 4
        }

    elif "password" in email_lower or "reset" in email_lower:
        return {
            "response_text": "Please try using the latest reset link, check your spam folder, and clear your browser cache.",
            "category": "technical",
            "priority": 3
        }

    elif "blocked" in email_lower or "can't access" in email_lower:
        return {
            "response_text": "Please reset your password and verify your email. If the issue persists, contact support.",
            "category": "technical",
            "priority": 5
        }

    elif "unsubscribe" in email_lower or "emails" in email_lower:
        return {
            "response_text": "You can unsubscribe using the link provided.",
            "category": "general",
            "priority": 2
        }

    return {
        "response_text": "Thank you for reaching out. We will review your issue and get back to you shortly.",
        "category": "general",
        "priority": 2
    }


# ---------------------------
# Main Function
# ---------------------------
def analyze_email(email):
    try:
        # Reset env
        reset_res = requests.post(f"{BASE_URL}/reset")

        if reset_res.status_code != 200:
            return ("Error", "-", "Reset failed", "-")

        action = smart_action(email)

        # IMPORTANT: action wrapper
        step_res = requests.post(
            f"{BASE_URL}/step",
            json={"action": action}
        )

        if step_res.status_code != 200:
            return ("Error", "-", step_res.text, "-")

        result = step_res.json()

        return (
            action["category"],
            action["priority"],
            action["response_text"],
            result.get("reward", "N/A"),
        )

    except Exception as e:
        return ("Error", "-", str(e), "-")


# ---------------------------
# UI
# ---------------------------
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