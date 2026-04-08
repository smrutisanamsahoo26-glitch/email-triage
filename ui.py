import requests
import gradio as gr

BASE_URL = "http://127.0.0.1:7860"

def analyze_email(email):
    try:
        # 1. Reset environment
        reset_res = requests.post(f"{BASE_URL}/reset")
        obs = reset_res.json()["observation"]

        # 2. Create action (you can later improve this with AI)
        def smart_action(email):
            email_lower = email.lower()

            if "refund" in email_lower or "charged" in email_lower:
                return {
                    "response_text": "We will reverse the duplicate charge and confirm your refund.",
                    "category": "billing",
                    "priority": 4
                }

            elif "password" in email_lower:
                return {
                    "response_text": "Please try resetting your password again and check your spam folder.",
                    "category": "technical",
                    "priority": 3
                }

            elif "blocked" in email_lower:
                return {
                    "response_text": "Please reset your password and verify your email to regain access.",
                    "category": "technical",
                    "priority": 5
                }

            return {
                "response_text": "We will look into your issue and get back to you.",
                "category": "general",
                "priority": 2
            }
        action = smart_action(email)

        # 3. Send step request
        step_res = requests.post(
            f"{BASE_URL}/step",
            json={"action": action}
        )

        result = step_res.json()

        return {
            "email_input": email,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}


demo = gr.Interface(
    fn=analyze_email,
    inputs=gr.Textbox(lines=10, placeholder="Paste email here..."),
    outputs="json",
    title="📧 Email Triage AI",
    description="Classify emails, assign priority, and generate responses"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)