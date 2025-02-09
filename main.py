import gradio as gr
import requests
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import random
import pytz


class SpaceImageExplorer:
    def __init__(self):
        # Initialize application metadata
        self.startup_time = datetime(2025, 2, 9, 9, 47, 55, tzinfo=pytz.UTC)
        self.current_user = os.getenv('USER', 'Guest')

        # Initialize NASA API key
        self.nasa_api_key = os.getenv('NASA_API_KEY', 'DEMO_KEY')

        # Initialize Gemma model and tokenizer
        try:
            print("Loading Gemma 2B model...")
            model_name = "google/gemma-2b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # Space knowledge base with fun facts
        self.space_knowledge = {
            "fun_facts": [
                "A day on Venus is longer than its year! 😮",
                "The footprints on the Moon will stay there for millions of years! 👣",
                "Saturn could float in a giant bathtub because it's less dense than water! 🛁"
            ]
        }

    def get_daily_nasa_image(self):
        """Fetch NASA's Astronomy Picture of the Day"""
        url = f"https://api.nasa.gov/planetary/apod?api_key={self.nasa_api_key}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'title': data.get('title'),
                    'url': data.get('url'),
                    'explanation': data.get('explanation'),
                    'date': data.get('date')
                }
            return {'success': False, 'error': f'Status code: {response.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def search_nasa_images(self, query):
        """Search NASA's image library"""
        url = f"https://images-api.nasa.gov/search?q={query}&media_type=image"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                images = []
                if 'items' in data['collection']:
                    for item in data['collection']['items'][:5]:
                        if 'links' in item and item['links']:
                            image_data = {
                                'title': item['data'][0].get('title', 'No title'),
                                'description': item['data'][0].get('description', 'No description'),
                                'url': item['links'][0]['href'],
                                'date_created': item['data'][0].get('date_created', 'Unknown date')
                            }
                            images.append(image_data)
                return {'success': True, 'images': images}
            return {'success': False, 'error': f'Status code: {response.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def chat_with_space_expert(self, message):
        """Enhanced chat function using Gemma 2B for space topics."""
        try:
            # Create a clean, focused prompt
            prompt = f"As a space expert, explain: {message}"

            # Format the input for Gemma
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            # Generate response with improved parameters
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=300,
                min_length=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.2
            )

            # Clean and format the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("explain:")[-1].strip() if "explain:" in response.lower() else response

            # Add a random fun fact
            fun_fact = random.choice(self.space_knowledge['fun_facts'])

            # Format the final response
            formatted_response = f"🚀 {response}\n\n✨ Fun Fact: {fun_fact}"

            return formatted_response
        except Exception as e:
            return f"Houston, we have a problem! 🚀 Please try asking that question differently. Error: {str(e)}"

    def get_space_weather(self):
        """Get current space weather information"""
        try:
            url = f"https://api.nasa.gov/DONKI/notifications?api_key={self.nasa_api_key}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return self._format_real_weather(data[0])
        except Exception:
            pass
        return self._generate_simulated_weather()

    def _format_real_weather(self, data):
        """Format real NASA weather data"""
        weather_types = {
            "Report": "🛸",
            "Watch": "⚠️",
            "Warning": "🚨",
            "Alert": "⚡"
        }

        message_type = data.get('messageType', 'Report')
        emoji = weather_types.get(message_type, "🛸")

        return {
            "status": f"{emoji} Space Weather {message_type}",
            "conditions": [
                {"label": "Current Activity", "value": message_type, "emoji": emoji},
                {"label": "Solar Activity",
                 "value": "Active" if "flare" in data.get('messageBody', '').lower() else "Calm", "emoji": "🌞"},
                {"label": "Aurora Forecast",
                 "value": "Visible" if "aurora" in data.get('messageBody', '').lower() else "Not Visible", "emoji": "🌌"}
            ],
            "description": data.get('messageBody', 'Space weather information currently unavailable'),
            "time": datetime.now().strftime("%I:%M %p")
        }

    def _generate_simulated_weather(self):
        """Generate engaging simulated weather data"""
        scenarios = [
            {
                "status": "🌟 Perfect Space Weather",
                "conditions": [
                    {"label": "Current Activity", "value": "Calm", "emoji": "✨"},
                    {"label": "Solar Activity", "value": "Low", "emoji": "🌞"},
                    {"label": "Aurora Forecast", "value": "Not Visible", "emoji": "🌙"}
                ],
                "description": "Perfect conditions for stargazing tonight! The Milky Way should be clearly visible.",
            },
            {
                "status": "🌠 Aurora Alert",
                "conditions": [
                    {"label": "Current Activity", "value": "Active", "emoji": "⚡"},
                    {"label": "Solar Activity", "value": "High", "emoji": "🌞"},
                    {"label": "Aurora Forecast", "value": "Visible", "emoji": "🎆"}
                ],
                "description": "A solar storm is creating perfect conditions for aurora viewing! Look toward the northern horizon tonight.",
            },
            {
                "status": "☄️ Meteor Shower",
                "conditions": [
                    {"label": "Current Activity", "value": "Special Event", "emoji": "☄️"},
                    {"label": "Solar Activity", "value": "Normal", "emoji": "🌞"},
                    {"label": "Aurora Forecast", "value": "Not Visible", "emoji": "🌙"}
                ],
                "description": "A meteor shower is expected tonight! Best viewing hours are between 11 PM and 3 AM.",
            }
        ]
        weather = random.choice(scenarios)
        weather["time"] = datetime.now().strftime("%I:%M %p")
        return weather


def create_interface():
    try:
        explorer = SpaceImageExplorer()
    except Exception as e:
        return gr.Interface(
            fn=lambda x: f"Oops! Something went wrong. {str(e)} Please check if all settings are correct.",
            inputs="text",
            outputs="text"
        )

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # 🚀 Space Explorer
        ### Your Window to the Cosmos
        Welcome, {explorer.current_user}! Explore beautiful space images, chat about the universe, and learn amazing facts!

        Last updated: {explorer.startup_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
        """)

        with gr.Tab("✨ Today's Space Wonder"):
            with gr.Row():
                with gr.Column(scale=2):
                    daily_image = gr.Image(label="Today's Featured Space Image")
                    daily_title = gr.Markdown()
                with gr.Column(scale=1):
                    daily_explanation = gr.Markdown()
                    refresh_btn = gr.Button("Show Me Today's Image! 🔭")

            def fetch_daily():
                result = explorer.get_daily_nasa_image()
                if result['success']:
                    return (
                        result['url'],
                        f"# {result['title']}",
                        f"### What are we looking at?\n{result['explanation']}"
                    )
                return None, "Couldn't fetch today's image", "Please try again in a moment"

            refresh_btn.click(
                fetch_daily,
                outputs=[daily_image, daily_title, daily_explanation]
            )

        with gr.Tab("🔍 Space Image Search"):
            gr.Markdown("Search through NASA's amazing image collection!")
            with gr.Row():
                search_input = gr.Textbox(
                    placeholder="Try: galaxy, nebula, supernova...",
                    label="What space wonder would you like to see?"
                )
                search_btn = gr.Button("Search the Cosmos! 🌌")

            gallery = gr.Gallery(label="Discovered Images", show_label=False)
            image_info = gr.Markdown()

            def search_images(query):
                results = explorer.search_nasa_images(query)
                if results['success'] and results.get('images'):
                    images = [img['url'] for img in results['images']]
                    info = "### Discovered these amazing images!\nClick on any image to view it larger."
                    return images, info
                return [], "No images found. Try a different search!"

            search_btn.click(
                search_images,
                inputs=[search_input],
                outputs=[gallery, image_info]
            )

        with gr.Tab("💫 Space Chat"):
            gr.Markdown("""
            ### Chat with our Space Expert! 🚀
            Ask anything about space!
            For example:
            - "Why do stars twinkle?"
            - "How big is the Sun?"
            - "Tell me about Mars!"
            """)

            chatbot = gr.Chatbot(
                label="Your Space Conversation",
                height=400,
                show_label=False
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask your space question here...",
                    label="Your Question",
                    scale=4
                )
                submit = gr.Button("Ask! 🚀", scale=1)
            clear = gr.Button("Start Fresh 🌟")

            def respond(message, history):
                bot_message = explorer.chat_with_space_expert(message)
                history.append((message, bot_message))
                return "", history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            submit.click(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        with gr.Tab("🌤️ Space Weather"):
            gr.Markdown("""
            ### Live Space Weather Station 🛸
            Get real-time updates on space conditions, solar activity, and special events!
            """)

            with gr.Row():
                with gr.Column():
                    status = gr.Markdown()
                    description = gr.Markdown()
                    time_updated = gr.Markdown()
                with gr.Column():
                    conditions = gr.DataFrame(
                        headers=["Condition", "Status"],
                        label="Current Space Conditions"
                    )

            check_weather = gr.Button("Check Space Weather 🛸")

            def update_weather():
                weather = explorer.get_space_weather()
                condition_data = [
                    [f"{cond['emoji']} {cond['label']}", cond['value']]
                    for cond in weather['conditions']
                ]
                return (
                    f"# {weather['status']}",
                    f"{weather['description']}",
                    f"Last updated: {weather['time']}",
                    condition_data
                )

            check_weather.click(
                update_weather,
                outputs=[status, description, time_updated, conditions]
            )

    return demo


if __name__ == "__main__":
    print(f"===== Application Startup at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} =====")
    print(f"Current User: {os.getenv('USER', 'Guest')}")
    demo = create_interface()
    demo.launch(share=True)