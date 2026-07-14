# Lume
Lume is an adaptive reading assistant. It uses eye tracking, facial signals, and language models to help people while they read. The application follows the paragraph that the user is reading. When it detects signs of difficulty, it shows a short and simple explanation without interrupting the reading session.

## Features
Organizes text into clear and readable paragraphs
Tracks the user's gaze through the webcam with WebGazer
Includes calibration for the gaze and the face
Analyzes geometric signals inspired by Action Units AU4 and AU7
Detects possible hesitation during reading
Provides automatic explanations when support may be useful
Lets the user request an explanation with the **Spiegami** button
Offers different themes, text sizes, and a focus mode
Supports OpenRouter and local language models

## Requirements
Node.js 18 or newer
A modern browser with webcam support
An OpenRouter API key or a local language model server that supports the OpenAI API
Permission to use the webcam in the browser

## Installation
Clone the repository and open the project directory. Then install the dependencies:

```bash
npm install
```

Create a local configuration file from the included example:

```bash
cp .env_example .env
```
Git ignores the `.env` file. Do not publish this file because it can contain private API keys.

## Model configuration
### OpenRouter
OpenRouter is the default option. Add these values to the `.env` file:

```env
VITE_OPENROUTER_API_KEY=your_api_key
VITE_USE_LOCAL_LLM=false
```

The application uses this key to contact OpenRouter.

### Local model
To use a local model, add these values to the `.env` file:

```env
VITE_USE_LOCAL_LLM=true
VLLM_MLX_BASE_URL=http://127.0.0.1:8000/v1
```

Start the local server in a separate terminal. Replace `<model>` with the name of the installed model:

```bash
vllm-mlx serve <model> --port 8000
```

You can also set `VLLM_MLX_MODEL` and `VLLM_MLX_API_KEY` when needed. If no model name is set, Lume uses the first model returned by `/v1/models`.

## Running the application
Start the development server:

```bash
npm run dev
```

Open the address shown in the terminal. The default address is usually `http://localhost:5173`.

Use these commands to build and preview the optimized version:

```bash
npm run build
npm run preview
```

## Usage
1. Click **Try now** and allow the browser to use the webcam.
2. Look at each calibration point and click it five times.
3. Keep your face relaxed and facing forward during face calibration.
4. Paste the text and click **Prepara testo**.
5. Read normally. Lume highlights the current paragraph and may show an explanation when it detects possible difficulty.
6. Click **Spiegami** next to a paragraph to request an explanation manually.

## Project structure

```text
.
├── index.html              # Application interface
├── style.css               # Styles, themes, and reading layout
├── app.js                  # Main application flow and interactions
├── faceAnalyzer.js         # Geometric analysis and face calibration
├── llmExplainer.js         # Text organization and LLM explanations
├── vite.config.js          # Vite settings and connection to the local model
├── package.json            # Project dependencies and commands
├── fonts/                  # Fonts used by the interface
└── public/
    ├── webgazer.js         # Local copy of the gaze tracking library
    └── mediapipe/          # Local MediaPipe models and runtime files
```

## Privacy and limitations
The browser processes webcam frames and facial landmarks on the user's device. The application does not send this information to the language model. Technical logs stay in memory until the user resets the session or closes the page. The application saves them to a file only when the user chooses to export them.

The application sends the entered text to the selected language model when it needs to organize the text or create an explanation. With OpenRouter, the text leaves the user's device. With a local model, the text remains on the computer that runs the model.

Lume uses experimental geometric signals inspired by AU4 and AU7. These signals do not provide a full FACS classification. They cannot identify emotions with certainty and must not be used as a diagnostic tool.

## Context
Lume is an academic prototype. Its purpose is to explore a reading interface that uses different types of input and provides help only when it may be useful.
