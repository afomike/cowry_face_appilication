# Cowry Face — Biometric Authentication for the Cowry Card System

## Overview

The Cowry card is Lagos's contactless payment card for BRT buses. The current system deducts fares on tap — but it does nothing to verify *who* is tapping. Anyone holding the card can use it.

This project adds a facial authentication layer on top of the existing tap flow. When a card is tapped, the terminal retrieves the registered cardholder's passport photo and confirms the person presenting the card matches the stored identity.

The security model is straightforward: **something you have** (the card) + **something you are** (your face).

---

## Features

- Flask web server that handles card tap events
- Face recognition pipeline using `face_recognition` and `dlib`
- Passport photo retrieval and display on tap
- Real-time identity verification against stored facial encodings
- Lightweight front-end UI for terminal display

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | Python 3.10 / Flask |
| Face recognition | `face_recognition` 1.3.0, `dlib` 19.24.2 |
| Image processing | `opencv-python-headless`, `numpy` 1.26.4 |
| Front-end | HTML / Jinja2 templates |
| Photo storage | Local `Photo/` directory |

---

## Project Structure

```
cowry-face/
│
├── app.py                  # Flask app entry point
├── test.py                 # Test script for core routes and recognition logic
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── templates/
│   └── index.html          # Terminal-facing UI
└── Photo/                  # Cardholder passport photos
```

---

## Prerequisites

- Python 3.8 or higher
- `cmake` (required to build `dlib`)
- A C++ compiler (GCC on Linux/macOS, MSVC on Windows)

Install system dependencies on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential
```

---

## Setup

**1. Clone the repository**

```bash
git clone <repo-url>
cd cowry-face
```

**2. Create and activate a virtual environment**

With Conda:

```bash
conda create -n cowry_face python=3.10
conda activate cowry_face
```

Or with venv:

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> `dlib` takes a few minutes to compile from source. This is expected.

---

## Running the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser.

---

## Running Tests

```bash
python test.py
```

Or do a quick sanity check on the server:

```bash
curl http://127.0.0.1:5000/
```

---

## Adding Cardholder Photos

Place passport photos in the `Photo/` folder. Name each file to match the card ID or user identifier used by your system (e.g., `CARD_001.jpg`). The recognition pipeline will load and encode them on startup.

---

## How It Works

1. Rider taps Cowry card on the terminal.
2. The app looks up the card ID and retrieves the matching photo from `Photo/`.
3. `face_recognition` generates a facial encoding from the stored photo.
4. The terminal displays the photo for visual or automated comparison.
5. Identity is confirmed or flagged based on the match result.

---

## Roadmap

- [ ] Live camera feed integration for real-time face capture at tap
- [ ] Automated match scoring with configurable confidence threshold
- [ ] Encrypted photo storage and secure encoding cache
- [ ] Audit logging for every authentication event
- [ ] REST API endpoints for integration with Cowry card backend
- [ ] Handling for edge cases: poor lighting, occlusion, card mismatches

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push and open a pull request

Include a short description of your change and the steps needed to test it.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
