# 2048 Deep Learning AI

A deep learning project that teaches an AI to play the popular game 2048. The AI learns and improves its gameplay strategies through reinforcement learning.

![2048 Game Interface](project-screenshot.png)

## Overview

This project implements a deep learning model that learns to play 2048 through experience. The AI uses reinforcement learning techniques to develop strategies for combining tiles and achieving high scores.

## Technologies Used

- **Backend:**
  - Python (Deep Learning & Game Logic)
  - Django (Web Framework)
  - SQLite (Database)

- **Frontend:**
  - HTML
  - CSS
  - JavaScript
  - Ajax (Asynchronous Communications)

## Features

- Interactive 2048 game interface
- Deep learning model that learns from gameplay
- Real-time visualization of AI decision-making
- Performance tracking and statistics (in dev)
- Training mode and play mode

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/2048-deep-learning.git
cd 2048-deep-learning
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
python manage.py migrate
```

5. Start the development server:
```bash
cd deep2048web/
python manage.py runserver
```

## How It Works

1. **Game Engine**: Implementation of 2048 game rules and mechanics
2. **Deep Learning Model**: Neural network architecture for game state evaluation
3. **Training Process**: Reinforcement learning, prediction algorithm for improving gameplay
4. **Web Interface**: User interface for interacting with the AI and visualizing its decisions

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## Contact

For questions and support, please open an issue in the GitHub repository or contact florian.demartini.dev@gmail.com.
