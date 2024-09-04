
# DoorCam

## Overview
The **DoorCam** project is designed to provide a security solution that integrates machine learning to recognize and log door entry events. It includes both the backend processing of data and the frontend for interacting with the system.

## Features
- **Real-time Door Monitoring**: Capture and process door entry data.
- **Machine Learning Integration**: Models for predicting and analyzing entry events.
- **Aggregation and Logging**: Tools for aggregating data and maintaining logs of events.
- **Training and Prediction**: Scripts to train the machine learning models and make predictions based on new data.
- **Web Interface**: A user-friendly interface to interact with the DoorCam system.

## Project Structure

```
DoorCam/
│
├── deploy/                     # Deployment configuration and scripts
│
├── doorcam/                    # Main application files
│   ├── aggregation.py          # Handles data aggregation
│   ├── models.py               # Defines and handles machine learning models
│   ├── entry_record.csv         # Log of door entry events
│   ├── client.py               # Handles client-side functionality
│   ├── predict.py              # Script to run predictions on new data
│   ├── config.yaml             # Configuration file for the system
│   ├── logs/                   # Log files
│   ├── train.py                # Script to train models
│   ├── app.py                  # Main application file (web or server interface)
│   ├── templates/              # HTML templates for web interface
│   └── utils.py                # Utility functions
│
├── tests/                      # Unit tests for the application
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
└── .git/                       # Version control metadata
```

## Getting Started

### Prerequisites
- **Python 3.8+**
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Application
1. **Train the model**:
   To train the machine learning model with available data:
   ```bash
   python doorcam/train.py
   ```

2. **Make predictions**:
   After training, you can run predictions using:
   ```bash
   python doorcam/predict.py
   ```

3. **Start the application**:
   To run the web interface:
   ```bash
   python doorcam/app.py
   ```

### Configuration
Configuration for the system is stored in `doorcam/config.yaml`. You can modify this file to adjust parameters such as model settings, logging preferences, and more.

## Testing
Unit tests are located in the `tests/` directory. Run the tests using:
```bash
python -m unittest discover tests
```

## Contributing
Feel free to open issues or submit pull requests if you would like to contribute to the project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
