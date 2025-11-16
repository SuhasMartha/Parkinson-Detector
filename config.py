# Model paths - CUSTOMIZED FOR YOUR REPOSITORY
MODEL_CONFIG = {
    "MRI": {
        "path": "models/mri_model.h5",
        "input_shape": (224, 224, 3),
        "description": "MRI brain scan CNN model"
    },
    "DRAWING": {
        "path": "models/drawing_model.h5",
        "input_shape": (224, 224, 1),
        "description": "Spiral drawing CNN model"
    },
    "SPEECH": {
        "path": "models/speech_model.pkl",
        "description": "Speech feature ML model"
    },
    "GAIT": {
        "path": "models/gait_model.pkl",
        "description": "Gait analysis ML model"
    }
}

# Speech features (22 total)
SPEECH_FEATURES = [
    'MDVP:Fo(Hz)',
    'MDVP:Fhi(Hz)',
    'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)',
    'MDVP:RAP',
    'MDVP:PPQ',
    'Jitter:DDP',
    'MDVP:Shimmer',
    'MDVP:Shimmer(dB)',
    'Shimmer:APQ3',
    'Shimmer:APQ5',
    'MDVP:APQ',
    'Shimmer:DDA',
    'NHR',
    'HNR',
    'RPDE',
    'D2',
    'DFA',
    'spread1',
    'spread2',
    'PPE'
]

# Gait parameters
GAIT_PARAMETERS = [
    'stride_length',
    'gait_speed',
    'cadence',
    'step_width',
    'swing_phase',
    'stance_phase'
]

# Thresholds
CONFIDENCE_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.7
