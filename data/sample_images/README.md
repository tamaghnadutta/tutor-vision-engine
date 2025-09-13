# Sample Images

This directory contains the real sample data provided for the Error Detection API assignment.

## Structure

```
sample_images/
├── questions/          # Question images (Q1-Q4)
│   ├── Q1.jpeg        # Probability problem (Bayes' theorem)
│   ├── Q2.jpeg        # Trigonometry problem (angle of elevation)
│   ├── Q3.jpeg        # Algebra problem (quadratic equation)
│   └── Q4.jpeg        # Complex numbers problem
├── attempts/           # Student attempt images (Attempt1-4)
│   ├── Attempt1.jpeg  # Student solution for Q1 (probability)
│   ├── Attempt2.jpeg  # Student solution for Q2 (trigonometry)
│   ├── Attempt3.jpeg  # Student solution for Q3 (algebra)
│   └── Attempt4.jpeg  # Student solution for Q4 (complex numbers)
└── README.md          # This file
```

## Problem-Attempt Mapping

| Question | Topic | Student Attempt | Error Analysis |
|----------|-------|----------------|----------------|
| Q1.jpeg | Probability (Bayes' theorem) | Attempt1.jpeg | ✅ Appears correct |
| Q2.jpeg | Trigonometry (angle of elevation) | Attempt2.jpeg | ❌ Arithmetic error in tan calculation |
| Q3.jpeg | Algebra (quadratic equation) | Attempt3.jpeg | ❌ Missing negative solution |
| Q4.jpeg | Complex numbers | Attempt4.jpeg | ❌ Calculation error in modulus |

## Usage

These images are used by:
- `scripts/create_dataset.py` - Generate evaluation dataset
- `scripts/demo.py` - Demonstrate error detection
- `scripts/run_eval.py` - Evaluate model performance

The images represent real handwritten mathematical solutions that provide authentic test cases for the Error Detection API.