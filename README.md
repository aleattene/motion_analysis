
# Motion Analysis Project

This project demonstrates motion analysis by detecting and calculating angles of human joints in real-time using a webcam. 
It utilizes OpenCV and MediaPipe for pose estimation.

## Getting Started

Follow these instructions to get the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**

   To get started with the project, first, clone the repository to your local machine.

   ```bash
   git clone https://github.com/aleattene/motion_analysis.git
   ```

2. **Set up a virtual environment**

   Navigate to the project directory and create a virtual environment.

   ```bash
   python3 -m venv motion_analysis_venv
   ```

3. **Activate the virtual environment**

   On Windows:
   ```bash
   .\motion_analysis_venv\Scripts\activate
   ```
   On macOS and Linux:
   ```bash
   source motion_analysis_venv/bin/activate
   ```

4. **Install dependencies**

   Install all the required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

To run the application, execute the main script:

```bash
python3 main.py
```

### Usage

Once the application starts, it will use your webcam to detect and display the angles of your movements in real-time. 

**To exit the application, press the `q` key. 
Upon exiting, the last detected angle will be printed to the console.**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

---
