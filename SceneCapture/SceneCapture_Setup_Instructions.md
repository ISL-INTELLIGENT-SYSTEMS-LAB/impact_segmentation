
# 🔧 Setup Instructions for SceneCapture

## 🔽 Download the SAM Model

Download the model file (**2.4GB**):  
**[SAM Model - sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**

Place the downloaded file into the `EXPERIMENTS_SCENECAPTURE` folder.

---

## 🧱 Set Up Environment in VSCode

1. **Open the Command Palette**  
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)

2. **Create a New Conda Environment**  
   - Search for `Python: Create Environment`  
   - Select:  
     > *Conda - Creates a `.conda` Conda environment in the current workspace*

3. **Select Python Version**  
   - Choose: `Python 3.12.9`

4. **Install Required Python Packages**  
   In the terminal, run:
   ```bash
   python -m pip install opencv-python numpy pillow matplotlib segment-anything ttkbootstrap
   ```

5. **Install PyTorch**

   - **For Windows:**
     ```bash
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
     ```

   - **For Linux:**
     ```bash
     pip3 install torch torchvision torchaudio
     ```

---

## ▶️ Run the Application

Run the main script:
```bash
python sceneCapture.py
```
