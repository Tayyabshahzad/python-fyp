# How to Run This Project (Easy Steps)

This folder contains 14 Python files (`1.py` to `14.py`). Each file makes one picture (graph) and saves it as a PNG image. You do **not** need to know how to code — just follow the steps below.

---

## Step 1: Install Python

1. Go to this website: https://www.python.org/downloads/
2. Click the big yellow **"Download Python"** button.
3. Open the downloaded file to install it.
4. **Important:** On the first install screen, check the box that says **"Add Python to PATH"** before clicking Install.

---

## Step 2: Unzip the Project Folder

1. Right-click the `Project.zip` file.
2. Click **"Extract All..."**
3. Choose a location (like Desktop) and click **Extract**.
4. Open the extracted folder — you should see files like `1.py`, `2.py`, ... `14.py`.

---

## Step 3: Open the Command Window in This Folder

1. Open the extracted folder in File Explorer.
2. Click on the address bar at the top (where the folder path is written).
3. Type `cmd` and press **Enter**. A black window will open — this is the Command Prompt.

---

## Step 4: Install the Required Tools (one-time only)

In the black Command Prompt window, type this and press **Enter**:

```
pip install numpy matplotlib scipy
```

Wait for it to finish (it may take a minute). This installs the tools the scripts need.

---

## Step 5: Run a File

To run the first file, type:

```
python 1.py
```

and press **Enter**. A picture window may pop up — that's the graph. It also gets saved automatically as a PNG image in the same folder.

Close the picture window, then run the next one the same way:

```
python 2.py
python 3.py
python 4.py
```

...and so on, up to `python 14.py`.

---

## Where Are the Results?

After running a file, look in the same folder — a new image file (for example `bilayer_MoS2_fig1_matched.png`) will appear or be updated. That is the output picture.

---

## If Something Goes Wrong

- **"python is not recognized"** → Python wasn't added to PATH. Reinstall Python and make sure to check "Add Python to PATH" during setup.
- **"No module named numpy" (or matplotlib/scipy)** → Run the Step 4 command again: `pip install numpy matplotlib scipy`
- Any other error → take a screenshot of the black window and send it back for help.
